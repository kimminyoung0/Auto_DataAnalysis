from io import BytesIO
from typing import Any
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from analysis import build_analysis_report, classify_variables
from storage import store

app = FastAPI(title="Auto Data Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

MAX_ROWS_RETURN = 10_000

_sessions: dict[str, pd.DataFrame] = {}
_analysis_cache: dict[str, dict[str, Any]] = {}
_session_refs: dict[str, dict[str, str]] = {}  # session_id -> project_id, analysis_id

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    session_id: str
    expression: str = Field(default="df", description="pandas 표현식. df는 업로드 데이터")
    date_col: str | None = None
    date_start: str | None = None
    date_end: str | None = None


class SessionRequest(BaseModel):
    session_id: str


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class AnalysisStateRequest(BaseModel):
    expression: str = "df"
    target_col: str | None = None
    date_col: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    date_filter_active: bool = False


class RenameAnalysisRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class RenameProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


def _persist_analysis_state(session_id: str, state: AnalysisStateRequest) -> None:
    ref = _session_refs.get(session_id)
    if not ref:
        return
    store.update_analysis(
        ref["project_id"],
        ref["analysis_id"],
        expression=state.expression,
        target_col=state.target_col,
        date_col=state.date_col,
        date_start=state.date_start,
        date_end=state.date_end,
        date_filter_active=state.date_filter_active,
    )


def _load_analysis_into_session(
    project_id: str, analysis_id: str
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    meta = store.get_analysis(project_id, analysis_id)
    df = store.load_dataframe(project_id, analysis_id)
    dt_cols = ensure_datetime_columns(df)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = df
    _session_refs[session_id] = {
        "project_id": project_id,
        "analysis_id": analysis_id,
    }
    store.update_analysis(project_id, analysis_id, session_id=session_id)

    report = store.load_report(project_id, analysis_id)
    if report is not None:
        report = dict(report)
        report["session_id"] = session_id
        _analysis_cache[session_id] = report
    else:
        _analysis_cache.pop(session_id, None)

    meta["session_id"] = session_id
    meta["datetime_cols"] = dt_cols
    meta["date_bounds"] = build_date_bounds(df, dt_cols)
    if report is not None:
        meta["analysis_report"] = report
    return df, meta, session_id


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        return pd.read_csv(BytesIO(content))
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(content))

    raise HTTPException(
        status_code=400,
        detail="CSV 또는 Excel(.xlsx, .xls)만 지원합니다.",
    )


def dataframe_to_table_payload(
    df: pd.DataFrame,
    *,
    session_id: str | None = None,
    expression: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    truncated = len(df) > MAX_ROWS_RETURN
    display_df = df.head(MAX_ROWS_RETURN) if truncated else df

    display_df = display_df.fillna("")
    columns = [str(c) for c in display_df.columns.tolist()]
    # to_numpy + str 변환 (대용량에서 astype(str).values.tolist() 보다 가벼움)
    rows = [
        [str(v) for v in row]
        for row in display_df.to_numpy(dtype=object)
    ]

    payload: dict[str, Any] = {
        "columns": columns,
        "rows": rows,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "display_rows": int(len(rows)),
        "truncated": truncated,
        "max_rows_returned": MAX_ROWS_RETURN,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if expression is not None:
        payload["expression"] = expression
    if filename is not None:
        payload["filename"] = filename
    return payload


def get_session_df(session_id: str) -> pd.DataFrame:
    df = _sessions.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="세션이 없습니다. 파일을 다시 업로드하세요.")
    return df


def _ensure_datetime_column(df: pd.DataFrame, col: str) -> None:
    """세션 내에서 특정 컬럼을 datetime으로 변환(필요 시)"""
    if col not in df.columns:
        return
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return
    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)


def ensure_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    업로드된 데이터에서 날짜/시간 컬럼을 자동 탐지하고,
    해당 컬럼들을 datetime으로 변환해 둔다.
    """
    variables = classify_variables(df)
    dt_cols = variables.get("datetime", []) or []
    for c in dt_cols:
        _ensure_datetime_column(df, c)
    return dt_cols


def build_date_bounds(df: pd.DataFrame, dt_cols: list[str]) -> dict[str, dict[str, str | None]]:
    date_bounds: dict[str, dict[str, str | None]] = {}
    for c in dt_cols:
        s = df[c].dropna()
        if s.empty:
            date_bounds[c] = {"min": None, "max": None}
        else:
            date_bounds[c] = {
                "min": pd.to_datetime(s.min()).date().isoformat(),
                "max": pd.to_datetime(s.max()).date().isoformat(),
            }
    return date_bounds


def _parse_date_or_none(value: str | None):
    if value is None or not str(value).strip():
        return None
    return pd.to_datetime(value, errors="coerce")


def filter_df_by_date(
    df: pd.DataFrame,
    date_col: str | None,
    date_start: str | None,
    date_end: str | None,
) -> pd.DataFrame:
    if not date_col:
        return df
    if date_col not in df.columns:
        raise HTTPException(status_code=400, detail="date_col을 찾을 수 없습니다.")

    _ensure_datetime_column(df, date_col)
    series = df[date_col]
    start_dt = _parse_date_or_none(date_start)
    end_dt = _parse_date_or_none(date_end)

    if start_dt is None:
        start_dt = series.min()
    if end_dt is None:
        end_dt = series.max()

    if pd.isna(start_dt) or pd.isna(end_dt):
        raise HTTPException(
            status_code=400,
            detail=f"{date_col}에서 유효한 날짜를 찾지 못했습니다.",
        )

    end_exclusive = end_dt + pd.Timedelta(days=1)
    mask = (series >= start_dt) & (series < end_exclusive)
    return df.loc[mask]


def evaluate_pandas_expression(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    expr = (expression or "df").strip()
    if not expr:
        expr = "df"

    lowered = expr.lower()
    if "import" in lowered or "__" in expr:
        raise HTTPException(status_code=400, detail="허용되지 않는 표현입니다.")

    namespace = {"df": df, "pd": pd, "np": np}
    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"pandas 실행 오류: {e}") from e

    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        if result.name:
            return result.to_frame()
        return result.reset_index()

    raise HTTPException(
        status_code=400,
        detail="결과가 DataFrame 또는 Series여야 합니다. 예: df.head(10)",
    )


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/projects")
def list_projects():
    return {"projects": store.list_projects()}


@app.post("/api/projects")
def create_project(body: CreateProjectRequest):
    try:
        project = store.create_project(body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return project


@app.get("/api/projects/{project_id}")
def get_project_detail(project_id: str):
    try:
        project = store.get_project(project_id)
        analyses = store.list_analyses(project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"project": project, "analyses": analyses}


@app.patch("/api/projects/{project_id}/rename")
def rename_project(project_id: str, body: RenameProjectRequest):
    try:
        meta = store.rename_project(project_id, body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return meta


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    try:
        store.delete_project(project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"ok": True}


@app.post("/api/projects/{project_id}/analyses/{analysis_id}/load")
def load_saved_analysis(project_id: str, analysis_id: str):
    try:
        df, meta, session_id = _load_analysis_into_session(project_id, analysis_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    expr = meta.get("expression") or "df"
    if meta.get("date_filter_active") and meta.get("date_col"):
        view = filter_df_by_date(
            df,
            meta.get("date_col"),
            meta.get("date_start"),
            meta.get("date_end"),
        )
    else:
        view = df

    try:
        result = evaluate_pandas_expression(view, expr)
    except HTTPException:
        result = view
        expr = "df"

    payload = dataframe_to_table_payload(
        result,
        session_id=session_id,
        expression=expr,
        filename=meta.get("filename"),
    )
    payload["project_id"] = project_id
    payload["analysis_id"] = analysis_id
    payload["analysis_title"] = meta.get("title")
    payload["datetime_cols"] = meta.get("datetime_cols", [])
    payload["date_bounds"] = meta.get("date_bounds", {})
    payload["saved_state"] = {
        "expression": meta.get("expression", "df"),
        "target_col": meta.get("target_col"),
        "date_col": meta.get("date_col"),
        "date_start": meta.get("date_start"),
        "date_end": meta.get("date_end"),
        "date_filter_active": bool(meta.get("date_filter_active")),
    }
    if meta.get("analysis_report"):
        payload["analysis_report"] = meta["analysis_report"]
    return payload


@app.patch("/api/projects/{project_id}/analyses/{analysis_id}/state")
def patch_analysis_state(
    project_id: str, analysis_id: str, body: AnalysisStateRequest
):
    try:
        meta = store.update_analysis(
            project_id,
            analysis_id,
            expression=body.expression,
            target_col=body.target_col,
            date_col=body.date_col,
            date_start=body.date_start,
            date_end=body.date_end,
            date_filter_active=body.date_filter_active,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return meta


@app.patch("/api/projects/{project_id}/analyses/{analysis_id}/rename")
def rename_analysis(project_id: str, analysis_id: str, body: RenameAnalysisRequest):
    try:
        meta = store.update_analysis(project_id, analysis_id, title=body.title.strip())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return meta


@app.delete("/api/projects/{project_id}/analyses/{analysis_id}")
def delete_analysis(project_id: str, analysis_id: str):
    try:
        store.delete_analysis(project_id, analysis_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"ok": True}


@app.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    project_id: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")

    try:
        df = read_uploaded_file(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 읽기 실패: {e}") from e

    if df.empty:
        raise HTTPException(status_code=400, detail="데이터가 비어 있습니다.")

    project_id = project_id.strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id가 필요합니다.")

    try:
        store.get_project(project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    session_id = str(uuid.uuid4())
    dt_cols = ensure_datetime_columns(df)
    _sessions[session_id] = df
    _analysis_cache.pop(session_id, None)

    try:
        analysis = store.create_analysis(
            project_id,
            filename=file.filename or "data.csv",
            session_id=session_id,
        )
        store.save_dataframe(project_id, analysis["id"], df)
        store.update_analysis(
            project_id,
            analysis["id"],
            shape={"rows": int(len(df)), "columns": int(len(df.columns))},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 저장 실패: {e}") from e

    _session_refs[session_id] = {
        "project_id": project_id,
        "analysis_id": analysis["id"],
    }

    payload = dataframe_to_table_payload(
        df,
        session_id=session_id,
        expression="df",
        filename=file.filename,
    )
    payload["project_id"] = project_id
    payload["analysis_id"] = analysis["id"]
    payload["analysis_title"] = analysis["title"]
    payload["saved_state"] = {
        "expression": "df",
        "target_col": None,
        "date_col": None,
        "date_start": None,
        "date_end": None,
        "date_filter_active": False,
    }
    payload["datetime_cols"] = dt_cols
    payload["date_bounds"] = build_date_bounds(df, dt_cols)
    return payload


@app.post("/query")
def query_dataframe(body: QueryRequest):
    df = get_session_df(body.session_id)
    df = filter_df_by_date(df, body.date_col, body.date_start, body.date_end)
    result = evaluate_pandas_expression(df, body.expression)
    expr = (body.expression or "df").strip() or "df"

    payload = dataframe_to_table_payload(result, expression=expr)
    payload["session_id"] = body.session_id
    if body.date_col:
        payload["date_col"] = body.date_col
        payload["date_filtered"] = True
    return payload


@app.post("/analyze")
def auto_analyze(body: SessionRequest):
    df = get_session_df(body.session_id)
    report = build_analysis_report(df)
    report["session_id"] = body.session_id
    _analysis_cache[body.session_id] = report

    ref = _session_refs.get(body.session_id)
    if ref:
        store.save_report(ref["project_id"], ref["analysis_id"], report)
        store.update_analysis(
            ref["project_id"],
            ref["analysis_id"],
            has_report=True,
            shape=report.get("shape"),
        )

    return report


@app.get("/analyze/{session_id}")
def get_analysis(session_id: str):
    report = _analysis_cache.get(session_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="자동분석 결과가 없습니다. Data 탭에서 자동분석을 실행하세요.",
        )
    return report


class DistributionMetaRequest(BaseModel):
    session_id: str


class DistributionDataRequest(BaseModel):
    session_id: str
    date_col: str | None = None
    numeric_col: str
    group_cols: list[str] = []
    date_start: str | None = None  # YYYY-MM-DD
    date_end: str | None = None  # YYYY-MM-DD


@app.post("/distribution/meta")
def distribution_meta(body: DistributionMetaRequest):
    df = get_session_df(body.session_id)
    dt_cols = ensure_datetime_columns(df)

    variables = classify_variables(df)
    numeric_cols = variables["numeric"]
    # group-by로는 datetime도 포함(사용자가 원하면 선택)
    categorical_cols = [*variables["categorical"], *variables["datetime"]]

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": dt_cols,
        "date_bounds": build_date_bounds(df, dt_cols),
    }


class NumericQualityRequest(BaseModel):
    session_id: str
    group_cols: list[str] = []
    group_values: list[str] | None = None


@app.post("/quality/numeric")
def numeric_quality(body: NumericQualityRequest):
    df = get_session_df(body.session_id)
    variables = classify_variables(df)
    numeric_cols = [c for c in variables.get("numeric", []) if c in df.columns]
    if not numeric_cols:
        return {"numeric_cols": [], "overall": [], "group_col": body.group_col, "groups": []}

    # 숫자 변환(문자열 숫자도 포함) + 결측/0 계산은 변환 결과 기준
    num_df = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        num_df[c] = pd.to_numeric(df[c], errors="coerce")

    n_rows = int(len(num_df))

    def build_overall_rows(sub: pd.DataFrame) -> list[dict[str, Any]]:
        denom = max(1, int(len(sub)))
        miss = sub.isna().sum()
        zero = (sub == 0).sum()
        rows: list[dict[str, Any]] = []
        for c in numeric_cols:
            mc = int(miss.get(c, 0))
            zc = int(zero.get(c, 0))
            rows.append(
                {
                    "col": c,
                    "missing_count": mc,
                    "missing_rate": mc / denom,
                    "zero_count": zc,
                    "zero_rate": zc / denom,
                }
            )
        return rows

    overall = build_overall_rows(num_df)

    group_cols = [c for c in (body.group_cols or []) if c in df.columns]
    group_cols = group_cols[:4]
    groups: list[dict[str, Any]] = []
    if group_cols:
        requested = body.group_values or []
        requested_set = {str(v) for v in requested if str(v).strip()}
        grouped = df.groupby(group_cols, dropna=False)
        for key, gdf in grouped:
            if isinstance(key, tuple):
                parts = [("NaN" if pd.isna(k) else str(k)) for k in key]
                label = " / ".join(parts)
            else:
                label = "NaN" if pd.isna(key) else str(key)
            if requested_set and label not in requested_set:
                continue
            idx = gdf.index
            sub = num_df.loc[idx]
            groups.append(
                {
                    "group": label,
                    "count": int(len(sub)),
                    "rows": build_overall_rows(sub),
                }
            )
        groups.sort(key=lambda x: x["count"], reverse=True)

    return {
        "rows": n_rows,
        "numeric_cols": numeric_cols,
        "overall": overall,
        "group_cols": group_cols,
        "groups": groups,
    }


class CategoricalCountRequest(BaseModel):
    session_id: str
    group_cols: list[str] = []


@app.post("/quality/categorical_counts")
def categorical_counts(body: CategoricalCountRequest):
    df = get_session_df(body.session_id)
    group_cols = [c for c in (body.group_cols or []) if c in df.columns][:10]
    if not group_cols:
        raise HTTPException(status_code=400, detail="group_cols를 1개 이상 선택하세요.")

    grouped = df.groupby(group_cols, dropna=False).size().reset_index(name="count")

    labels: list[str] = []
    counts: list[int] = []
    for _, row in grouped.iterrows():
        parts = []
        for c in group_cols:
            v = row[c]
            parts.append("NaN" if pd.isna(v) else str(v))
        labels.append(" / ".join(parts))
        counts.append(int(row["count"]))

    # count 내림차순 정렬 (표시 자체는 모두)
    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]

    return {"group_cols": group_cols, "labels": labels, "counts": counts, "total_groups": len(labels)}


@app.post("/distribution/data")
def distribution_data(body: DistributionDataRequest):
    df = get_session_df(body.session_id)
    dt_cols = ensure_datetime_columns(df)
    if not dt_cols:
        raise HTTPException(status_code=400, detail="날짜/시간 컬럼을 찾지 못했습니다.")

    date_col = body.date_col or dt_cols[0]
    if date_col not in df.columns:
        raise HTTPException(status_code=400, detail="date_col을 찾을 수 없습니다.")
    _ensure_datetime_column(df, date_col)

    if body.numeric_col not in df.columns:
        raise HTTPException(status_code=400, detail="numeric_col을 찾을 수 없습니다.")

    numeric = pd.to_numeric(df[body.numeric_col], errors="coerce")
    df_work = df.copy()
    df_work[body.numeric_col] = numeric

    start_dt = _parse_date_or_none(body.date_start)
    end_dt = _parse_date_or_none(body.date_end)

    df_work = filter_df_by_date(
        df_work, date_col, body.date_start, body.date_end
    )

    if start_dt is None:
        start_dt = _parse_date_or_none(
            str(df_work[date_col].min()) if len(df_work) else None
        )
    if end_dt is None:
        end_dt = _parse_date_or_none(
            str(df_work[date_col].max()) if len(df_work) else None
        )

    # numeric 결측 제거
    df_work = df_work.loc[df_work[body.numeric_col].notna()]

    group_cols = [c for c in body.group_cols if c in df_work.columns]
    MAX_VALUES_PER_GROUP = 5000

    records: list[dict[str, Any]] = []
    if not group_cols:
        vals = df_work[body.numeric_col].astype(float).to_numpy()
        records = [{"group": "All", "values": vals.tolist(), "count": int(len(vals))}]
    else:
        grouped = df_work.groupby(group_cols, dropna=False)
        for key, g in grouped:
            if isinstance(key, tuple):
                label = " / ".join(str(k) for k in key)
            else:
                label = str(key)

            vals = g[body.numeric_col].astype(float).dropna().to_numpy()
            original_count = int(len(vals))
            if original_count > MAX_VALUES_PER_GROUP:
                # 대용량 전송 방지: 샘플링
                rng = np.random.default_rng(0)
                idx = rng.choice(original_count, size=MAX_VALUES_PER_GROUP, replace=False)
                vals = vals[idx]

            records.append(
                {
                    "group": label,
                    "values": vals.tolist(),
                    "count": original_count,
                }
            )

        records.sort(key=lambda r: r["count"], reverse=True)

    return {
        "date_col": date_col,
        "numeric_col": body.numeric_col,
        "group_cols": group_cols,
        "date_start": str(start_dt.date()) if hasattr(start_dt, "date") else str(start_dt),
        "date_end": str(end_dt.date()) if hasattr(end_dt, "date") else str(end_dt),
        "groups": records,
        "filtered_rows": int(len(df_work)),
    }
