from io import BytesIO
from typing import Any
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from analysis import build_analysis_report, classify_variables

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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    session_id: str
    expression: str = Field(default="df", description="pandas 표현식. df는 업로드 데이터")


class SessionRequest(BaseModel):
    session_id: str


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


@app.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
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

    session_id = str(uuid.uuid4())
    ensure_datetime_columns(df)
    _sessions[session_id] = df
    _analysis_cache.pop(session_id, None)

    return dataframe_to_table_payload(
        df,
        session_id=session_id,
        expression="df",
        filename=file.filename,
    )


@app.post("/query")
def query_dataframe(body: QueryRequest):
    df = get_session_df(body.session_id)
    result = evaluate_pandas_expression(df, body.expression)
    expr = (body.expression or "df").strip() or "df"

    payload = dataframe_to_table_payload(result, expression=expr)
    payload["session_id"] = body.session_id
    return payload


@app.post("/analyze")
def auto_analyze(body: SessionRequest):
    df = get_session_df(body.session_id)
    report = build_analysis_report(df)
    report["session_id"] = body.session_id
    _analysis_cache[body.session_id] = report
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
    max_groups: int = 5


@app.post("/distribution/meta")
def distribution_meta(body: DistributionMetaRequest):
    df = get_session_df(body.session_id)
    dt_cols = ensure_datetime_columns(df)

    variables = classify_variables(df)
    numeric_cols = variables["numeric"]
    # group-by로는 datetime도 포함(사용자가 원하면 선택)
    categorical_cols = [*variables["categorical"], *variables["datetime"]]

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

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": dt_cols,
        "date_bounds": date_bounds,
    }


def _parse_date_or_none(value: str | None):
    if value is None or not str(value).strip():
        return None
    return pd.to_datetime(value, errors="coerce")


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

    in_date = df_work[date_col]
    start_dt = _parse_date_or_none(body.date_start)
    end_dt = _parse_date_or_none(body.date_end)

    if start_dt is None:
        start_dt = in_date.min()
    if end_dt is None:
        end_dt = in_date.max()

    if pd.isna(start_dt) or pd.isna(end_dt):
        raise HTTPException(
            status_code=400,
            detail=f"{date_col}에서 유효한 날짜를 찾지 못했습니다.",
        )

    # date input은 날짜 단위이므로 end를 inclusive로 처리
    end_exclusive = end_dt + pd.Timedelta(days=1)
    mask = (in_date >= start_dt) & (in_date < end_exclusive)
    df_work = df_work.loc[mask]

    # numeric 결측 제거
    df_work = df_work.loc[df_work[body.numeric_col].notna()]

    group_cols = [c for c in body.group_cols if c in df_work.columns]
    max_groups = max(1, int(body.max_groups))
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

        # count 기준 상위 그룹만 반환
        records.sort(key=lambda r: r["count"], reverse=True)
        records = records[:max_groups]

    return {
        "date_col": date_col,
        "numeric_col": body.numeric_col,
        "group_cols": group_cols,
        "date_start": str(start_dt.date()) if hasattr(start_dt, "date") else str(start_dt),
        "date_end": str(end_dt.date()) if hasattr(end_dt, "date") else str(end_dt),
        "groups": records,
        "filtered_rows": int(len(df_work)),
    }
