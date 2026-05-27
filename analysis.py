from typing import Any

import pandas as pd


def _try_parse_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.8


def classify_variables(df: pd.DataFrame) -> dict[str, list[str]]:
    datetime_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        s = df[col]
        name = str(col)

        if pd.api.types.is_datetime64_any_dtype(s) or _try_parse_datetime(s):
            datetime_cols.append(name)
        elif pd.api.types.is_bool_dtype(s) or isinstance(
            s.dtype, pd.CategoricalDtype
        ):
            categorical_cols.append(name)
        elif pd.api.types.is_numeric_dtype(s):
            n = len(s)
            nunique = int(s.nunique(dropna=True))
            # 행이 충분히 많을 때만 저카디널리티 수치를 범주형으로 처리
            if (
                n > 50
                and nunique <= 30
                and nunique / max(n, 1) < 0.1
            ):
                categorical_cols.append(name)
            else:
                numeric_cols.append(name)
        else:
            categorical_cols.append(name)

    return {
        "datetime": datetime_cols,
        "categorical": categorical_cols,
        "numeric": numeric_cols,
    }


def _format_describe_value(val: Any) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def describe_to_table(series: pd.Series) -> dict[str, Any]:
    try:
        desc = series.describe()
    except Exception:
        desc = pd.Series({"count": series.count()})

    rows = [[str(stat), _format_describe_value(val)] for stat, val in desc.items()]
    return {"columns": ["statistic", "value"], "rows": rows}


def dataframe_describe_to_table(desc_df: pd.DataFrame) -> dict[str, Any]:
    desc_df = desc_df.copy()
    desc_df.index = desc_df.index.astype(str)
    desc_df.columns = [str(c) for c in desc_df.columns]

    columns = ["variable", *desc_df.columns.tolist()]
    rows = []
    for idx, row in desc_df.iterrows():
        rows.append([str(idx), *[_format_describe_value(v) for v in row.values]])

    return {"columns": columns, "rows": rows}


def build_analysis_report(df: pd.DataFrame) -> dict[str, Any]:
    variables = classify_variables(df)
    describes: dict[str, dict[str, Any]] = {}

    for group in ("datetime", "categorical", "numeric"):
        for col_name in variables[group]:
            if col_name in df.columns:
                describes[col_name] = describe_to_table(df[col_name])

    numeric_summary: dict[str, Any] | None = None
    numeric_cols = [c for c in variables["numeric"] if c in df.columns]
    if numeric_cols:
        numeric_summary = dataframe_describe_to_table(
            df[numeric_cols].describe().T
        )

    return {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "variables": variables,
        "describes": describes,
        "numeric_summary": numeric_summary,
    }
