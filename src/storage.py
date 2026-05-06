from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

DATA_DIR = Path("data")
HOLDINGS_PATH = DATA_DIR / "holdings.csv"
ORIGINAL_HOLDINGS_PATH = DATA_DIR / "original_holdings.csv"
PORTFOLIO_STATE_PATH = DATA_DIR / "portfolio_state.csv"
REMINDERS_PATH = DATA_DIR / "reminders.csv"
PERFORMANCE_PATH = DATA_DIR / "performance.csv"

HOLDINGS_COLUMNS = [
    "ticker",
    "shares",
    "avg_buy_price",
    "opened_date",
    "notes",
    "active",
    "position_type",
]

PORTFOLIO_STATE_COLUMNS = [
    "free_cash",
    "updated_at",
]

REMINDERS_COLUMNS = [
    "ticker",
    "reminder_type",
    "message",
    "created_at",
    "status",
    "target_price",
    "target_condition",
    "linked_from_action",
    "priority",
    "resolved_at",
    "notes",
]

PERFORMANCE_COLUMNS = [
    "snapshot_date",
    "theoretical_portfolio_value",
    "actual_equity_value",
    "free_cash",
    "actual_total_portfolio_value",
    "strategy_edge_value",
    "strategy_edge_pct",
    "holdings_count",
    "notes",
]


def _ensure_columns(df: pd.DataFrame, expected_columns: List[str], table_name: str) -> None:
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def _build_empty_frame(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def initialize_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not HOLDINGS_PATH.exists():
        _build_empty_frame(HOLDINGS_COLUMNS).to_csv(HOLDINGS_PATH, index=False)

    if not ORIGINAL_HOLDINGS_PATH.exists():
        _build_empty_frame(HOLDINGS_COLUMNS).to_csv(ORIGINAL_HOLDINGS_PATH, index=False)

    if not PORTFOLIO_STATE_PATH.exists():
        pd.DataFrame([{"free_cash": 0.0, "updated_at": ""}], columns=PORTFOLIO_STATE_COLUMNS).to_csv(
            PORTFOLIO_STATE_PATH, index=False
        )

    if not REMINDERS_PATH.exists():
        _build_empty_frame(REMINDERS_COLUMNS).to_csv(REMINDERS_PATH, index=False)

    if not PERFORMANCE_PATH.exists():
        _build_empty_frame(PERFORMANCE_COLUMNS).to_csv(PERFORMANCE_PATH, index=False)


def _read_csv(path: str | Path, expected_columns: List[str], table_name: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return _build_empty_frame(expected_columns)

    df = pd.read_csv(csv_path)
    if df.empty and list(df.columns) == []:
        return _build_empty_frame(expected_columns)

    _ensure_columns(df, expected_columns, table_name)
    return df


def _normalize_ticker_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _normalize_boolean_series(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def validate_holdings(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []

    try:
        _ensure_columns(df, HOLDINGS_COLUMNS, "holdings")
    except ValueError as exc:
        return [str(exc)]

    if df.empty:
        return errors

    working_df = df.copy()
    working_df["ticker"] = _normalize_ticker_series(working_df["ticker"])
    working_df["shares"] = pd.to_numeric(working_df["shares"], errors="coerce")
    working_df["avg_buy_price"] = pd.to_numeric(working_df["avg_buy_price"], errors="coerce")
    working_df["active"] = _normalize_boolean_series(working_df["active"])
    working_df["position_type"] = (
        working_df["position_type"].fillna("").astype(str).str.strip().str.lower()
    )

    invalid_tickers = working_df["ticker"].eq("") | working_df["ticker"].eq("NAN")
    if invalid_tickers.any():
        errors.append("Holdings contain blank ticker values.")

    monitor_mask = working_df["position_type"].eq("monitor")
    regular_mask = ~monitor_mask

    if working_df.loc[regular_mask, "shares"].isna().any():
        errors.append("Non-monitor holdings must contain numeric share counts.")
    elif (working_df.loc[regular_mask, "shares"] <= 0).any():
        errors.append("Non-monitor holdings must have shares greater than 0.")

    if working_df.loc[regular_mask, "avg_buy_price"].isna().any():
        errors.append("Non-monitor holdings must contain numeric average buy prices.")
    elif (working_df.loc[regular_mask, "avg_buy_price"] < 0).any():
        errors.append("Non-monitor holdings must have non-negative average buy prices.")

    if (working_df.loc[monitor_mask, "shares"].dropna() <= 0).any():
        errors.append("Monitor rows with share values must have shares greater than 0.")

    if (working_df.loc[monitor_mask, "avg_buy_price"].dropna() < 0).any():
        errors.append("Monitor rows with average buy prices must have non-negative values.")

    active_duplicates = (
        working_df.loc[working_df["active"], "ticker"].value_counts().loc[lambda counts: counts > 1]
    )
    if not active_duplicates.empty:
        duplicate_list = ", ".join(active_duplicates.index.tolist())
        errors.append(f"Only one active holding row is allowed per ticker. Duplicates: {duplicate_list}")

    return errors


def load_holdings(path: str | Path = HOLDINGS_PATH) -> pd.DataFrame:
    df = _read_csv(path, HOLDINGS_COLUMNS, "holdings")
    if df.empty:
        return _build_empty_frame(HOLDINGS_COLUMNS)

    normalized_df = df.copy()
    normalized_df["ticker"] = _normalize_ticker_series(normalized_df["ticker"])
    normalized_df["shares"] = pd.to_numeric(normalized_df["shares"], errors="coerce")
    normalized_df["avg_buy_price"] = pd.to_numeric(normalized_df["avg_buy_price"], errors="coerce")
    normalized_df["active"] = _normalize_boolean_series(normalized_df["active"])
    normalized_df["opened_date"] = normalized_df["opened_date"].fillna("").astype(str)
    normalized_df["notes"] = normalized_df["notes"].fillna("").astype(str)
    normalized_df["position_type"] = (
        normalized_df["position_type"].fillna("").astype(str).str.strip().str.lower()
    )

    errors = validate_holdings(normalized_df)
    if errors:
        raise ValueError(" ; ".join(errors))

    return normalized_df.loc[normalized_df["active"]].reset_index(drop=True)


def load_original_holdings(path: str | Path = ORIGINAL_HOLDINGS_PATH) -> pd.DataFrame:
    return load_holdings(path)


def save_holdings(df: pd.DataFrame, path: str | Path = HOLDINGS_PATH) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    working_df = df.copy()
    _ensure_columns(working_df, HOLDINGS_COLUMNS, "holdings")
    working_df["ticker"] = _normalize_ticker_series(working_df["ticker"])
    working_df["active"] = _normalize_boolean_series(working_df["active"])
    working_df["position_type"] = (
        working_df["position_type"].fillna("").astype(str).str.strip().str.lower()
    )

    errors = validate_holdings(working_df)
    if errors:
        raise ValueError(" ; ".join(errors))

    working_df.to_csv(csv_path, index=False)


def save_original_holdings(df: pd.DataFrame, path: str | Path = ORIGINAL_HOLDINGS_PATH) -> None:
    save_holdings(df, path)


def load_portfolio_state(path: str | Path = PORTFOLIO_STATE_PATH) -> pd.DataFrame:
    df = _read_csv(path, PORTFOLIO_STATE_COLUMNS, "portfolio_state")
    if df.empty:
        return pd.DataFrame([{"free_cash": 0.0, "updated_at": ""}], columns=PORTFOLIO_STATE_COLUMNS)

    normalized_df = df.copy()
    normalized_df["free_cash"] = pd.to_numeric(normalized_df["free_cash"], errors="coerce").fillna(0.0)
    normalized_df["updated_at"] = normalized_df["updated_at"].fillna("").astype(str)
    return normalized_df.reset_index(drop=True)


def save_portfolio_state(df: pd.DataFrame, path: str | Path = PORTFOLIO_STATE_PATH) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    working_df = df.copy()
    _ensure_columns(working_df, PORTFOLIO_STATE_COLUMNS, "portfolio_state")
    working_df["free_cash"] = pd.to_numeric(working_df["free_cash"], errors="coerce").fillna(0.0)
    working_df["updated_at"] = working_df["updated_at"].fillna("").astype(str)
    working_df.to_csv(csv_path, index=False)


def load_reminders(path: str | Path = REMINDERS_PATH) -> pd.DataFrame:
    df = _read_csv(path, REMINDERS_COLUMNS, "reminders")
    if df.empty:
        return _build_empty_frame(REMINDERS_COLUMNS)

    normalized_df = df.copy()
    normalized_df["ticker"] = _normalize_ticker_series(normalized_df["ticker"])
    normalized_df["reminder_type"] = normalized_df["reminder_type"].fillna("").astype(str)
    normalized_df["message"] = normalized_df["message"].fillna("").astype(str)
    normalized_df["created_at"] = normalized_df["created_at"].fillna("").astype(str)
    normalized_df["status"] = normalized_df["status"].fillna("open").astype(str).str.strip().str.lower()
    normalized_df["target_price"] = pd.to_numeric(normalized_df["target_price"], errors="coerce")
    normalized_df["target_condition"] = normalized_df["target_condition"].fillna("").astype(str)
    normalized_df["linked_from_action"] = normalized_df["linked_from_action"].fillna("").astype(str)
    normalized_df["priority"] = normalized_df["priority"].fillna("medium").astype(str).str.strip().str.lower()
    normalized_df["resolved_at"] = normalized_df["resolved_at"].fillna("").astype(str)
    normalized_df["notes"] = normalized_df["notes"].fillna("").astype(str)

    return normalized_df.reset_index(drop=True)


def save_reminders(df: pd.DataFrame, path: str | Path = REMINDERS_PATH) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    working_df = df.copy()
    _ensure_columns(working_df, REMINDERS_COLUMNS, "reminders")
    if not working_df.empty:
        working_df["ticker"] = _normalize_ticker_series(working_df["ticker"])
        working_df["status"] = working_df["status"].fillna("open").astype(str).str.strip().str.lower()
        working_df["priority"] = working_df["priority"].fillna("medium").astype(str).str.strip().str.lower()

    working_df.to_csv(csv_path, index=False)


def load_performance(path: str | Path = PERFORMANCE_PATH) -> pd.DataFrame:
    df = _read_csv(path, PERFORMANCE_COLUMNS, "performance")
    if df.empty:
        return _build_empty_frame(PERFORMANCE_COLUMNS)

    normalized_df = df.copy()
    numeric_columns = [
        "theoretical_portfolio_value",
        "actual_equity_value",
        "free_cash",
        "actual_total_portfolio_value",
        "strategy_edge_value",
        "strategy_edge_pct",
        "holdings_count",
    ]
    for column in numeric_columns:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    text_columns = ["snapshot_date", "notes"]
    for column in text_columns:
        normalized_df[column] = normalized_df[column].fillna("").astype(str)

    return normalized_df.reset_index(drop=True)


def save_performance(df: pd.DataFrame, path: str | Path = PERFORMANCE_PATH) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    working_df = df.copy()
    _ensure_columns(working_df, PERFORMANCE_COLUMNS, "performance")
    working_df.to_csv(csv_path, index=False)