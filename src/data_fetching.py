from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf


def _normalize_history_dataframe(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df

    normalized_df = history_df.copy()
    if isinstance(normalized_df.columns, pd.MultiIndex):
        normalized_df.columns = normalized_df.columns.get_level_values(0)

    normalized_df = normalized_df.reset_index()
    if "Date" not in normalized_df.columns:
        date_column = normalized_df.columns[0]
        normalized_df = normalized_df.rename(columns={date_column: "Date"})

    normalized_df["Date"] = pd.to_datetime(normalized_df["Date"]).dt.tz_localize(None)
    normalized_df = normalized_df.sort_values("Date").reset_index(drop=True)
    return normalized_df


def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    history_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    return _normalize_history_dataframe(history_df)


def get_batch_price_history(
    tickers: List[str], period: str = "1y", interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    market_data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        normalized_ticker = ticker.strip().upper()
        market_data[normalized_ticker] = get_price_history(
            normalized_ticker,
            period=period,
            interval=interval,
        )

    return market_data


def extract_latest_price(history_df: pd.DataFrame) -> float:
    if history_df.empty:
        return float("nan")
    return float(history_df.iloc[-1]["Close"])


def extract_previous_close(history_df: pd.DataFrame) -> float:
    if len(history_df) < 2:
        return float("nan")
    return float(history_df.iloc[-2]["Close"])


def extract_week_ago_close(history_df: pd.DataFrame) -> float:
    if history_df.empty:
        return float("nan")

    latest_date = history_df.iloc[-1]["Date"]
    target_date = latest_date - pd.Timedelta(days=7)
    eligible_rows = history_df.loc[history_df["Date"] <= target_date]

    if eligible_rows.empty:
        if len(history_df) < 2:
            return float("nan")
        return float(history_df.iloc[0]["Close"])

    return float(eligible_rows.iloc[-1]["Close"])


def extract_52_week_high(history_df: pd.DataFrame) -> Tuple[float, str]:
    if history_df.empty:
        return float("nan"), ""

    high_index = history_df["High"].idxmax()
    high_row = history_df.loc[high_index]
    return float(high_row["High"]), high_row["Date"].date().isoformat()


def extract_52_week_low(history_df: pd.DataFrame) -> Tuple[float, str]:
    if history_df.empty:
        return float("nan"), ""

    low_index = history_df["Low"].idxmin()
    low_row = history_df.loc[low_index]
    return float(low_row["Low"]), low_row["Date"].date().isoformat()


def build_market_snapshot(history_df: pd.DataFrame) -> dict:
    latest_price = extract_latest_price(history_df)
    previous_close = extract_previous_close(history_df)
    week_ago_close = extract_week_ago_close(history_df)
    fifty_two_week_high, fifty_two_week_high_date = extract_52_week_high(history_df)
    fifty_two_week_low, fifty_two_week_low_date = extract_52_week_low(history_df)

    latest_date = ""
    if not history_df.empty:
        latest_date = history_df.iloc[-1]["Date"].date().isoformat()

    return {
        "current_price": latest_price,
        "prev_close": previous_close,
        "week_ago_close": week_ago_close,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_high_date": fifty_two_week_high_date,
        "fifty_two_week_low": fifty_two_week_low,
        "fifty_two_week_low_date": fifty_two_week_low_date,
        "latest_market_date": latest_date,
        "history_rows": len(history_df),
    }