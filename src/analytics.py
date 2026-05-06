from __future__ import annotations

from typing import Dict

import pandas as pd

NEAR_HIGH_THRESHOLD_PCT = 3.0
NEAR_LOW_THRESHOLD_PCT = 3.0
AVERAGE_WEEK_THRESHOLD_PCT = 2.0
BIG_DAILY_JUMP_THRESHOLD_PCT = 5.0


def _safe_pct_change(current_value: float, reference_value: float) -> float:
    if pd.isna(current_value) or pd.isna(reference_value) or reference_value == 0:
        return float("nan")
    return ((current_value - reference_value) / reference_value) * 100.0


def calculate_daily_change_pct(current_price: float, prev_close: float) -> float:
    return _safe_pct_change(current_price, prev_close)


def calculate_weekly_change_pct(current_price: float, week_ago_close: float) -> float:
    return _safe_pct_change(current_price, week_ago_close)


def calculate_distance_to_high_pct(current_price: float, high_52: float) -> float:
    if pd.isna(current_price) or pd.isna(high_52) or high_52 == 0:
        return float("nan")
    return ((high_52 - current_price) / high_52) * 100.0


def calculate_distance_to_low_pct(current_price: float, low_52: float) -> float:
    if pd.isna(current_price) or pd.isna(low_52) or low_52 == 0:
        return float("nan")
    return ((current_price - low_52) / low_52) * 100.0


def flag_near_52_week_high(distance_pct: float, threshold: float = NEAR_HIGH_THRESHOLD_PCT) -> bool:
    return not pd.isna(distance_pct) and distance_pct <= threshold


def flag_near_52_week_low(distance_pct: float, threshold: float = NEAR_LOW_THRESHOLD_PCT) -> bool:
    return not pd.isna(distance_pct) and distance_pct <= threshold


def flag_spike_after_flat_week(
    weekly_change_pct: float,
    daily_change_pct: float,
    weekly_threshold: float = AVERAGE_WEEK_THRESHOLD_PCT,
    daily_threshold: float = BIG_DAILY_JUMP_THRESHOLD_PCT,
) -> bool:
    if pd.isna(weekly_change_pct) or pd.isna(daily_change_pct):
        return False
    return abs(weekly_change_pct) <= weekly_threshold and daily_change_pct >= daily_threshold


def build_flag_summary(row: pd.Series) -> str:
    flags = []

    if bool(row.get("flag_near_high", False)):
        flags.append("Near 52-week high")
    if bool(row.get("flag_near_low", False)):
        flags.append("Near 52-week low")
    if bool(row.get("flag_spike_after_flat_week", False)):
        flags.append("Big daily jump after flat week")

    return ", ".join(flags)


def calculate_position_metrics(holding_row: pd.Series, market_snapshot: dict) -> dict:
    shares = pd.to_numeric(pd.Series([holding_row.get("shares")]), errors="coerce").iloc[0]
    avg_buy_price = pd.to_numeric(pd.Series([holding_row.get("avg_buy_price")]), errors="coerce").iloc[0]
    position_type = str(holding_row.get("position_type", "")).strip().lower()

    current_price = market_snapshot.get("current_price", float("nan"))
    prev_close = market_snapshot.get("prev_close", float("nan"))
    week_ago_close = market_snapshot.get("week_ago_close", float("nan"))
    fifty_two_week_high = market_snapshot.get("fifty_two_week_high", float("nan"))
    fifty_two_week_low = market_snapshot.get("fifty_two_week_low", float("nan"))

    has_position_values = not pd.isna(shares) and not pd.isna(avg_buy_price)
    market_value = shares * current_price if has_position_values and not pd.isna(current_price) else float("nan")
    cost_basis = shares * avg_buy_price if has_position_values else float("nan")
    unrealized_pl = (
        market_value - cost_basis if not pd.isna(market_value) and not pd.isna(cost_basis) else float("nan")
    )
    unrealized_pl_pct = _safe_pct_change(market_value, cost_basis)

    daily_change_pct = calculate_daily_change_pct(current_price, prev_close)
    weekly_change_pct = calculate_weekly_change_pct(current_price, week_ago_close)
    distance_to_52w_high_pct = calculate_distance_to_high_pct(current_price, fifty_two_week_high)
    distance_to_52w_low_pct = calculate_distance_to_low_pct(current_price, fifty_two_week_low)

    flag_high = flag_near_52_week_high(distance_to_52w_high_pct)
    flag_low = flag_near_52_week_low(distance_to_52w_low_pct)
    flag_spike = flag_spike_after_flat_week(weekly_change_pct, daily_change_pct)

    metrics = {
        "ticker": holding_row["ticker"],
        "position_type": position_type,
        "shares": shares,
        "avg_buy_price": avg_buy_price,
        "current_price": current_price,
        "market_value": market_value,
        "cost_basis": cost_basis,
        "unrealized_pl": unrealized_pl,
        "unrealized_pl_pct": unrealized_pl_pct,
        "prev_close": prev_close,
        "daily_change_pct": daily_change_pct,
        "week_ago_close": week_ago_close,
        "weekly_change_pct": weekly_change_pct,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_high_date": market_snapshot.get("fifty_two_week_high_date", ""),
        "fifty_two_week_low": fifty_two_week_low,
        "fifty_two_week_low_date": market_snapshot.get("fifty_two_week_low_date", ""),
        "distance_to_52w_high_pct": distance_to_52w_high_pct,
        "distance_to_52w_low_pct": distance_to_52w_low_pct,
        "flag_near_high": flag_high,
        "flag_near_low": flag_low,
        "flag_spike_after_flat_week": flag_spike,
        "latest_market_date": market_snapshot.get("latest_market_date", ""),
        "history_rows": market_snapshot.get("history_rows", 0),
    }
    metrics["flag_summary"] = build_flag_summary(pd.Series(metrics))
    return metrics


def build_holdings_analytics(
    holdings_df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    analytics_rows = []

    for _, holding_row in holdings_df.iterrows():
        ticker = str(holding_row["ticker"]).strip().upper()
        history_df = market_data.get(ticker, pd.DataFrame())

        market_snapshot = {
            "current_price": float("nan"),
            "prev_close": float("nan"),
            "week_ago_close": float("nan"),
            "fifty_two_week_high": float("nan"),
            "fifty_two_week_high_date": "",
            "fifty_two_week_low": float("nan"),
            "fifty_two_week_low_date": "",
            "latest_market_date": "",
            "history_rows": 0,
        }

        if not history_df.empty:
            latest_price = float(history_df.iloc[-1]["Close"])
            prev_close = float(history_df.iloc[-2]["Close"]) if len(history_df) >= 2 else float("nan")

            latest_date = history_df.iloc[-1]["Date"]
            target_date = latest_date - pd.Timedelta(days=7)
            eligible_rows = history_df.loc[history_df["Date"] <= target_date]
            if eligible_rows.empty:
                week_ago_close = float(history_df.iloc[0]["Close"])
            else:
                week_ago_close = float(eligible_rows.iloc[-1]["Close"])

            high_index = history_df["High"].idxmax()
            high_row = history_df.loc[high_index]
            low_index = history_df["Low"].idxmin()
            low_row = history_df.loc[low_index]

            market_snapshot = {
                "current_price": latest_price,
                "prev_close": prev_close,
                "week_ago_close": week_ago_close,
                "fifty_two_week_high": float(high_row["High"]),
                "fifty_two_week_high_date": high_row["Date"].date().isoformat(),
                "fifty_two_week_low": float(low_row["Low"]),
                "fifty_two_week_low_date": low_row["Date"].date().isoformat(),
                "latest_market_date": latest_date.date().isoformat(),
                "history_rows": len(history_df),
            }

        analytics_rows.append(calculate_position_metrics(holding_row, market_snapshot))

    analytics_df = pd.DataFrame(analytics_rows)
    if analytics_df.empty:
        return analytics_df

    return analytics_df.sort_values(
        by=["flag_near_high", "flag_near_low", "flag_spike_after_flat_week", "ticker"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def summarize_portfolio(analytics_df: pd.DataFrame) -> dict:
    if analytics_df.empty:
        return {
            "portfolio_value": 0.0,
            "total_cost_basis": 0.0,
            "total_unrealized_pl": 0.0,
            "total_unrealized_pl_pct": float("nan"),
            "flagged_count": 0,
            "positions_count": 0,
        }

    portfolio_value = float(analytics_df["market_value"].sum(skipna=True))
    total_cost_basis = float(analytics_df["cost_basis"].sum(skipna=True))
    total_unrealized_pl = float(analytics_df["unrealized_pl"].sum(skipna=True))
    total_unrealized_pl_pct = _safe_pct_change(portfolio_value, total_cost_basis)

    flag_columns = ["flag_near_high", "flag_near_low", "flag_spike_after_flat_week"]
    available_flag_columns = [column for column in flag_columns if column in analytics_df.columns]

    if available_flag_columns:
        flagged_mask = analytics_df[available_flag_columns].fillna(False).any(axis=1)
    elif "flag_summary" in analytics_df.columns:
        flagged_mask = analytics_df["flag_summary"].fillna("").astype(str).str.strip().ne("")
    else:
        flagged_mask = pd.Series(False, index=analytics_df.index)

    return {
        "portfolio_value": portfolio_value,
        "total_cost_basis": total_cost_basis,
        "total_unrealized_pl": total_unrealized_pl,
        "total_unrealized_pl_pct": total_unrealized_pl_pct,
        "flagged_count": int(flagged_mask.sum()),
        "positions_count": int(len(analytics_df)),
    }


def calculate_theoretical_vs_actual_value(
    actual_analytics_df: pd.DataFrame,
    original_analytics_df: pd.DataFrame,
    free_cash: float,
) -> dict:
    actual_equity_value = float(actual_analytics_df["market_value"].sum(skipna=True)) if not actual_analytics_df.empty else 0.0
    original_equity_value = (
        float(original_analytics_df["market_value"].sum(skipna=True)) if not original_analytics_df.empty else 0.0
    )
    actual_total_value = actual_equity_value + (0.0 if pd.isna(free_cash) else float(free_cash))

    return {
        "theoretical_portfolio_value": original_equity_value,
        "actual_equity_value": actual_equity_value,
        "free_cash": 0.0 if pd.isna(free_cash) else float(free_cash),
        "actual_portfolio_value": actual_total_value,
        "strategy_value_add": actual_total_value - original_equity_value,
        "strategy_value_add_pct": _safe_pct_change(actual_total_value, original_equity_value),
    }


def summarize_performance(performance_df: pd.DataFrame, current_portfolio_value: float) -> dict:
    if performance_df.empty:
        return {
            "baseline_value": float("nan"),
            "latest_snapshot_value": float("nan"),
            "current_portfolio_value": current_portfolio_value,
            "change_since_baseline": float("nan"),
            "change_since_baseline_pct": float("nan"),
            "realized_pl": float("nan"),
        }

    baseline_rows = performance_df.loc[
        performance_df["snapshot_type"].astype(str).str.strip().str.lower() == "baseline"
    ]
    baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else performance_df.iloc[0]
    latest_row = performance_df.iloc[-1]

    baseline_value = float(baseline_row["portfolio_value"]) if pd.notna(baseline_row["portfolio_value"]) else float("nan")
    latest_snapshot_value = (
        float(latest_row["portfolio_value"]) if pd.notna(latest_row["portfolio_value"]) else float("nan")
    )
    realized_pl = float(latest_row["realized_pl"]) if pd.notna(latest_row["realized_pl"]) else float("nan")
    change_since_baseline = (
        current_portfolio_value - baseline_value if not pd.isna(current_portfolio_value) and not pd.isna(baseline_value) else float("nan")
    )
    change_since_baseline_pct = _safe_pct_change(current_portfolio_value, baseline_value)

    return {
        "baseline_value": baseline_value,
        "latest_snapshot_value": latest_snapshot_value,
        "current_portfolio_value": current_portfolio_value,
        "change_since_baseline": change_since_baseline,
        "change_since_baseline_pct": change_since_baseline_pct,
        "realized_pl": realized_pl,
    }