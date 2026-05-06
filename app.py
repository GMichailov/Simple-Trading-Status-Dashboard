from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics import (
    build_holdings_analytics,
    calculate_theoretical_vs_actual_value,
    summarize_portfolio,
)
from src.data_fetching import build_market_snapshot, get_batch_price_history
from src.storage import (
    HOLDINGS_COLUMNS,
    HOLDINGS_PATH,
    ORIGINAL_HOLDINGS_PATH,
    PORTFOLIO_STATE_COLUMNS,
    REMINDERS_COLUMNS,
    initialize_data_files,
    load_holdings,
    load_original_holdings,
    load_portfolio_state,
    load_reminders,
    save_holdings,
    save_original_holdings,
    save_portfolio_state,
    save_reminders,
)

ONBOARDING_COLUMNS = ["ticker", "shares", "avg_buy_price", "position_type"]
SELL_COLUMNS = ["ticker", "shares_sold", "cash_received"]
BUYBACK_COLUMNS = ["ticker", "shares_bought", "avg_buy_price", "cash_spent"]
DEFAULT_ANALYTICS_COLUMNS = [
    "ticker",
    "position_type",
    "shares",
    "avg_buy_price",
    "current_price",
    "daily_change_pct",
    "weekly_change_pct",
    "fifty_two_week_high",
    "fifty_two_week_high_date",
    "fifty_two_week_low",
    "fifty_two_week_low_date",
    "distance_to_52w_high_pct",
    "distance_to_52w_low_pct",
    "market_value",
    "cost_basis",
    "unrealized_pl",
    "unrealized_pl_pct",
    "flag_summary",
]


def reset_app_state() -> None:
    pd.DataFrame(columns=HOLDINGS_COLUMNS).to_csv(HOLDINGS_PATH, index=False)
    pd.DataFrame(columns=REMINDERS_COLUMNS).to_csv("data/reminders.csv", index=False)
    pd.DataFrame([{"free_cash": 0.0, "updated_at": datetime.now().isoformat()}], columns=PORTFOLIO_STATE_COLUMNS).to_csv(
        "data/portfolio_state.csv", index=False
    )

    original_holdings_path = Path(ORIGINAL_HOLDINGS_PATH)
    if original_holdings_path.exists():
        original_holdings_path.unlink()


def holdings_file_is_initialized() -> bool:
    holdings_path = Path(HOLDINGS_PATH)
    if not holdings_path.exists():
        return False

    try:
        holdings_df = pd.read_csv(holdings_path)
    except Exception:
        return False

    if holdings_df.empty:
        return False

    if "ticker" not in holdings_df.columns:
        return False

    ticker_series = holdings_df["ticker"].fillna("").astype(str).str.strip()
    return ticker_series.ne("").any()


def build_default_onboarding_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "", "shares": None, "avg_buy_price": None, "position_type": ""},
        ],
        columns=ONBOARDING_COLUMNS,
    )


def build_default_sell_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "", "shares_sold": None, "cash_received": None},
        ],
        columns=SELL_COLUMNS,
    )


def build_default_buyback_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "", "shares_bought": None, "avg_buy_price": None, "cash_spent": None},
        ],
        columns=BUYBACK_COLUMNS,
    )


def normalize_onboarding_table(editor_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = editor_df.copy()
    normalized_df["ticker"] = normalized_df["ticker"].fillna("").astype(str).str.strip().str.upper()
    normalized_df["position_type"] = (
        normalized_df["position_type"].fillna("").astype(str).str.strip().str.lower()
    )
    normalized_df["shares"] = pd.to_numeric(normalized_df["shares"], errors="coerce")
    normalized_df["avg_buy_price"] = pd.to_numeric(normalized_df["avg_buy_price"], errors="coerce")
    return normalized_df


def normalize_sell_table(editor_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = editor_df.copy()
    normalized_df["ticker"] = normalized_df["ticker"].fillna("").astype(str).str.strip().str.upper()
    normalized_df["shares_sold"] = pd.to_numeric(normalized_df["shares_sold"], errors="coerce")
    normalized_df["cash_received"] = pd.to_numeric(normalized_df["cash_received"], errors="coerce")
    return normalized_df


def normalize_buyback_table(editor_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = editor_df.copy()
    normalized_df["ticker"] = normalized_df["ticker"].fillna("").astype(str).str.strip().str.upper()
    normalized_df["shares_bought"] = pd.to_numeric(normalized_df["shares_bought"], errors="coerce")
    normalized_df["avg_buy_price"] = pd.to_numeric(normalized_df["avg_buy_price"], errors="coerce")
    normalized_df["cash_spent"] = pd.to_numeric(normalized_df["cash_spent"], errors="coerce")
    return normalized_df


def filter_blank_rows(onboarding_df: pd.DataFrame) -> pd.DataFrame:
    has_any_values = onboarding_df.fillna("").astype(str).apply(lambda column: column.str.strip()).ne("").any(axis=1)
    has_numeric_values = onboarding_df.select_dtypes(include=["number"]).notna().any(axis=1) if not onboarding_df.empty else pd.Series(dtype=bool)
    if len(has_numeric_values) == len(onboarding_df):
        meaningful_row_mask = has_any_values | has_numeric_values
    else:
        meaningful_row_mask = has_any_values
    return onboarding_df.loc[meaningful_row_mask].reset_index(drop=True)


def validate_onboarding_table(
    onboarding_df: pd.DataFrame,
    existing_tickers: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    existing_tickers = existing_tickers or set()

    if onboarding_df.empty:
        return ["Please add at least one ticker before continuing."]

    blank_tickers = onboarding_df["ticker"].eq("")
    if blank_tickers.any():
        errors.append("Every entered row must include a ticker.")

    monitor_mask = onboarding_df["position_type"].eq("monitor")
    regular_mask = ~monitor_mask

    if onboarding_df.loc[regular_mask, "shares"].isna().any():
        errors.append("Rows without 'monitor' selected must include a share count.")
    elif (onboarding_df.loc[regular_mask, "shares"] <= 0).any():
        errors.append("Share counts must be greater than 0.")

    if onboarding_df.loc[regular_mask, "avg_buy_price"].isna().any():
        errors.append("Rows without 'monitor' selected must include an average buy price.")
    elif (onboarding_df.loc[regular_mask, "avg_buy_price"] < 0).any():
        errors.append("Average buy prices must be non-negative.")

    if (onboarding_df.loc[monitor_mask, "shares"].dropna() <= 0).any():
        errors.append("Monitor rows with a share count must use a value greater than 0.")

    if (onboarding_df.loc[monitor_mask, "avg_buy_price"].dropna() < 0).any():
        errors.append("Monitor rows with an average buy price must use a non-negative value.")

    duplicate_tickers = onboarding_df["ticker"].value_counts().loc[lambda counts: counts > 1]
    if not duplicate_tickers.empty:
        errors.append(
            f"Each ticker can only appear once. Duplicates: {', '.join(duplicate_tickers.index.tolist())}"
        )

    overlapping_tickers = sorted(set(onboarding_df["ticker"]) & existing_tickers)
    if overlapping_tickers:
        errors.append(
            f"These tickers already exist and cannot be added again: {', '.join(overlapping_tickers)}"
        )

    return errors


def validate_sell_table(sell_df: pd.DataFrame, current_holdings_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if sell_df.empty:
        return ["Please add at least one sale row."]

    current_shares_by_ticker = current_holdings_df.set_index("ticker")["shares"].to_dict()

    for _, row in sell_df.iterrows():
        ticker = row["ticker"]
        shares_sold = row["shares_sold"]
        cash_received = row["cash_received"]

        if not ticker:
            errors.append("Each sale row must include a ticker.")
            continue

        if ticker not in current_shares_by_ticker:
            errors.append(f"{ticker} is not in current holdings.")
            continue

        if pd.isna(shares_sold) or shares_sold <= 0:
            errors.append(f"{ticker} must have a positive shares sold value.")
            continue

        if shares_sold > current_shares_by_ticker[ticker]:
            errors.append(f"{ticker} cannot sell more shares than are currently held.")

        if pd.isna(cash_received) or cash_received < 0:
            errors.append(f"{ticker} must have a non-negative cash received value.")

    return errors


def validate_buyback_table(buyback_df: pd.DataFrame, current_holdings_df: pd.DataFrame, free_cash: float) -> list[str]:
    errors: list[str] = []

    if buyback_df.empty:
        return ["Please add at least one buyback row."]

    current_tickers = set(current_holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().tolist())

    total_cash_spent = 0.0
    for _, row in buyback_df.iterrows():
        ticker = row["ticker"]
        shares_bought = row["shares_bought"]
        avg_buy_price = row["avg_buy_price"]
        cash_spent = row["cash_spent"]

        if not ticker:
            errors.append("Each buyback row must include a ticker.")
            continue

        if ticker not in current_tickers:
            errors.append(f"{ticker} must already exist in holdings or monitors before buyback.")
            continue

        if pd.isna(shares_bought) or shares_bought <= 0:
            errors.append(f"{ticker} must have a positive shares bought value.")

        if pd.isna(avg_buy_price) or avg_buy_price < 0:
            errors.append(f"{ticker} must have a non-negative average buy price.")

        if pd.isna(cash_spent) or cash_spent < 0:
            errors.append(f"{ticker} must have a non-negative cash spent value.")
        else:
            total_cash_spent += float(cash_spent)

    if total_cash_spent > free_cash:
        errors.append("Cash spent across buybacks cannot exceed current free cash.")

    return errors


def onboarding_to_holdings(onboarding_df: pd.DataFrame) -> pd.DataFrame:
    holdings_df = pd.DataFrame(columns=HOLDINGS_COLUMNS)
    holdings_df["ticker"] = onboarding_df["ticker"]
    holdings_df["shares"] = onboarding_df["shares"]
    holdings_df["avg_buy_price"] = onboarding_df["avg_buy_price"]
    holdings_df["opened_date"] = ""
    holdings_df["notes"] = ""
    holdings_df["active"] = True
    holdings_df["position_type"] = onboarding_df["position_type"]
    return holdings_df


def render_onboarding_page() -> None:
    st.title("Trading Dashboard Setup")
    st.write(
        "Enter the tickers you own or want to monitor. Leave the final column blank for normal holdings, "
        "or choose monitor if share count and average buy price should be optional."
    )

    if "onboarding_table_seed" not in st.session_state:
        st.session_state.onboarding_table_seed = build_default_onboarding_table()

    with st.form("onboarding_form"):
        edited_df = st.data_editor(
            st.session_state.onboarding_table_seed,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=False),
                "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=0.01, format="%.4f"),
                "avg_buy_price": st.column_config.NumberColumn(
                    "Average Buy Price",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                ),
                "position_type": st.column_config.SelectboxColumn(
                    "Mode",
                    options=["", "monitor"],
                    default="",
                    required=False,
                ),
            },
            key="holdings_onboarding_editor",
        )

        submitted = st.form_submit_button("Save holdings", type="primary", use_container_width=True)

    if submitted:
        normalized_df = normalize_onboarding_table(edited_df)
        filtered_df = filter_blank_rows(normalized_df)
        errors = validate_onboarding_table(filtered_df)

        if errors:
            for error in errors:
                st.error(error)
            return

        holdings_df = onboarding_to_holdings(filtered_df)
        save_holdings(holdings_df)
        if not Path(ORIGINAL_HOLDINGS_PATH).exists() or load_original_holdings().empty:
            save_original_holdings(holdings_df)

        save_portfolio_state(
            pd.DataFrame([{"free_cash": 0.0, "updated_at": datetime.now().isoformat()}], columns=PORTFOLIO_STATE_COLUMNS)
        )
        st.session_state.onboarding_table_seed = build_default_onboarding_table()
        st.session_state["show_main_page"] = True
        st.rerun()


def load_analytics_table(holdings_df: pd.DataFrame) -> pd.DataFrame:
    tickers = holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()
    tickers = [ticker for ticker in tickers if ticker]

    if not tickers:
        return pd.DataFrame(columns=DEFAULT_ANALYTICS_COLUMNS)

    market_data = {}
    raw_market_data = get_batch_price_history(tickers)

    for ticker, history_df in raw_market_data.items():
        if history_df.empty:
            market_data[ticker] = history_df
            continue

        build_market_snapshot(history_df)
        market_data[ticker] = history_df

    analytics_df = build_holdings_analytics(holdings_df, market_data)
    if analytics_df.empty:
        return pd.DataFrame(columns=DEFAULT_ANALYTICS_COLUMNS)

    for column in DEFAULT_ANALYTICS_COLUMNS:
        if column not in analytics_df.columns:
            analytics_df[column] = pd.NA

    return analytics_df


def render_add_stocks_section(current_holdings_df: pd.DataFrame) -> None:
    st.subheader("Add Stocks")

    if "add_stocks_table_seed" not in st.session_state:
        st.session_state.add_stocks_table_seed = build_default_onboarding_table()

    with st.expander("Add new stocks or monitors"):
        with st.form("add_stocks_form"):
            edited_df = st.data_editor(
                st.session_state.add_stocks_table_seed,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", required=False),
                    "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=0.01, format="%.4f"),
                    "avg_buy_price": st.column_config.NumberColumn(
                        "Average Buy Price",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                    ),
                    "position_type": st.column_config.SelectboxColumn(
                        "Mode",
                        options=["", "monitor"],
                        default="",
                        required=False,
                    ),
                },
                key="add_stocks_editor",
            )

            submitted = st.form_submit_button("Add to holdings", type="primary", use_container_width=True)

        if submitted:
            normalized_df = normalize_onboarding_table(edited_df)
            filtered_df = filter_blank_rows(normalized_df)
            existing_tickers = set(
                current_holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()
            )
            errors = validate_onboarding_table(filtered_df, existing_tickers=existing_tickers)

            if errors:
                for error in errors:
                    st.error(error)
                return

            new_rows_df = onboarding_to_holdings(filtered_df)
            updated_holdings_df = pd.concat([current_holdings_df, new_rows_df], ignore_index=True)
            save_holdings(updated_holdings_df)
            st.session_state.add_stocks_table_seed = build_default_onboarding_table()
            st.success("New stocks added to holdings.")
            st.rerun()


def render_sell_section(current_holdings_df: pd.DataFrame) -> None:
    st.subheader("Sell")

    if "sell_table_seed" not in st.session_state:
        st.session_state.sell_table_seed = build_default_sell_table()

    with st.expander("Record sales"):
        with st.form("sell_form"):
            edited_df = st.data_editor(
                st.session_state.sell_table_seed,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "ticker": st.column_config.SelectboxColumn(
                        "Ticker",
                        options=sorted(current_holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()),
                        required=False,
                    ),
                    "shares_sold": st.column_config.NumberColumn("Shares Sold", min_value=0.0, step=0.01, format="%.4f"),
                    "cash_received": st.column_config.NumberColumn("Cash Received", min_value=0.0, step=0.01, format="%.2f"),
                },
                key="sell_editor",
            )
            submitted = st.form_submit_button("Apply sales", type="primary", use_container_width=True)

        if submitted:
            normalized_df = normalize_sell_table(edited_df)
            filtered_df = filter_blank_rows(normalized_df)

            errors = validate_sell_table(filtered_df, current_holdings_df)
            if errors:
                for error in errors:
                    st.error(error)
                return

            updated_holdings_df = current_holdings_df.copy()
            reminders_df = load_reminders()
            portfolio_state_df = load_portfolio_state()
            free_cash = float(portfolio_state_df.iloc[0]["free_cash"]) if not portfolio_state_df.empty else 0.0

            for _, sale_row in filtered_df.iterrows():
                ticker = sale_row["ticker"]
                shares_sold = float(sale_row["shares_sold"])
                cash_received = float(sale_row["cash_received"])

                holding_index = updated_holdings_df.index[updated_holdings_df["ticker"] == ticker][0]
                updated_holdings_df.loc[holding_index, "shares"] = float(updated_holdings_df.loc[holding_index, "shares"]) - shares_sold

                if updated_holdings_df.loc[holding_index, "shares"] <= 0:
                    updated_holdings_df.loc[holding_index, "shares"] = 0.0
                    updated_holdings_df.loc[holding_index, "active"] = False

                free_cash += cash_received

                reminder_row = pd.DataFrame(
                    [
                        {
                            "ticker": ticker,
                            "reminder_type": "rebuy",
                            "message": f"Sold {shares_sold:.4f} shares and generated cash.",
                            "created_at": datetime.now().isoformat(),
                            "status": "open",
                            "target_price": pd.NA,
                            "target_condition": "Monitor for buyback opportunity",
                            "linked_from_action": "sold into spike",
                            "priority": "medium",
                            "shares_sold": shares_sold,
                            "cash_generated": cash_received,
                            "resolved_at": "",
                            "notes": "",
                        }
                    ],
                    columns=REMINDERS_COLUMNS,
                )
                reminders_df = pd.concat([reminders_df, reminder_row], ignore_index=True)

            save_holdings(updated_holdings_df)
            save_portfolio_state(
                pd.DataFrame([{"free_cash": free_cash, "updated_at": datetime.now().isoformat()}], columns=PORTFOLIO_STATE_COLUMNS)
            )
            save_reminders(reminders_df)

            st.session_state.sell_table_seed = build_default_sell_table()
            st.success("Sales recorded and reminders created.")
            st.rerun()


def render_buyback_section(current_holdings_df: pd.DataFrame) -> None:
    st.subheader("Bought Back")

    if "buyback_table_seed" not in st.session_state:
        st.session_state.buyback_table_seed = build_default_buyback_table()

    with st.expander("Record buybacks"):
        with st.form("buyback_form"):
            edited_df = st.data_editor(
                st.session_state.buyback_table_seed,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "ticker": st.column_config.SelectboxColumn(
                        "Ticker",
                        options=sorted(current_holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()),
                        required=False,
                    ),
                    "shares_bought": st.column_config.NumberColumn("Shares Bought", min_value=0.0, step=0.01, format="%.4f"),
                    "avg_buy_price": st.column_config.NumberColumn("Average Buy Price", min_value=0.0, step=0.01, format="%.2f"),
                    "cash_spent": st.column_config.NumberColumn("Cash Spent", min_value=0.0, step=0.01, format="%.2f"),
                },
                key="buyback_editor",
            )
            submitted = st.form_submit_button("Apply buybacks", type="primary", use_container_width=True)

        if submitted:
            normalized_df = normalize_buyback_table(edited_df)
            filtered_df = filter_blank_rows(normalized_df)

            portfolio_state_df = load_portfolio_state()
            free_cash = float(portfolio_state_df.iloc[0]["free_cash"]) if not portfolio_state_df.empty else 0.0

            errors = validate_buyback_table(filtered_df, current_holdings_df, free_cash)
            if errors:
                for error in errors:
                    st.error(error)
                return

            updated_holdings_df = current_holdings_df.copy()
            reminders_df = load_reminders()

            total_cash_spent = 0.0
            for _, buyback_row in filtered_df.iterrows():
                ticker = buyback_row["ticker"]
                shares_bought = float(buyback_row["shares_bought"])
                avg_buy_price = float(buyback_row["avg_buy_price"])
                cash_spent = float(buyback_row["cash_spent"])
                total_cash_spent += cash_spent

                holding_index = updated_holdings_df.index[updated_holdings_df["ticker"] == ticker][0]
                existing_shares = float(updated_holdings_df.loc[holding_index, "shares"]) if pd.notna(updated_holdings_df.loc[holding_index, "shares"]) else 0.0
                existing_avg_buy = float(updated_holdings_df.loc[holding_index, "avg_buy_price"]) if pd.notna(updated_holdings_df.loc[holding_index, "avg_buy_price"]) else 0.0

                new_total_shares = existing_shares + shares_bought
                new_total_cost = (existing_shares * existing_avg_buy) + (shares_bought * avg_buy_price)
                new_avg_buy = new_total_cost / new_total_shares if new_total_shares > 0 else avg_buy_price

                updated_holdings_df.loc[holding_index, "shares"] = new_total_shares
                updated_holdings_df.loc[holding_index, "avg_buy_price"] = new_avg_buy
                updated_holdings_df.loc[holding_index, "active"] = True
                updated_holdings_df.loc[holding_index, "position_type"] = ""

                matching_open_reminders = reminders_df.index[
                    (reminders_df["ticker"] == ticker)
                    & (reminders_df["status"].astype(str).str.lower() == "open")
                    & (reminders_df["reminder_type"].astype(str).str.lower() == "rebuy")
                ]
                if len(matching_open_reminders) > 0:
                    reminders_df.loc[matching_open_reminders, "status"] = "done"
                    reminders_df.loc[matching_open_reminders, "resolved_at"] = datetime.now().isoformat()

            free_cash -= total_cash_spent

            save_holdings(updated_holdings_df)
            save_portfolio_state(
                pd.DataFrame([{"free_cash": free_cash, "updated_at": datetime.now().isoformat()}], columns=PORTFOLIO_STATE_COLUMNS)
            )
            save_reminders(reminders_df)

            st.session_state.buyback_table_seed = build_default_buyback_table()
            st.success("Buybacks recorded and holdings updated.")
            st.rerun()


def render_portfolio_comparison(actual_analytics_df: pd.DataFrame) -> None:
    try:
        original_holdings_df = load_original_holdings()
    except Exception as exc:
        st.error(f"Could not load original holdings: {exc}")
        return

    try:
        portfolio_state_df = load_portfolio_state()
    except Exception as exc:
        st.error(f"Could not load portfolio state: {exc}")
        return

    free_cash_default = float(portfolio_state_df.iloc[0]["free_cash"]) if not portfolio_state_df.empty else 0.0
    free_cash = st.number_input("Free Cash", min_value=0.0, value=free_cash_default, step=1.0)

    if st.button("Save free cash", use_container_width=False):
        save_portfolio_state(
            pd.DataFrame(
                [{"free_cash": free_cash, "updated_at": datetime.now().isoformat()}],
                columns=PORTFOLIO_STATE_COLUMNS,
            )
        )
        st.success("Free cash updated.")

    original_analytics_df = load_analytics_table(original_holdings_df) if not original_holdings_df.empty else pd.DataFrame()
    comparison = calculate_theoretical_vs_actual_value(
        actual_analytics_df=actual_analytics_df,
        original_analytics_df=original_analytics_df,
        free_cash=free_cash,
    )

    st.subheader("Theoretical vs Actual Portfolio Value")
    comparison_columns = st.columns(4)
    comparison_columns[0].metric(
        "Theoretical Value",
        _format_currency(comparison["theoretical_portfolio_value"]),
    )
    comparison_columns[1].metric(
        "Actual Equity Value",
        _format_currency(comparison["actual_equity_value"]),
    )
    comparison_columns[2].metric(
        "Actual Total + Cash",
        _format_currency(comparison["actual_portfolio_value"]),
        f"Cash: {_format_currency(comparison['free_cash'])}",
    )
    comparison_columns[3].metric(
        "Strategy Edge",
        _format_currency(comparison["strategy_value_add"]),
        _format_percent(comparison["strategy_value_add_pct"]),
    )


def _render_focus_table(title: str, df: pd.DataFrame, help_text: str) -> None:
    st.subheader(title)
    st.caption(help_text)

    if df.empty:
        st.info("No stocks currently match this condition.")
        return

    display_columns = [
        "ticker",
        "position_type",
        "current_price",
        "daily_change_pct",
        "weekly_change_pct",
        "distance_to_52w_high_pct",
        "distance_to_52w_low_pct",
        "flag_summary",
    ]
    available_columns = [column for column in display_columns if column in df.columns]
    st.dataframe(
        df[available_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "position_type": st.column_config.TextColumn("Mode"),
            "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "daily_change_pct": st.column_config.NumberColumn("1D %", format="%.2f%%"),
            "weekly_change_pct": st.column_config.NumberColumn("7D %", format="%.2f%%"),
            "distance_to_52w_high_pct": st.column_config.NumberColumn("Dist. to High %", format="%.2f%%"),
            "distance_to_52w_low_pct": st.column_config.NumberColumn("Dist. to Low %", format="%.2f%%"),
            "flag_summary": st.column_config.TextColumn("Flags"),
        },
    )


def render_reminders_section(holdings_df: pd.DataFrame) -> None:
    st.subheader("Rebuy / Watch Reminders")

    try:
        reminders_df = load_reminders()
    except Exception as exc:
        st.error(f"Could not load reminders: {exc}")
        return

    ticker_options = sorted(holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist())

    with st.expander("Add reminder"):
        reminder_ticker = st.selectbox("Ticker", options=ticker_options, index=None, placeholder="Select ticker")
        reminder_type = st.selectbox("Reminder Type", options=["rebuy", "watch", "review", "trim"])
        reminder_message = st.text_input("Reminder")
        target_price = st.number_input("Target Price", min_value=0.0, value=0.0, step=0.01)
        target_condition = st.text_input("Target Condition")
        linked_from_action = st.selectbox(
            "Linked From Action",
            options=["", "sold into spike", "trimmed position", "manual note"],
        )
        priority = st.selectbox("Priority", options=["low", "medium", "high"])
        shares_sold = st.number_input("Shares Sold", min_value=0.0, value=0.0, step=0.01)
        cash_generated = st.number_input("Cash Generated", min_value=0.0, value=0.0, step=0.01)
        reminder_notes = st.text_area("Notes")

        if st.button("Save reminder", use_container_width=True):
            if not reminder_ticker or not reminder_message.strip():
                st.error("Ticker and reminder text are required.")
            else:
                new_row = pd.DataFrame(
                    [
                        {
                            "ticker": reminder_ticker,
                            "reminder_type": reminder_type,
                            "message": reminder_message.strip(),
                            "created_at": datetime.now().isoformat(),
                            "status": "open",
                            "target_price": target_price if target_price > 0 else pd.NA,
                            "target_condition": target_condition.strip(),
                            "linked_from_action": linked_from_action,
                            "priority": priority,
                            "shares_sold": shares_sold if shares_sold > 0 else pd.NA,
                            "cash_generated": cash_generated if cash_generated > 0 else pd.NA,
                            "resolved_at": "",
                            "notes": reminder_notes.strip(),
                        }
                    ]
                )
                updated_reminders_df = pd.concat([reminders_df, new_row], ignore_index=True)
                save_reminders(updated_reminders_df)
                st.success("Reminder saved.")
                st.rerun()

    if reminders_df.empty:
        st.info("No reminders yet.")
        return

    open_reminders_df = reminders_df.loc[reminders_df["status"].astype(str).str.lower() == "open"].copy()
    if open_reminders_df.empty:
        st.info("No open reminders.")
        return

    for row_index, row in open_reminders_df.iterrows():
        columns = st.columns([2, 1, 3, 2, 2, 2, 1, 1])
        columns[0].write(f"**{row['ticker']}**")
        columns[1].write(str(row["reminder_type"]))
        columns[2].write(str(row["message"]))
        columns[3].write(
            f"Target: {_format_currency(row['target_price'])}" if pd.notna(row["target_price"]) else "Target: —"
        )
        columns[4].write(
            f"{row['priority']} | {row['target_condition']}".strip(" |")
            if str(row["priority"]).strip() or str(row["target_condition"]).strip()
            else "—"
        )
        columns[5].write(
            f"Sold {row['shares_sold']:.4f} | Cash {_format_currency(row['cash_generated'])}"
            if pd.notna(row["shares_sold"]) or pd.notna(row["cash_generated"])
            else "—"
        )

        done_key = f"done_reminder_{row_index}"
        ignore_key = f"ignore_reminder_{row_index}"

        if columns[6].button("Done", key=done_key):
            reminders_df.loc[row_index, "status"] = "done"
            reminders_df.loc[row_index, "resolved_at"] = datetime.now().isoformat()
            save_reminders(reminders_df)
            st.rerun()

        if columns[7].button("Ignore", key=ignore_key):
            reminders_df.loc[row_index, "status"] = "ignored"
            reminders_df.loc[row_index, "resolved_at"] = datetime.now().isoformat()
            save_reminders(reminders_df)
            st.rerun()


def render_main_page() -> None:
    st.title("Trading Dashboard")

    try:
        holdings_df = load_holdings()
    except Exception as exc:
        st.error(f"Could not load holdings: {exc}")
        return

    if holdings_df.empty:
        st.info("No holdings found yet. Add positions on the setup page.")
        return

    with st.spinner("Fetching market data and building analysis..."):
        try:
            analytics_df = load_analytics_table(holdings_df)
        except Exception as exc:
            st.error(f"Could not build the analysis table: {exc}")
            return

    portfolio_summary = summarize_portfolio(analytics_df)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Tracked tickers", portfolio_summary["positions_count"])
    metric_columns[1].metric("Flagged stocks", portfolio_summary["flagged_count"])
    metric_columns[2].metric("Portfolio value", _format_currency(portfolio_summary["portfolio_value"]))
    metric_columns[3].metric(
        "Unrealized P/L",
        _format_currency(portfolio_summary["total_unrealized_pl"]),
        _format_percent(portfolio_summary["total_unrealized_pl_pct"]),
    )

    render_portfolio_comparison(analytics_df)

    near_high_df = analytics_df.loc[analytics_df["flag_summary"].astype(str).str.contains("Near 52-week high", na=False)]
    near_low_df = analytics_df.loc[analytics_df["flag_summary"].astype(str).str.contains("Near 52-week low", na=False)]
    spike_df = analytics_df.loc[
        analytics_df["flag_summary"].astype(str).str.contains("Big daily jump after flat week", na=False)
    ]

    _render_focus_table(
        "Near 52-Week High",
        near_high_df,
        "Stocks close to their 52-week highs.",
    )
    _render_focus_table(
        "Near 52-Week Low",
        near_low_df,
        "Stocks close to their 52-week lows.",
    )
    _render_focus_table(
        "Spike After Flat Week",
        spike_df,
        "Possible sell/rebuy-watch candidates: average-ish week, then large previous-day jump.",
    )

    render_reminders_section(holdings_df)
    render_add_stocks_section(holdings_df)
    render_sell_section(holdings_df)
    render_buyback_section(holdings_df)

    st.subheader("Analysis Table")
    display_analytics_df = analytics_df[DEFAULT_ANALYTICS_COLUMNS].copy()
    st.dataframe(
        display_analytics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "position_type": st.column_config.TextColumn("Mode"),
            "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
            "avg_buy_price": st.column_config.NumberColumn("Avg Buy", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "daily_change_pct": st.column_config.NumberColumn("1D %", format="%.2f%%"),
            "weekly_change_pct": st.column_config.NumberColumn("7D %", format="%.2f%%"),
            "fifty_two_week_high": st.column_config.NumberColumn("52W High", format="$%.2f"),
            "fifty_two_week_high_date": st.column_config.TextColumn("52W High Date"),
            "fifty_two_week_low": st.column_config.NumberColumn("52W Low", format="$%.2f"),
            "fifty_two_week_low_date": st.column_config.TextColumn("52W Low Date"),
            "distance_to_52w_high_pct": st.column_config.NumberColumn("Dist. to High %", format="%.2f%%"),
            "distance_to_52w_low_pct": st.column_config.NumberColumn("Dist. to Low %", format="%.2f%%"),
            "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
            "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
            "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L", format="$%.2f"),
            "unrealized_pl_pct": st.column_config.NumberColumn("Unrealized P/L %", format="%.2f%%"),
            "flag_summary": st.column_config.TextColumn("Flags"),
        },
    )


def _format_currency(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"${value:,.2f}"


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2f}%"


def main() -> None:
    initialize_data_files()

    if "--reset" in sys.argv:
        reset_app_state()

    st.set_page_config(page_title="Simple Trading Dashboard", layout="wide")

    show_main_page = st.session_state.get("show_main_page", False)
    if show_main_page or holdings_file_is_initialized():
        render_main_page()
        return

    render_onboarding_page()


if __name__ == "__main__":
    main()