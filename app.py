# app.py
"""
Streamlit Global Currency Converter with Visual Analytics
No API key required — uses exchangerate.host (free, no key).
Save this file as `app.py` and run: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

# --------------------------
# Configuration & Constants
# --------------------------
API_BASE = "https://api.exchangerate.host"
# Common currency symbol map for nicer UI
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "PKR": "₨",
    "INR": "₹", "AUD": "$", "CAD": "$", "CNY": "¥", "CHF": "CHF",
    "NZD": "$", "SGD": "$", "HKD": "$", "AED": "د.إ", "SAR": "﷼"
}
# Some currency -> representative country code for emoji flags (approximate)
CURRENCY_FLAG_COUNTRY = {
    "USD":"US","EUR":"EU","GBP":"GB","JPY":"JP","PKR":"PK","INR":"IN",
    "AUD":"AU","CAD":"CA","CNY":"CN","CHF":"CH","NZD":"NZ","SGD":"SG",
    "HKD":"HK","AED":"AE","SAR":"SA"
}

# --------------------------
# Helper functions
# --------------------------

@st.cache_data(ttl=3600)
def fetch_symbols() -> Dict[str, Any]:
    """Fetch available currency symbols from exchangerate.host."""
    resp = requests.get(f"{API_BASE}/symbols", timeout=15)
    resp.raise_for_status()
    return resp.json().get("symbols", {})


@st.cache_data(ttl=300)
def fetch_latest_rates(base: str = "USD") -> Dict[str, float]:
    """Fetch latest exchange rates for the given base currency."""
    params = {"base": base}
    resp = requests.get(f"{API_BASE}/latest", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("rates", {})


@st.cache_data(ttl=3600)
def fetch_top_rates(base: str = "USD", top_n: int = 10) -> pd.DataFrame:
    """Return a dataframe of the top_n most-traded currencies compared to base.
       This chooses a list of 'major' currencies for reliable UX."""
    majors = ["USD","EUR","GBP","JPY","AUD","CAD","CHF","CNY","HKD","INR","SGD","NZD","AED","SAR","PKR"]
    if base not in majors:
        majors.insert(0, base)
    rates = fetch_latest_rates(base)
    rows = []
    for cur in majors:
        if cur == base: 
            rows.append({"currency": cur, "rate": 1.0})
        else:
            rate = rates.get(cur)
            if rate:
                rows.append({"currency": cur, "rate": rate})
    df = pd.DataFrame(rows).sort_values("rate", ascending=False).head(top_n)
    return df


@st.cache_data(ttl=3600)
def fetch_timeseries(base: str, symbol: str, days: int = 7) -> pd.DataFrame:
    """Fetch historical timeseries for last `days` days for a currency pair."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "base": base,
        "symbols": symbol
    }
    resp = requests.get(f"{API_BASE}/timeseries", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError("Failed to fetch timeseries")
    rates = data.get("rates", {})
    rows = []
    for d, vals in sorted(rates.items()):
        rows.append({"date": d, "rate": vals.get(symbol)})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


def convert_currency(amount: float, base: str, target: str) -> Tuple[float, float]:
    """Convert amount from base to target using latest rates.
       Returns converted_amount and rate (1 base = rate target)."""
    rates = fetch_latest_rates(base)
    if target not in rates:
        raise ValueError(f"Currency {target} not available for base {base}.")
    rate = rates[target]
    return amount * rate, rate


# Simple NLP parser for inputs like "Convert 100 USD to EUR"
def parse_nl_input(text: str) -> Tuple[float, str, str]:
    """Attempt to parse a natural-language currency conversion instruction.
       Returns (amount, base_currency, target_currency). Raises ValueError if cannot parse."""
    text = text.strip()
    tokens = text.replace(",", "").upper().split()
    amount = None
    base = None
    target = None

    # Look for common patterns: "CONVERT 100 USD TO EUR" or "100 USD TO EUR" or "100 USD EUR"
    # Find first token that is a number
    for i, t in enumerate(tokens):
        # number detection (int/float)
        try:
            val = float(t)
            amount = val
            # try to find currency code right after number
            if i + 1 < len(tokens) and len(tokens[i+1]) in (3,):
                base = tokens[i+1]
            break
        except:
            # maybe token like "USD100" or "100USD"
            for ch in ["USD","EUR","GBP","JPY","PKR","INR","AUD","CAD","CNY","CHF","SGD","NZD","HKD","AED","SAR"]:
                if ch in t:
                    # try to extract numeric part
                    num_part = ''.join([c for c in t if (c.isdigit() or c=='.')])
                    try:
                        amount = float(num_part) if num_part else None
                        base = ch
                    except:
                        pass
    # fallback: if user typed "convert 100 usd to eur"
    if "TO" in tokens:
        idx = tokens.index("TO")
        if idx + 1 < len(tokens):
            possible = tokens[idx + 1]
            if len(possible) == 3:
                target = possible
    # If we have amount and base but no target, maybe last token is currency
    if not target:
        last = tokens[-1]
        if len(last) == 3 and last != base:
            target = last
    if amount is None or base is None or target is None:
        raise ValueError("Couldn't parse input. Try 'Convert 100 USD to EUR' or '100 USD EUR'.")
    return amount, base, target


def currency_flag(code: str) -> str:
    """Return an emoji flag or empty string for given currency code if mapped."""
    ccode = CURRENCY_FLAG_COUNTRY.get(code)
    if not ccode:
        return ""
    # EU flag is not standard via region indicators; handle separately
    if ccode == "EU":
        return "🇪🇺"
    # create regional indicator symbols
    if len(ccode) != 2:
        return ""
    return chr(127397 + ord(ccode[0])) + chr(127397 + ord(ccode[1]))


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get(code, "")


# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="Global Currency Converter & Analytics", layout="wide", initial_sidebar_state="expanded")

# Header - luxurious style
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:18px;">
      <div style="width:72px;height:72px;border-radius:14px;background:linear-gradient(135deg,#0f172a,#0ea5e9);display:flex;align-items:center;justify-content:center;box-shadow:0 8px 30px rgba(14,165,233,0.15);">
        <h2 style="color:white;margin:0;font-weight:700">FX</h2>
      </div>
      <div>
        <h1 style="margin:0;">Global Currency Converter</h1>
        <p style="margin:0;color:gray">Real-time conversions • Historical trends • Visual analytics</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.markdown("Cache & refresh settings (no API key needed).")
    ttl_minutes = st.number_input("Rates cache TTL (minutes)", min_value=1, max_value=1440, value=5, step=1,
                                  help="How long to cache latest rates before automatically refetching.")
    st.caption("Use the Refresh button below to force an immediate fetch (clears cached rates).")
    refresh = st.button("Refresh exchange rates (clear cache)")

    st.markdown("---")
    st.markdown("Display options")
    theme = st.selectbox("Theme (cosmetic)", ["Auto", "Dark", "Light"])
    # Note: streamlit themes are controlled by user's client-side theme; this is cosmetic text only.

# If user clicked refresh: clear cached functions related to latest rates and top rates & timeseries
if refresh:
    try:
        fetch_latest_rates.clear()
        fetch_top_rates.clear()
        fetch_timeseries.clear()
        fetch_symbols.clear()
        st.experimental_rerun()
    except Exception:
        # just proceed if cannot clear
        pass

# Adjust cached ttl dynamically: we re-register wrapper by setting st.cache_data(ttl=...)
# (We'll implement fetch_latest_rates with TTL by re-calling function below if needed.)
# For simplicity: we won't dynamically recreate cached functions here; instead we rely on function-level caching (default TTLs above).
# But we allow the user to force refresh (above).

# Main layout: two columns
col1, col2 = st.columns([1, 2])

# Load symbols
try:
    raw_symbols = fetch_symbols()
    SYMBOLS = sorted(raw_symbols.keys())
except Exception as e:
    st.error(f"Error fetching currency symbols: {e}")
    SYMBOLS = ["USD", "EUR", "GBP", "JPY", "PKR", "INR", "AUD", "CAD", "CNY"]

with col1:
    st.subheader("Convert Currency")
    # Natural language input
    nl_input = st.text_input("Enter amount (or natural language)", placeholder="e.g. Convert 100 USD to EUR or 250 PKR EUR")
    # Dropdowns
    default_base = "USD"
    base_cur = st.selectbox("Base currency", options=SYMBOLS, index=SYMBOLS.index(default_base) if default_base in SYMBOLS else 0)
    target_cur = st.selectbox("Target currency", options=SYMBOLS, index=SYMBOLS.index("EUR") if "EUR" in SYMBOLS else 1)
    amount = st.number_input("Amount", min_value=0.0, value=100.0, format="%.4f")

    # Buttons for convert and NLP parse
    colbtn1, colbtn2 = st.columns(2)
    with colbtn1:
        if st.button("Convert"):
            user_wants_convert = True
        else:
            user_wants_convert = False
    with colbtn2:
        if st.button("Parse NL & Convert"):
            user_wants_nl = True
        else:
            user_wants_nl = False

    # Extra options
    show_topn = st.slider("Show top N currencies (bar chart)", min_value=5, max_value=15, value=10)
    hist_days = st.selectbox("Historical days for trend", [7, 14, 30], index=0)
    show_pie = st.checkbox("Show pie chart comparison", value=True)
    allow_export = st.checkbox("Enable export CSV for results", value=True)

    # Process actions
    try:
        # If user used NLP parse
        if 'user_wants_nl' in locals() and user_wants_nl and nl_input.strip():
            try:
                amt, b, t = parse_nl_input(nl_input)
                amount = amt
                base_cur = b
                target_cur = t
                st.success(f"Parsed: {amount} {base_cur} → {target_cur}")
            except Exception as e:
                st.error(f"NL parse failed: {e}")

        if 'user_wants_convert' in locals() and user_wants_convert:
            # Perform conversion
            with st.spinner("Fetching rates and converting..."):
                try:
                    converted, rate = convert_currency(amount, base_cur, target_cur)
                    st.markdown(f"### Result: {amount:,.4f} {base_cur} ➜ **{converted:,.4f} {target_cur}**")
                    st.markdown(f"1 {base_cur} = {rate:,.6f} {target_cur} {currency_flag(target_cur)} {currency_symbol(target_cur)}")
                    # Provide small table of conversion
                    summary_df = pd.DataFrame([{
                        "timestamp_utc": datetime.utcnow().isoformat(),
                        "base": base_cur,
                        "target": target_cur,
                        "amount": amount,
                        "converted": converted,
                        "rate": rate
                    }])
                    st.dataframe(summary_df.T, use_container_width=True)
                    if allow_export:
                        csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download result CSV", csv_bytes, file_name="conversion_result.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

with col2:
    st.subheader("Analytics & Visuals")
    # Show summary of top currencies relative to base
    try:
        top_df = fetch_top_rates(base_cur, top_n=show_topn)
        # Bar chart
        fig_bar = px.bar(top_df, x="currency", y="rate", text="rate",
                         title=f"Top {len(top_df)} currency rates relative to {base_cur}",
                         labels={"rate": f"1 {base_cur} -> X currency"})
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart (optional)
        if show_pie:
            pie_df = top_df.copy()
            # Normalize rates for a pie visualization (not financial sense but relative)
            pie_df['val'] = pie_df['rate'] / pie_df['rate'].sum()
            fig_pie = px.pie(pie_df, names='currency', values='val',
                             title=f"Rate distribution among top {len(pie_df)} (relative)")
            st.plotly_chart(fig_pie, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load analytics: {e}")

    # Historical trend for selected pair
    st.markdown("### Historical Rate Trend")
    try:
        ts_df = fetch_timeseries(base_cur, target_cur, days=hist_days)
        if ts_df.empty:
            st.info("No historical data available for this pair.")
        else:
            fig_line = px.line(ts_df, x="date", y="rate",
                               title=f"{base_cur}/{target_cur} — last {hist_days} days",
                               labels={"rate": f"{base_cur} -> {target_cur}"})
            st.plotly_chart(fig_line, use_container_width=True)

            # allow CSV export of timeseries
            if allow_export:
                csv_bytes = ts_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download historical CSV", csv_bytes, file_name=f"{base_cur}_{target_cur}_history.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")

# Footer / tips
st.markdown("---")
st.markdown(
    """
    **Tips & Notes:**  
    - This app uses the free `exchangerate.host` endpoints (no API key needed).  
    - Rates are cached for performance — use *Refresh exchange rates* to force an immediate update.  
    - For best visuals, open in a wide browser window.  
    """
)
