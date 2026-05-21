import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timedelta
import os

ETFS = {
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLY":  "Cons. Discret.",
    "XLP":  "Cons. Staples",
    "XLE":  "Energy",
    "XLB":  "Materials",
    "XLC":  "Comm. Services",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
}

LOOKBACK_DAYS = 20
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_prices(tickers: list[str], lookback: int = LOOKBACK_DAYS) -> pd.DataFrame | None:
    end = datetime.today()
    start = end - timedelta(days=lookback * 2)
    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )["Close"]
        prices = raw.iloc[-lookback:]
        if prices.empty or prices.isnull().all().all():
            return None
        return prices
    except Exception:
        return None


def make_demo_prices(tickers: list[str], lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Generate realistic-looking demo prices when network is unavailable."""
    np.random.seed(42)
    end = datetime.today()
    dates = pd.bdate_range(end=end, periods=lookback)
    base = {t: 100.0 for t in tickers}
    rows = []
    for _ in dates:
        row = {t: base[t] for t in tickers}
        rows.append(row)
        for t in tickers:
            base[t] *= 1 + np.random.normal(0.001, 0.012)
    return pd.DataFrame(rows, index=dates)


def compute_cumulative_returns(prices: pd.DataFrame) -> pd.Series:
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100  # in %


def build_chart(returns: pd.Series, prices: pd.DataFrame, demo: bool = False) -> str:
    returns_sorted = returns.sort_values(ascending=False)

    labels = [f"{t}\n({ETFS[t]})" for t in returns_sorted.index]
    values = returns_sorted.values
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        y = bar.get_height()
        va = "bottom" if val >= 0 else "top"
        offset = 0.15 if val >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{val:+.2f}%",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_ylim(min(values) * 1.3 - 1, max(values) * 1.3 + 1)

    period_start = prices.index[0].strftime("%b %d, %Y")
    period_end = prices.index[-1].strftime("%b %d, %Y")
    demo_tag = "  [DEMO DATA]" if demo else ""
    ax.set_title(
        f"ETF Cumulative Returns — {LOOKBACK_DAYS}-Day Period{demo_tag}\n"
        f"{period_start}  →  {period_end}",
        fontsize=14,
        fontweight="bold",
        color="#2c3e50" if not demo else "#c0392b",
        pad=16,
    )
    ax.set_ylabel("Cumulative Return (%)", fontsize=11, color="#2c3e50")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, linestyle="-")
    ax.spines[["top", "right"]].set_visible(False)

    today_str = datetime.today().strftime("%Y-%m-%d")
    suffix = "_demo" if demo else ""
    filename = os.path.join(OUTPUT_DIR, f"etf_returns_{today_str}{suffix}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def main():
    import shutil

    tickers = list(ETFS.keys())
    print(f"Downloading {LOOKBACK_DAYS}-day price history for {len(tickers)} ETFs…")
    prices = fetch_prices(tickers)
    demo = False

    if prices is None:
        print("  ⚠ Network unavailable — using demo data for preview.")
        prices = make_demo_prices(tickers)
        demo = True
    else:
        print(f"Trading days fetched: {len(prices)}  "
              f"({prices.index[0].date()} → {prices.index[-1].date()})")

    returns = compute_cumulative_returns(prices)
    print("\nCumulative returns (sorted):")
    for ticker, ret in returns.sort_values(ascending=False).items():
        print(f"  {ticker:5s}  {ret:+.2f}%")

    chart_path = build_chart(returns, prices, demo=demo)
    print(f"\nChart saved → {chart_path}")

    latest = os.path.join(OUTPUT_DIR, "etf_returns_latest.png")
    shutil.copy2(chart_path, latest)
    print(f"Latest copy  → {latest}")


if __name__ == "__main__":
    main()
