"""
logic.py
--------
Pure Python logic — no Streamlit here.
This is where you put data fetching, calculations, and transformations.
The UI in app.py calls these functions and displays the results.
"""

from datetime import datetime, timedelta
import random


def get_last_updated() -> str:
    """Return the current timestamp as a readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_sample_data(days: int = 30) -> list[dict]:
    """
    Generate a simple time series of daily values.

    Returns a list of dicts like:
        [{"date": "2026-01-01", "value": 42.5}, ...]

    Replace this with your real data source (CSV, API, database, etc.)
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")

    random.seed(42)          # fixed seed → reproducible results
    start = datetime(2026, 1, 1)
    value = 100.0
    rows = []

    for i in range(days):
        value += random.uniform(-5, 7)   # small random walk
        rows.append({
            "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "value": round(value, 2),
        })

    return rows


def compute_summary(data: list[dict]) -> dict:
    """
    Compute basic summary statistics from the data.

    Returns a dict with min, max, mean, and last value.
    """
    if not data:
        raise ValueError("Cannot compute summary of empty data")

    values = [row["value"] for row in data]
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
        "last": values[-1],
    }
