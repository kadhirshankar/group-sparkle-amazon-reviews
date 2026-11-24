"""Capture genuine screenshots only after trained models are available.

Run against a local Streamlit server (port 8501 by default). This script fails
rather than taking images of an unavailable or simulated prediction.
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT = Path("outputs/m3_app")
OUTPUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
    page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded", timeout=90000)
    page.get_by_text("Amazon Review Sentiment Lab", exact=True).wait_for(timeout=90000)

    # Require that two real fitted artifact files loaded, not just placeholder cards.
    text = page.locator("body").inner_text(timeout=20000)
    if "Models ready" not in text or "/ 3" not in text:
        raise RuntimeError("Model availability summary is missing")
    if "0 / 3" in text or "1 / 3" in text:
        raise RuntimeError("Fewer than two real trained models loaded")

    page.get_by_role("textbox", name="Review text").fill(
        "These crackers taste delicious, arrived fresh, and I would definitely order them again."
    )
    page.get_by_role("button", name="Analyze sentiment").click()
    page.get_by_text("Model predictions", exact=True).wait_for(timeout=40000)
    page.get_by_text("Multinomial Naive Bayes (BoW)", exact=True).first.wait_for(timeout=20000)
    page.get_by_text("Logistic Regression (TF-IDF)", exact=True).first.wait_for(timeout=20000)
    page.screenshot(path=str(OUTPUT / "01_live_predictions.png"), full_page=True)

    page.get_by_role("tab", name="Dataset & model results").click()
    page.get_by_text("Macro F1 by model", exact=True).wait_for(timeout=40000)
    page.get_by_text("Macro F1 by model", exact=True).scroll_into_view_if_needed()
    # Streamlit charts render asynchronously and sometimes only when scrolled into view.
    page.wait_for_timeout(4000)
    page.get_by_text("Dataset overview", exact=True).scroll_into_view_if_needed()
    page.wait_for_timeout(1200)
    page.screenshot(path=str(OUTPUT / "02_model_comparison.png"), full_page=True)
    browser.close()
    print("Saved two real Streamlit screenshots to outputs/m3_app")
