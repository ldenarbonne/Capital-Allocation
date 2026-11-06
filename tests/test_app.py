"""Smoke test: the Streamlit app runs end-to-end without raising."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_runs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) >= 4


def test_app_with_var_measure():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("VaR").run()
    assert not at.exception
