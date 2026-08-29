from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_initial_render() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "Churn Risk Agent — Human-in-the-Loop"
    assert app.button[0].label == "Đánh giá khách hàng"
