from idp.ingest import MAX_SIDE, MIN_SIDE, fit_for_models
from idp.ocr import Token, rebuild_text
from PIL import Image


def test_rebuild_joins_words_and_marks_columns():
    tokens = [
        Token(300, 101, 340, 113, "40"),       # same row, far right -> column break
        Token(10, 100, 60, 112, "Servo"),
        Token(63, 100, 110, 112, "motor"),     # small gap -> space
        Token(10, 160, 80, 172, "Subtotal"),   # big vertical gap -> blank line before
        Token(300, 160, 360, 172, "3,383.00"),
    ]
    assert rebuild_text(tokens) == "Servo motor | 40\n\nSubtotal | 3,383.00"


def test_rebuild_empty():
    assert rebuild_text([]) == ""


def test_fit_for_models_bounds():
    big = fit_for_models(Image.new("RGB", (2550, 3300)))
    assert max(big.size) == MAX_SIDE
    narrow = fit_for_models(Image.new("RGB", (200, 900)))
    assert min(narrow.size) >= MIN_SIDE
