import pytest
from src.processing.strip_markdown import strip_markdown_elements


def test_strip_checkbox():
    markdown = "- [ ] This is a checkbox"
    assert strip_markdown_elements(markdown) == "This is a checkbox"


def test_strip_bold_text():
    markdown = "This is **bold** text"
    assert strip_markdown_elements(markdown) == "This is bold text"
