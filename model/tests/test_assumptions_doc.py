import importlib.util
from pathlib import Path

from sunrisk import config

ROOT = Path(__file__).resolve().parents[2]


def test_every_assumption_has_valid_confidence():
    for name, a in config.all_assumptions().items():
        assert a.confidence in ("high", "medium", "low"), name
        assert a.source, name


def test_assumptions_doc_is_up_to_date():
    spec = importlib.util.spec_from_file_location("render", ROOT / "model" / "scripts" / "render_assumptions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "docs" / "assumptions.md").read_text() == mod.render(), \
        "run: python model/scripts/render_assumptions.py"
