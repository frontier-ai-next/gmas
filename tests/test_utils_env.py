"""Tests for gmas.utils.env."""

import os
import sys
from pathlib import Path

from gmas.utils import env as env_module
from gmas.utils.env import configure_console, load_dotenv_file


class TestConfigureConsole:
    def test_reconfigure_when_supported(self, monkeypatch):
        class FakeStream:
            def __init__(self):
                self.calls = []

            def reconfigure(self, **kwargs):
                self.calls.append(kwargs)

        fake_out = FakeStream()
        fake_err = FakeStream()
        monkeypatch.setattr(sys, "stdout", fake_out)
        monkeypatch.setattr(sys, "stderr", fake_err)
        configure_console()
        assert len(fake_out.calls) == 1
        assert fake_out.calls[0]["encoding"] == "utf-8"
        assert len(fake_err.calls) == 1

    def test_skips_when_no_reconfigure(self, monkeypatch):
        class NoReconf:
            pass

        monkeypatch.setattr(sys, "stdout", NoReconf())
        monkeypatch.setattr(sys, "stderr", NoReconf())
        configure_console()


class TestLoadDotenvFile:
    def test_missing_file_noop(self, tmp_path: Path):
        load_dotenv_file(tmp_path / "nope.env")
        assert "FROM_DOTENV" not in os.environ

    def test_loads_pairs_and_skips_rules(self, tmp_path: Path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text(
            "\n# comment\n\nKEY1=value1\nINVALID\n KEY2 = 'quoted' \nEXISTING=should_not_override\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("EXISTING", "prior")
        monkeypatch.delenv("KEY1", raising=False)
        monkeypatch.delenv("KEY2", raising=False)
        try:
            load_dotenv_file(p)
            assert os.environ["KEY1"] == "value1"
            assert os.environ["KEY2"] == "quoted"
            assert os.environ["EXISTING"] == "prior"
        finally:
            os.environ.pop("KEY1", None)
            os.environ.pop("KEY2", None)

    def test_utf8_sig_bom(self, tmp_path: Path, monkeypatch):
        p = tmp_path / "e.env"
        p.write_bytes(b"\xef\xbb\xbfBOMKEY=bomval\n")
        monkeypatch.delenv("BOMKEY", raising=False)
        try:
            load_dotenv_file(p)
            assert os.environ["BOMKEY"] == "bomval"
        finally:
            os.environ.pop("BOMKEY", None)


def test_module_all_exports():
    assert callable(configure_console)
    assert callable(load_dotenv_file)
    assert hasattr(env_module, "configure_console")
