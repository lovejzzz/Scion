from __future__ import annotations

from pathlib import Path

import pytest

from scion.server import server_command, validate_adapter


def test_server_command_matches_coursemapper_contract(tmp_path: Path) -> None:
    command = server_command(
        binary=tmp_path / "llama-server",
        base=tmp_path / "base.gguf",
        adapter=tmp_path / "scion.gguf",
    )
    assert command[command.index("--port") + 1] == "8799"
    assert command[command.index("--alias") + 1] == "scion-1"
    assert command[command.index("--reasoning") + 1] == "off"
    assert command[command.index("--reasoning-budget") + 1] == "0"
    assert command[command.index("--reasoning-format") + 1] == "deepseek"
    assert command[-2:] == ["--lora", str(tmp_path / "scion.gguf")]


def test_adapter_validation_checks_magic_and_minimum_size(tmp_path: Path) -> None:
    valid = tmp_path / "valid.gguf"
    valid.write_bytes(b"GGUF" + bytes(1020))
    assert validate_adapter(valid)["bytes"] == 1024
    invalid = tmp_path / "invalid.gguf"
    invalid.write_bytes(b"NOPE" + bytes(1020))
    with pytest.raises(RuntimeError, match="not a GGUF"):
        validate_adapter(invalid)
