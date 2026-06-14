from types import SimpleNamespace

import pytest

from src.infrastructure.adapters.livewire.factory import build_livewire_provider


def test_factory_builds_provider(tmp_path) -> None:
    p = build_livewire_provider(SimpleNamespace(livewire_bronze_root=str(tmp_path)))
    assert p.source_name == "livewire"


def test_factory_requires_root() -> None:
    with pytest.raises(ValueError, match="livewire_bronze_root"):
        build_livewire_provider(SimpleNamespace(livewire_bronze_root=None))
