"""Базовые тесты структуры проекта."""

from __future__ import annotations

import pathlib


def test_main_entry_exists():
    import main  # noqa: F401


def test_paths_and_arrays_load():
    from app import constants

    # Базовый каталог должен быть родителем пакета app
    assert constants.BASE_DIR.is_dir()
    # Данные m/M должны быть считаны и конечны
    assert constants.DIFF_MAG_ARRAY.shape == constants.SUM_MAG_ARRAY.shape
    assert (constants.DIFF_MAG_ARRAY == constants.DIFF_MAG_ARRAY).all()
    assert (constants.SUM_MAG_ARRAY == constants.SUM_MAG_ARRAY).all()


def test_logs_and_data_dirs():
    from app import constants

    data_dir = pathlib.Path(constants.BASE_DIR / "data")
    logs_dir = pathlib.Path(constants.BASE_DIR / "logs")
    assert data_dir.is_dir()
    assert logs_dir.is_dir()
