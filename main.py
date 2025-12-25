"""Запуск аппроксимации из корня проекта."""

from __future__ import annotations

import argparse

from app.approximation import main as run_approximation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подбор параметров и визуализация.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Каталог с Excel-файлами (по умолчанию ./data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_approximation(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
