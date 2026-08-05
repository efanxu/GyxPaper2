"""Thin executable wrapper around the single benchmark_v2 CLI."""

if __package__:
    from .cli import main
else:
    import sys
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from benchmark_v2.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
