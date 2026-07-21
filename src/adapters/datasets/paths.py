from __future__ import annotations

from pathlib import Path


def resolve_dataset_root(path: str | Path) -> Path:
    """Resolve repo-relative dataset paths when notebooks run from subfolders."""
    root = Path(path).expanduser()
    if root.is_absolute():
        return root
    if root.exists():
        return root.resolve()

    project_root = _find_project_root(Path.cwd()) or _find_project_root(Path(__file__).resolve())
    if project_root is None:
        return root
    return project_root / root


def _find_project_root(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    while True:
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent
