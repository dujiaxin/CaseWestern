from pathlib import Path


def absolute_path(filepath):
    relative = Path(filepath)
    return relative.absolute()
