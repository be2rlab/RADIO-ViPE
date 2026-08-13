from __future__ import annotations

from pathlib import Path

import yaml

PRESET_DIR = Path(__file__).resolve().parents[1] / "configs" / "ablation"


def list_presets() -> list[str]:
    """Every preset name, sorted."""
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_preset(name: str) -> dict:
    """Load one preset by name (no .yaml suffix)."""
    path = PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No ablation preset {name!r} in {PRESET_DIR}. "
            f"Available: {', '.join(list_presets())}"
        )
    return yaml.safe_load(path.read_text()) or {}


def _render(value) -> str:
    """Render a Python scalar the way Hydra expects it on the command line."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def flatten_overrides(preset: dict, prefix: str = "pipeline.slam") -> list[str]:
    out: list[str] = []

    def walk(node: dict, path: str) -> None:
        for key in sorted(node):
            value = node[key]
            full = f"{path}.{key}"
            if isinstance(value, dict):
                walk(value, full)
            else:
                out.append(f"{full}={_render(value)}")

    walk(preset, prefix)
    return sorted(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print Hydra overrides for a preset.")
    ap.add_argument("preset")
    ap.add_argument("--prefix", default="pipeline.slam")
    args = ap.parse_args()
    print(" ".join(flatten_overrides(load_preset(args.preset), args.prefix)))
