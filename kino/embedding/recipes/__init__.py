import json
from pathlib import Path

RECIPES_DIR = Path(__file__).parent
FULL_RECIPE_KEY = "full"


def _load_recipe_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "groups" not in data:
        raise ValueError(f"{path} is missing a required 'groups' key")
    data.setdefault("description", "")
    return data


def resolve_recipe(key):
    # looks up a recipe by its JSON filename stem, e.g. 'R01_plot' for recipes/R01_plot.json
    path = RECIPES_DIR / f"{key}.json"
    if not path.exists():
        available = ", ".join(all_recipe_keys())
        raise KeyError(f"Unknown recipe {key!r} (no {path}). Available: {available}")
    return _load_recipe_file(path)


def all_recipe_keys(include_full=True):
    keys = sorted(
        p.stem for p in RECIPES_DIR.glob("*.json")
        if p.stem != FULL_RECIPE_KEY
    )
    if include_full and (RECIPES_DIR / f"{FULL_RECIPE_KEY}.json").exists():
        keys.append(FULL_RECIPE_KEY)
    return keys


def load_all_recipes():
    # key -> recipe dict, for every *.json file in recipes/ (includes 'full')
    return {p.stem: _load_recipe_file(p) for p in sorted(RECIPES_DIR.glob("*.json"))}
