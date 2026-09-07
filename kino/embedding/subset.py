import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Running this file directly (`python kino/embedding/subset.py`) puts this
# file's own directory on sys.path, not the repo root -- the lazy
# `kino.embedding.recipes` import below (and the jsonl_store import) would
# fail without this. No effect when imported normally as a module (repo
# root is already importable).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kino import jsonl_store

RAW_DIR = jsonl_store.DEFAULT_DIR
OUTPUT_DIR = "out/embedding_analysis"
CACHE_PATH = os.path.join(OUTPUT_DIR, "_cache", "ranked_full.parquet")
TOP_N_DEFAULT = 10000
MAX_WORKERS = os.cpu_count() or 1

MAX_CAST = 20          # top-billed cast members kept in the cache
MAX_REVIEWS = 100         # review excerpts kept in the cache
REVIEW_EXCERPT_CHARS = 300

# Fields that are always present in the output, regardless of --fields.
IDENTITY_FIELDS = [
    "id", "title", "original_title", "year", "release_date",
    "status", "runtime", "original_language",
]

# Optional field groups -> output columns they contribute.
# "cast" is special-cased: the cache stores up to MAX_CAST names in
# cast_full, and the output column top_cast is sliced from it via --cast-n.
# "title" / "year" / "language" contribute no extra CSV columns (title,
# year and original_language are always present as identity metadata) --
# selecting them only gates whether they show up as a line in
# embedding_text. This lets a recipe be genuinely title-only, year-only, etc.
FIELD_GROUPS = {
    "title": [],
    "year": [],
    "language": [],
    "financial": ["revenue", "budget", "popularity", "vote_average", "vote_count"],
    "taxonomy": ["genres", "keywords"],
    "production": ["production_companies", "production_countries", "spoken_languages"],
    "collection": ["collection"],
    "crew": ["directors", "writers", "producers", "composers", "cinematographers", "editors"],
    "cast": ["top_cast"],
    "text": ["tagline", "overview"],
    "external": ["imdb_id", "homepage"],
    "flags": ["origin_country", "adult", "video"],
    "reviews": ["review_excerpts"],
}

DEFAULT_GROUPS = [
    "title", "year", "language", "financial", "taxonomy", "production",
    "collection", "crew", "cast", "text", "flags",
]

# Columns stored in the cache that don't map 1:1 to an output column name.
CACHE_ONLY_COLUMNS = ["cast_full"]


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def clean_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split()).strip()


def print_schema(data):
    # prints the full shape of a single movie JSON record
    print("\n=== FULL SCHEMA ===")
    for key, value in sorted(data.items()):
        if isinstance(value, dict):
            print(f"  {key}: dict -> {sorted(value.keys())}")
        elif isinstance(value, list):
            sample = value[0] if value else None
            if isinstance(sample, dict):
                print(f"  {key}: list[dict] -> {sorted(sample.keys())}")
            else:
                print(f"  {key}: list[{type(sample).__name__ if sample is not None else 'empty'}]")
        else:
            print(f"  {key}: {type(value).__name__}")
    print()


def process_record(data):
    # pulls the full superset of fields worth caching out of one record --
    # field selection for the final CSV happens later, from the cache
    if data.get("success") is False:
        return None

    rd = data.get("release_date") or ""
    year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None

    genres = [g.get("name") for g in data.get("genres", []) if g.get("name")]
    companies = [c.get("name") for c in data.get("production_companies", []) if c.get("name")]
    countries = [c.get("name") for c in data.get("production_countries", []) if c.get("name")]
    spoken = [l.get("english_name") for l in data.get("spoken_languages", []) if l.get("english_name")]
    keywords = [k.get("name") for k in safe_get(data, "keywords", "keywords", default=[]) if k.get("name")]
    origin_country = [c for c in data.get("origin_country", []) if c]

    cast = safe_get(data, "credits", "cast", default=[])
    crew = safe_get(data, "credits", "crew", default=[])
    cast_full = [c.get("name") for c in sorted(cast, key=lambda c: c.get("order", 999))[:MAX_CAST] if c.get("name")]
    directors = [c.get("name") for c in crew if c.get("job") == "Director" and c.get("name")]
    writers = [c.get("name") for c in crew if c.get("job") in ("Writer", "Screenplay", "Story") and c.get("name")]
    producers = [c.get("name") for c in crew if c.get("job") == "Producer" and c.get("name")]
    composers = [c.get("name") for c in crew if c.get("job") == "Original Music Composer" and c.get("name")]
    cinematographers = [c.get("name") for c in crew if c.get("job") == "Director of Photography" and c.get("name")]
    editors = [c.get("name") for c in crew if c.get("job") == "Editor" and c.get("name")]

    collection = safe_get(data, "belongs_to_collection", "name", default="") or ""

    imdb_id = data.get("imdb_id") or safe_get(data, "external_ids", "imdb_id", default="") or ""

    reviews = safe_get(data, "reviews", "results", default=[])[:MAX_REVIEWS]
    review_excerpts = " ||| ".join(
        f"{r.get('author', '')}: {clean_text(r.get('content', ''))[:REVIEW_EXCERPT_CHARS]}"
        for r in reviews if r.get("content")
    )

    return {
        "id": data.get("id"),
        "title": data.get("title") or "",
        "original_title": data.get("original_title") or "",
        "year": year,
        "release_date": rd,
        "status": data.get("status") or "",
        "runtime": data.get("runtime") or 0,
        "original_language": data.get("original_language") or "",
        "revenue": data.get("revenue") or 0,
        "budget": data.get("budget") or 0,
        "popularity": data.get("popularity") or 0,
        "vote_average": data.get("vote_average") or 0,
        "vote_count": data.get("vote_count") or 0,
        "genres": "|".join(genres),
        "keywords": "|".join(keywords),
        "production_companies": "|".join(companies),
        "production_countries": "|".join(countries),
        "spoken_languages": "|".join(spoken),
        "collection": collection,
        "directors": "|".join(directors),
        "writers": "|".join(writers),
        "producers": "|".join(producers),
        "composers": "|".join(composers),
        "cinematographers": "|".join(cinematographers),
        "editors": "|".join(editors),
        "cast_full": "|".join(cast_full),
        "tagline": data.get("tagline") or "",
        "overview": data.get("overview") or "",
        "imdb_id": imdb_id,
        "homepage": data.get("homepage") or "",
        "origin_country": "|".join(origin_country),
        "adult": bool(data.get("adult", False)),
        "video": bool(data.get("video", False)),
        "review_excerpts": review_excerpts,
    }


def process_shard(shard_path):
    return [
        rec for _, _, data in jsonl_store.iter_shard_records(shard_path)
        if (rec := process_record(data)) is not None
    ]


def build_full_ranked(input_dir, workers):
    # the expensive path: scans every shard, extracts the full field
    # superset, returns the complete revenue>0 set ranked by revenue
    shards = jsonl_store.list_shards(input_dir)
    print(f"Found {len(shards)} shard(s) in {input_dir}. Processing with {workers} workers...")

    for _, _, first_record in jsonl_store.iter_shard_records(shards[0]) if shards else []:
        print_schema(first_record)
        break

    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_shard, sp): sp for sp in shards}
        for i, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            print(f"  shard {i}/{len(shards)} done ({len(records)} records so far)")

    print(f"\nProcessed {len(records)} valid records. Ranking by revenue...")
    df = pd.DataFrame(records)
    df = df[df["revenue"] > 0].sort_values("revenue", ascending=False).reset_index(drop=True)
    return df


def load_or_build_cache(input_dir, cache_path, workers, rebuild=False):
    cache_path = Path(cache_path)
    if not rebuild and cache_path.exists():
        print(f"Loading cached full ranked list -> {cache_path}")
        return pd.read_parquet(cache_path)

    df = build_full_ranked(input_dir, workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Cached {len(df)} ranked records -> {cache_path}")
    return df


def build_embedding_text(row, groups):
    parts = []

    if "title" in groups:
        title = clean_text(row.get("title"))
        if title:
            parts.append(f"Title: {title}")

        original_title = clean_text(row.get("original_title"))
        if original_title and original_title != title:
            parts.append(f"Original title: {original_title}")

    if "year" in groups:
        year = clean_text(row.get("year"))
        if year:
            parts.append(f"Release year: {year}")

    if "collection" in groups:
        collection = clean_text(row.get("collection"))
        if collection:
            parts.append(f"Part of collection: {collection}")

    if "taxonomy" in groups:
        genres = clean_text(row.get("genres"))
        if genres:
            parts.append(f"Genres: {genres}")

    if "language" in groups:
        language = clean_text(row.get("original_language"))
        if language:
            parts.append(f"Original language: {language}")

    if "production" in groups:
        countries = clean_text(row.get("production_countries"))
        if countries:
            parts.append(f"Production countries: {countries}")
        companies = clean_text(row.get("production_companies"))
        if companies:
            parts.append(f"Produced by: {companies}")

    if "crew" in groups:
        directors = clean_text(row.get("directors"))
        if directors:
            parts.append(f"Director: {directors}")
        writers = clean_text(row.get("writers"))
        if writers:
            parts.append(f"Writers: {writers}")
        producers = clean_text(row.get("producers"))
        if producers:
            parts.append(f"Producers: {producers}")
        composers = clean_text(row.get("composers"))
        if composers:
            parts.append(f"Composer: {composers}")
        cinematographers = clean_text(row.get("cinematographers"))
        if cinematographers:
            parts.append(f"Cinematographer: {cinematographers}")

    if "cast" in groups:
        cast = clean_text(row.get("top_cast"))
        if cast:
            parts.append(f"Starring: {cast}")

    if "taxonomy" in groups:
        keywords = clean_text(row.get("keywords"))
        if keywords:
            parts.append(f"Themes and keywords: {keywords}")

    if "text" in groups:
        tagline = clean_text(row.get("tagline"))
        if tagline:
            parts.append(f"Tagline: {tagline}")
        overview = clean_text(row.get("overview"))
        if overview:
            parts.append(f"Synopsis: {overview}")

    if "reviews" in groups:
        reviews = clean_text(row.get("review_excerpts"))
        if reviews:
            parts.append(f"Reviews: {reviews}")

    return "\n".join(parts)


def main(
    input_dir=RAW_DIR,
    workers=MAX_WORKERS,
    top_n=TOP_N_DEFAULT,
    output_dir=OUTPUT_DIR,
    output_name=None,
    cache_path=CACHE_PATH,
    rebuild_cache=False,
    groups=None,
    recipe=None,
    cast_n=5,
):
    if recipe is not None:
        from kino.embedding.recipes import resolve_recipe
        recipe_groups = resolve_recipe(recipe)["groups"]
        groups = list(FIELD_GROUPS) if recipe_groups == "all" else list(recipe_groups)

    groups = list(FIELD_GROUPS) if groups is None or "all" in groups else groups
    cast_n = min(cast_n, MAX_CAST)

    os.makedirs(output_dir, exist_ok=True)

    cache_df = load_or_build_cache(input_dir, cache_path, workers, rebuild=rebuild_cache)
    if top_n > len(cache_df) and not rebuild_cache:
        print(f"Requested top {top_n} exceeds cached {len(cache_df)} records -- rescanning {input_dir}...")
        cache_df = load_or_build_cache(input_dir, cache_path, workers, rebuild=True)

    top_df = cache_df.head(top_n).copy()
    print(f"Selected top {len(top_df)} by revenue "
          f"(highest: {top_df.iloc[0]['title']!r} @ ${top_df.iloc[0]['revenue']:,}, "
          f"lowest in cut: {top_df.iloc[-1]['title']!r} @ ${top_df.iloc[-1]['revenue']:,})")

    if "cast" in groups:
        top_df["top_cast"] = top_df["cast_full"].fillna("").apply(
            lambda s: "|".join(str(s).split("|")[:cast_n]) if s else ""
        )

    selected_cols = IDENTITY_FIELDS + [
        col for g in groups for col in FIELD_GROUPS[g]
    ]
    out_df = top_df[selected_cols].copy()
    out_df["embedding_text"] = top_df.apply(lambda r: build_embedding_text(r, groups), axis=1)

    name = output_name or f"top_{top_n}_by_revenue.csv"
    out_path = os.path.join(output_dir, name)
    out_df.to_csv(out_path, index=False)
    print(f"CSV written -> {out_path} ({os.path.getsize(out_path)/1024:.1f} KB, "
          f"{len(out_df.columns)} columns: {list(out_df.columns)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a revenue-ranked, field-selected CSV from the movie corpus")
    parser.add_argument("input_dir", nargs="?", default=RAW_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--output", default=None, help="Override the output CSV filename")
    parser.add_argument("--cache-path", default=CACHE_PATH,
                         help="Where the full unedited ranked list is cached (parquet)")
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force a full rescan of input_dir even if a cache is present")
    parser.add_argument("--fields", nargs="+", choices=list(FIELD_GROUPS) + ["all"], default=None,
                         help=f"Optional field groups to include -- gates which lines make it into "
                              f"embedding_text (id/title/year/original_language/etc. remain CSV columns "
                              f"regardless). Groups: {', '.join(FIELD_GROUPS)}. Default: {' '.join(DEFAULT_GROUPS)}. "
                              f"Ignored if --recipe is given.")
    parser.add_argument("--recipe", default=None,
                         help="Named recipe from kino/embedding/recipes/*.json (or 'full' for every group) -- "
                              "overrides --fields")
    parser.add_argument("--cast-n", type=int, default=5,
                         help=f"Top-billed cast members to keep when 'cast' is selected (cache holds up to {MAX_CAST})")
    parser.add_argument("--list-fields", action="store_true", help="Print available field groups and exit")
    args = parser.parse_args()

    if args.list_fields:
        print("Always included:", ", ".join(IDENTITY_FIELDS))
        for g, cols in FIELD_GROUPS.items():
            print(f"  {g}: {', '.join(cols) if cols else '(gates embedding_text only)'}")
        raise SystemExit(0)

    main(
        input_dir=args.input_dir,
        workers=args.workers,
        top_n=args.top_n,
        output_dir=args.output_dir,
        output_name=args.output,
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        groups=args.fields if args.fields is not None else DEFAULT_GROUPS,
        recipe=args.recipe,
        cast_n=args.cast_n,
    )
