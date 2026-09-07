import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from kino import jsonl_store

INPUT_DIR = jsonl_store.DEFAULT_DIR
OUTPUT_DIR = "out"
MAX_WORKERS = 8


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def analyze_record(data):
    rd = data.get("release_date") or ""
    year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None

    genres = [g.get("name") for g in data.get("genres", []) if g.get("name")]
    countries = [c.get("name") for c in data.get("production_countries", []) if c.get("name")]
    companies = [c.get("name") for c in data.get("production_companies", []) if c.get("name")]
    spoken = [l.get("english_name") for l in data.get("spoken_languages", []) if l.get("english_name")]
    keywords = [k.get("name") for k in safe_get(data, "keywords", "keywords", default=[]) if k.get("name")]

    cast = safe_get(data, "credits", "cast", default=[])
    crew = safe_get(data, "credits", "crew", default=[])
    top_cast = [c.get("name") for c in sorted(cast, key=lambda c: c.get("order", 999))[:5] if c.get("name")]
    directors = [c.get("name") for c in crew if c.get("job") == "Director" and c.get("name")]
    writers = [c.get("name") for c in crew if c.get("job") in ("Writer", "Screenplay") and c.get("name")]

    movie_lists = [l.get("name") for l in safe_get(data, "movie_lists", "results", default=[]) if l.get("name")]

    return {
        "id": data.get("id"),
        "title": data.get("title") or "",
        "year": year,
        "popularity": data.get("popularity", 0) or 0,
        "vote_average": data.get("vote_average", 0) or 0,
        "vote_count": data.get("vote_count", 0) or 0,
        "runtime": data.get("runtime") or 0,
        "budget": data.get("budget", 0) or 0,
        "revenue": data.get("revenue", 0) or 0,
        "adult": data.get("adult", False),
        "video": data.get("video", False),
        "status": data.get("status") or "",
        "original_language": data.get("original_language") or "",
        "genres": genres,
        "production_countries": countries,
        "production_companies": companies,
        "spoken_languages": spoken,
        "keywords": keywords,
        "top_cast": top_cast,
        "directors": directors,
        "writers": writers,
        "movie_lists": movie_lists,
        "review_count": safe_get(data, "reviews", "total_results", default=0) or 0,
        "similar_count": safe_get(data, "similar", "total_results", default=0) or 0,
        "recommendations_count": safe_get(data, "recommendations", "total_results", default=0) or 0,
        "alt_titles_count": len(safe_get(data, "alternative_titles", "titles", default=[]) or []),
        "translations_count": len(safe_get(data, "translations", "translations", default=[]) or []),
        "has_imdb": bool(safe_get(data, "external_ids", "imdb_id")),
        "has_homepage": bool(data.get("homepage")),
        "has_collection": data.get("belongs_to_collection") is not None,
    }


def analyze_shard(shard_path):
    return [
        analyze_record(data)
        for _, _, data in jsonl_store.iter_shard_records(shard_path)
        if data.get("success") is not False
    ]


def collect_records(input_dir, workers):
    shards = jsonl_store.list_shards(input_dir)
    print(f"Found {len(shards)} shard(s). Extracting stats with {workers} workers...")
    records = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(analyze_shard, sp) for sp in shards]
        for i, fut in enumerate(as_completed(futures), 1):
            records.extend(fut.result())
            print(f"  shard {i}/{len(shards)} done ({len(records)} records so far)")
    print(f"Extracted {len(records)} valid records.")
    return records


def write_tsv(records, out_path):
    fieldnames = [
        "id", "title", "year", "popularity", "vote_average", "vote_count",
        "runtime", "budget", "revenue", "adult", "video", "status",
        "original_language", "genres", "production_countries",
        "production_companies", "spoken_languages", "keywords", "top_cast",
        "directors", "writers", "movie_lists", "review_count",
        "similar_count", "recommendations_count", "alt_titles_count",
        "translations_count", "has_imdb", "has_homepage", "has_collection",
    ]
    list_fields = {
        "genres", "production_countries", "production_companies",
        "spoken_languages", "keywords", "top_cast", "directors", "writers",
        "movie_lists",
    }
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in records:
            row = dict(r)
            for lf in list_fields:
                row[lf] = "|".join(row[lf])
            writer.writerow(row)
    print(f"TSV written -> {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


def extra_stats_tsv(records, out_path):
    keyword_counter = Counter(k for r in records for k in r["keywords"])
    company_counter = Counter(c for r in records for c in r["production_companies"])
    director_counter = Counter(d for r in records for d in r["directors"])
    actor_counter = Counter(a for r in records for a in r["top_cast"])
    list_counter = Counter(l for r in records for l in r["movie_lists"])
    country_counter = Counter(c for r in records for c in r["production_countries"])

    rows = []
    for counter, category in [
        (keyword_counter, "Keyword"),
        (company_counter, "Production Company"),
        (director_counter, "Director"),
        (actor_counter, "Top-billed Actor"),
        (list_counter, "TMDB List"),
        (country_counter, "Production Country"),
    ]:
        for value, count in counter.most_common(200):
            rows.append({"category": category, "value": value, "count": count})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "value", "count"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extra stats TSV -> {out_path}")


def main(input_dir=INPUT_DIR, workers=MAX_WORKERS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = collect_records(input_dir, workers)
    write_tsv(records, os.path.join(OUTPUT_DIR, "movie_stats.tsv"))
    extra_stats_tsv(records, os.path.join(OUTPUT_DIR, "movie_extra_stats.tsv"))
    print("Done. Drag movie_stats.tsv / movie_extra_stats.tsv into kino-universe.html to explore.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default=INPUT_DIR)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    main(args.input_dir, args.workers)
