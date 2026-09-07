#!/usr/bin/env python3
"""
Operation Kino - CLI entry point for the TMDB ingestion pipeline.

Pipeline order:
    fetch-ids      -> downloads/tmdb-ids/*.json               daily ID export + diff report
    fetch-movies   -> out/movies_raw.jsonl/shard_NNNNN.jsonl    full per-movie detail
    clean          -> (rewrites shards in place)                 drop failed/invalid fetches
    schema         -> out/discovered_schema.json                  infer field shapes
    to-parquet     -> out/movies_parquet/*.parquet                 flatten JSON -> Parquet
    export-tsv     -> out/movie_stats.tsv, ...                      curated TSV export
    migrate-jsonl  -> out/movies_raw/*.json -> out/movies_raw.jsonl/  one-time migration

A separate embedding pipeline (subset -> embed -> visualize) runs off the
same out/movies_raw.jsonl/ store but isn't wired into this CLI -- its own
entry point is kino/embedding/build_dataset.py:
    python kino/embedding/build_dataset.py --top-n 10000 --recipe R01_plot
    python kino/embedding/build_dataset.py --list-recipes

See docs/ARCHITECTURE.md for how it all fits together, docs/UPDATING.md for
the day-to-day incremental refresh workflow.
"""

import argparse
import sys


def cmd_fetch_ids(args):
    from kino import fetch_id_exports
    fetch_id_exports.main()


def cmd_fetch_movies(args):
    from kino import fetch_movies
    fetch_movies.main(refresh=args.refresh, refresh_only=args.refresh_only, workers=args.workers, limit=args.limit)


def cmd_clean(args):
    from kino import clean
    clean.main(args.folder, dry_run=args.dry_run, workers=args.workers)


def cmd_schema(args):
    from kino import discover_schema
    discover_schema.main()


def cmd_to_parquet(args):
    from kino import to_parquet
    to_parquet.main()


def cmd_export_tsv(args):
    from kino import export_tsv
    export_tsv.main(args.input_dir, args.workers)


def cmd_migrate_jsonl(args):
    from kino import migrate_to_jsonl
    migrate_to_jsonl.main(args.old_dir, args.out_dir, workers=args.workers, overwrite=args.overwrite)


def cmd_update(args):
    """Convenience: the routine incremental-refresh workflow from docs/UPDATING.md."""
    from kino import fetch_id_exports, fetch_movies
    print("=== step 1/2: fetch-ids ===")
    fetch_id_exports.main()
    print("\n=== step 2/2: fetch-movies ===")
    fetch_movies.main(refresh=args.refresh)
    print(
        "\nDone. Run `python main.py clean` to drop any failed fetches, then "
        "`python main.py to-parquet` / `export-tsv` to refresh derived outputs."
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-ids", help="Download today's TMDB ID exports + diff report")
    p.set_defaults(func=cmd_fetch_ids)

    p = sub.add_parser("fetch-movies", help="Fetch full detail for any movie IDs not already in out/movies_raw.jsonl")
    p.add_argument("--refresh", action="store_true",
                    help="Also refresh volatile fields (popularity, votes, revenue, status) on already-fetched movies")
    p.add_argument("--refresh-only", action="store_true",
                    help="Skip fetching new movies; only refresh existing ones (no full-database re-download)")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Refresh at most N existing movies (for testing)")
    p.set_defaults(func=cmd_fetch_movies)

    p = sub.add_parser("clean", help="Drop records that are actually API error responses")
    p.add_argument("folder", nargs="?", default="out/movies_raw.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("schema", help="Discover the Parquet schema from out/movies_raw.jsonl")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("to-parquet", help="Flatten out/movies_raw.jsonl into out/movies_parquet (incremental)")
    p.set_defaults(func=cmd_to_parquet)

    p = sub.add_parser("export-tsv", help="Export curated TSVs for kino-universe.html")
    p.add_argument("input_dir", nargs="?", default="out/movies_raw.jsonl")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_export_tsv)

    p = sub.add_parser("migrate-jsonl", help="One-time: out/movies_raw/*.json -> out/movies_raw.jsonl/ shards")
    p.add_argument("old_dir", nargs="?", default="out/movies_raw")
    p.add_argument("out_dir", nargs="?", default="out/movies_raw.jsonl")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Rebuild shards that already exist")
    p.set_defaults(func=cmd_migrate_jsonl)

    p = sub.add_parser("update", help="fetch-ids + fetch-movies: the routine daily/weekly refresh")
    p.add_argument("--refresh", action="store_true",
                    help="Also refresh volatile fields on already-fetched movies (see fetch-movies --refresh)")
    p.set_defaults(func=cmd_update)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
