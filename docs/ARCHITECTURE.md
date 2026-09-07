# Operation Kino — Architecture

A pipeline that pulls TMDB's full movie catalog down to local JSON, then flattens it into
Parquet/TSV for analysis. Runs against `.env` credentials (`TMDB_READ`, `TMDB_API_KEY`).

## Layout

```
main.py                  CLI entry point for TMDB ingestion — dispatches to kino/*
kino/
  fetch_id_exports.py     daily ID export download + diff report
  fetch_movies.py          full per-movie detail fetch (incremental)
  jsonl_store.py             shared sharded-JSONL storage layer (read/append/patch/filter)
  migrate_to_jsonl.py         one-time out/movies_raw/*.json -> out/movies_raw.jsonl/ migration
  clean.py                     drop failed/invalid raw fetches
  discover_schema.py            infer a Parquet schema from the raw corpus
  to_parquet.py                   flatten JSON -> Parquet (incremental)
  export_tsv.py                     curated TSV export for kino-universe.html
  embedding/                 the whole embedding-experiment pipeline lives here (see below),
                              entry point kino/embedding/build_dataset.py — no separate root script
    build_dataset.py            CLI entry point: subset -> embed -> visualize, per recipe
    subset.py                    out/movies_raw.jsonl -> a ranked, field-selected CSV
    recipes/                      named field-group experiments, one JSON file each
    harrier.py, word2vec.py        embedders: CSV -> vectors
    visualize.py                    vectors -> one dropdown-driven projection HTML
docs/
  ARCHITECTURE.md, UPDATING.md
downloads/tmdb-ids/       raw daily ID exports from TMDB (per type, per date)
out/movies_raw.jsonl/      sharded JSONL, one line per movie, keyed by TMDB id — source of truth.
                            shard_NNNNN.jsonl holds ids [NNNNN*10000, NNNNN*10000+10000) — see
                            kino/jsonl_store.py. Replaces the old out/movies_raw/{id}.json layout
                            (1.3M+ loose files was choking Finder/Terminal); migrate with
                            `python main.py migrate-jsonl`.
out/discovered_schema.json
out/movies_parquet/         columnar export of movies_raw.jsonl, deduped by id
out/data/                    fetch_id_exports.py's snapshot/diff/history CSVs
out/report_*.txt              fetch_id_exports.py's daily text report
out/movie_stats.tsv, out/movie_extra_stats.tsv   export_tsv.py's curated exports
out/embedding_analysis/      output of the embedding pipeline (see below)
```

## CLI

```
python main.py fetch-ids                # pull today's TMDB ID exports + diff report
python main.py fetch-movies              # fetch any movie IDs not yet in out/movies_raw.jsonl
python main.py clean [--dry-run]          # drop records that are actually API errors
python main.py schema                      # (re)discover the Parquet schema
python main.py to-parquet                    # flatten movies_raw.jsonl -> movies_parquet
python main.py export-tsv                     # write movie_stats.tsv / movie_extra_stats.tsv
python main.py update                          # fetch-ids + fetch-movies (routine refresh)
python main.py migrate-jsonl                    # one-time: movies_raw/*.json -> movies_raw.jsonl/
```

Each `kino/*.py` module is also runnable directly (`python -m kino.clean --dry-run`,
`python kino/discover_schema.py`, etc.) — `main.py` is a thin dispatcher, not a wrapper
that hides functionality.

## Pipeline order

```
fetch_id_exports.py       downloads/tmdb-ids/*.json      daily ID export + diff report
      |
      v
fetch_movies.py             out/movies_raw.jsonl/shard_NNNNN.jsonl  full per-movie detail (TMDB API)
      |
      v
clean.py                     (rewrites affected shards in place)     drop failed/invalid fetches
      |
      v
discover_schema.py            out/discovered_schema.json               infer field shapes across the corpus
      |
      v
to_parquet.py                   out/movies_parquet/*.parquet             flatten JSON -> columnar Parquet
      |                                                                   (uses discovered_schema.json)
      v
export_tsv.py                    out/movie_stats.tsv                       curated TSV export for
                                  out/movie_extra_stats.tsv                 kino-universe.html (external viewer)
```

## Module-by-module

### `kino/fetch_id_exports.py` — daily ID exports + trend report
TMDB publishes daily snapshot files of every ID it knows about (movies, TV, people,
collections, keywords, companies, adult variants — see `EXPORT_TYPES`). This module:
- Picks "today" using TMDB's actual publish cutoff (8am UTC — before that, use yesterday's date).
- Downloads each export as `.json.gz` to `downloads/tmdb-ids/`, extracts it, and **skips any
  file that's already on disk** (by filename, which encodes the date) — safe to re-run.
- Loads today's and yesterday's snapshots into pandas, and for each export type writes:
  - a per-day CSV snapshot (`out/data/{type}_snapshot_{date}.csv`)
  - a word-frequency CSV over titles/names (`out/data/{type}_word_freq_{date}.csv`)
  - a new/removed/changed diff CSV vs. yesterday (`out/data/{type}_diff_{date}.csv`) — only
    written if yesterday's export file also happens to be on disk; skip a day and this is
    silently omitted for that gap.
  - a human-readable report (`out/report_{date}.txt`) with popularity stats, lexical stats,
    and top gainers/losers
  - an appended row in `out/data/summary_history.csv` (long-running trend log, never overwritten)

This is the discovery layer — it tells you what IDs exist and what's new, but does **not**
fetch full movie detail.

### `kino/jsonl_store.py` — shared sharded-JSONL storage layer
Everything else in the pipeline that used to open `out/movies_raw/{id}.json` directly now goes
through this module. Records live in `out/movies_raw.jsonl/shard_NNNNN.jsonl` — one JSON object
per line, sharded by `id // SHARD_SIZE` (10,000 ids per shard) so any caller can compute *where*
a record would live from its id alone, with no separate index file. Key operations:
- `read_existing_ids()` / `iter_records()` / `get_record(id)` — read paths
- `append_records(records)` — append brand-new ids to their shards (not for updating existing ones)
- `patch_shards(updates)` — bulk in-place update; groups by shard and rewrites each affected shard
  **exactly once**, so callers must batch every update they have before calling this rather than
  patching one id at a time (a per-record call would mean a full shard rewrite per record)
- `filter_shards(predicate)` — keep-only rewrite, one pass per shard, used by `clean.py`

### `kino/migrate_to_jsonl.py` — one-time migration
Converts the old `out/movies_raw/{id}.json` layout (1.3M+ loose files — Finder/Terminal choke on
that many entries in one directory) into `out/movies_raw.jsonl/`. Groups old filenames by target
shard first, then a process pool builds one shard at a time (bounded memory — a worker only ever
holds one shard's records). Idempotent: shards that already exist are skipped unless
`--overwrite`. Doesn't delete `out/movies_raw/` — that's a manual step once you've verified the
migration.

### `kino/fetch_movies.py` — full detail fetch
Reads the **most recently downloaded** `movie_ids_*.json` file in `downloads/tmdb-ids/`
(`latest_movie_export()`, picked by file mtime — no hardcoded date to remember to edit) and,
for each ID, hits `/movie/{id}` with a large `append_to_response` (credits, keywords, videos,
images, reviews, similar, recommendations, release_dates, alternative_titles, external_ids,
translations, watch/providers), plus separate calls for collection details and movie lists.
Appends each result into `out/movies_raw.jsonl/` via `jsonl_store.append_records()`, guarded by a
lock (concurrent threads appending to the same shard file need serializing — a single `write()`
call is only atomic up to `PIPE_BUF`, and these JSON lines run well past that).

Key behavior: **`get_existing_ids()` scans `out/movies_raw.jsonl/` first and filters the ID list
down to only what's missing before hitting the API.** Already an incremental fetcher — never
re-downloads a movie it already has a record for. 20 threads (`MAX_WORKERS`), retries with
exponential backoff.

`--refresh` / `--refresh-only` (light per-movie refetch of volatile fields — popularity, votes,
revenue, status) fetches every existing id's light payload concurrently first (pure network I/O,
no file access), then merges and patches **once per affected shard** via `patch_shards()` — not
once per movie, which would mean rewriting the same ~10k-record shard file over and over.

Note: TMDB returns `{"success": false, "status_code": 34, ...}` for IDs it can't resolve
(deleted/merged/invalid). Since that's still a non-empty dict, it gets saved like any other
result — `kino/clean.py` is what sweeps those back out.

### `kino/clean.py` — drop invalid entries
Walks every shard in `out/movies_raw.jsonl/` (via `jsonl_store.filter_shards()`, parallelized
across CPUs, one shard per task) and drops any record that's actually a TMDB API error response
(`"success": false` — covers not-found, rate-limited, and anything else TMDB signals that way,
not just one specific status code). Has `--dry-run` (also writes the candidate IDs to
`out/failed_ids_dry_run.txt`) and `--workers`.

### `kino/discover_schema.py` — infer a schema from the raw corpus
Scans every shard in `out/movies_raw.jsonl/` (one shard per process-pool task) and classifies
each top-level key as:
- **scalar** (str/int/float/bool/None) → goes straight into a Parquet column
- **fixed nested dict** (a dict with a small, stable set of subkeys, e.g. `belongs_to_collection`)
  → gets flattened to `key.subkey` columns
- **dynamic/JSON** (lists, or dicts with too many distinct subkeys to flatten, e.g. `credits`)
  → stored as a JSON string column

Writes the result to `out/discovered_schema.json`, which `to_parquet.py` consumes directly. Only
needs re-running if TMDB's response shape changes or genuinely new fields show up — the schema
is a property of the *shape* of the data, not any particular snapshot.

### `kino/to_parquet.py` — JSON → Parquet
Loads `out/discovered_schema.json`, then for every record in `out/movies_raw.jsonl/` flattens it
into a row using that schema (scalars as-is, fixed nested as `key.subkey` columns, dynamic fields
`json.dumps`-encoded as strings). Writes to `out/movies_parquet/part-NNNN.parquet`, rolling to a
new part file once the current one hits 500MB.

Key behavior: **`load_existing_ids()` reads the `id` column out of whatever Parquet files
already exist and skips any raw record whose ID is already ingested.** Also incremental by
design — safe to re-run after `fetch_movies.py` adds new records.

### `kino/export_tsv.py` — curated TSV export
Walks `out/movies_raw.jsonl/` and pulls a hand-picked set of fields (year, cast, director, genres,
budget/revenue, keyword counts, etc.) into `out/movie_stats.tsv`, plus a second TSV of top-200
most-common values per category (keywords, companies, directors, actors, TMDB lists, countries)
in `out/movie_extra_stats.tsv`. These are meant to be dragged into a companion
`kino-universe.html` viewer (not present in this repo). This module is **not incremental** — it
re-reads and re-writes the full corpus every run — but that's fine since it's just a CSV/TSV
write, not an API call.

## Embedding pipeline

A second, independent pipeline that turns `out/movies_raw.jsonl/` into comparable
movie-embedding experiments. It reads the raw JSON `fetch_movies.py` already downloaded; it
doesn't touch TMDB.

```
kino/embedding/subset.py       out/movies_raw.jsonl -> a revenue-ranked, field-selected CSV
      |                          (extraction is cached; --fields or --recipe picks what's kept)
      v
kino/embedding/harrier.py        CSV's embedding_text -> vectors (sentence-transformers)
kino/embedding/word2vec.py         CSV's embedding_text -> vectors (trained from scratch, gensim)
      |
      v
kino/embedding/visualize.py      vectors -> one HTML file: PCA/UMAP/t-SNE, 2D/3D, dropdown-switchable
```

`kino/embedding/build_dataset.py` runs all three stages for a named **recipe** — a JSON file in
`kino/embedding/recipes/` describing which field groups to keep (e.g. `R01_plot.json` =
`{"groups": ["text"]}` for tagline+synopsis only). Each `(top_n, recipe)` combination gets its
own directory so recipes never clobber each other's output:

```
python kino/embedding/build_dataset.py --top-n 10000 --recipe R01_plot
python kino/embedding/build_dataset.py --top-n 10000 --recipe full          # every field group
python kino/embedding/build_dataset.py --top-n 10000 --recipe all --embedder both  # every recipe
python kino/embedding/build_dataset.py --top-n 10000 --recipe all --embedder both --resume  # continue after a shutdown
python kino/embedding/build_dataset.py --list-recipes

out/embedding_analysis/
  _cache/ranked_full.parquet         the full revenue-ranked extraction, shared across every
                                        recipe/embedder at a given --top-n; subset.py only
                                        rescans out/movies_raw.jsonl when a run asks for more rows
                                        than are cached
  runs/<top_n>/<recipe_key>/
    subset.csv                         this run's field-selected CSV (+ embedding_text)
    manifest.json                      recipe params + what was produced, for reference
    vectors/<embedder>/*.npy           vectors + aligned movie ids
    models/word2vec/*.model            word2vec only
  runs/<top_n>/explorer.html           ONE page covering every (recipe, embedder) run from
                                          this invocation — Run + Dimensions dropdowns on top of
                                          each figure's own PCA/UMAP/t-SNE dropdown, so comparing
                                          runs never means opening a second browser tab
```

**Checkpointing / `--resume`.** `build_dataset.py --recipe all --embedder both` can run for hours
(t-SNE especially), so every stage checkpoints to disk as it finishes rather than only at the very
end, and `--resume` picks back up from whatever's already there:
- Per recipe/embedder: `run_dir/manifest.json` is written as soon as that recipe's requested
  embedders finish, recording where their vectors landed. `--resume` re-verifies those files still
  exist on disk (not just the manifest entry — a killed process can leave a stale pointer) and
  skips straight to the next recipe/embedder combo that's actually missing; a recipe with only
  `word2vec` done and `harrier` still pending reruns just `harrier`, not the whole recipe.
- Inside the explorer build (`visualize.main_combined`, the slow part — 26 PCA/UMAP/t-SNE fits per
  run): each run's fitted 2D/3D coordinates are cached to
  `vectors/<embedder>/<vectors_stem>_projections_cache_<sample|full>.npz` right after that run's
  fits complete. `--resume` loads a run's cache instead of refitting it, so a shutdown mid-explorer
  only costs the one run that was in flight, not everything already fitted.
- Without `--resume`, behavior is unchanged — recipes/embedders/fits are always recomputed, and any
  existing manifest or cache files for that run are simply overwritten.

Add a new experiment by dropping a new `{"description": ..., "groups": [...]}` JSON file into
`kino/embedding/recipes/` — no code changes needed. `"groups"` is a subset of
`kino.embedding.subset.FIELD_GROUPS` (`title`, `year`, `language`, `taxonomy`, `production`,
`collection`, `crew`, `cast`, `text`, `financial`, `external`, `flags`, `reviews`); `"all"` means
every group (see `recipes/full.json`).

**Parallelism.** Three independent CPU-bound stages get real concurrency, each sized to avoid
oversubscribing the machine rather than fighting itself for cores:
- `subset.py` extracts every shard (one task per shard) in a `ProcessPoolExecutor` sized to every core.
- `visualize.py` fits all 26 (2 dims × [PCA + 5 UMAP presets + 6 t-SNE perplexities]) projections
  in a `ProcessPoolExecutor` sized to every core, with each worker pinned to a single BLAS/numba
  thread (`OMP_NUM_THREADS=1` etc., set before numpy/umap/sklearn are ever imported) — N worker
  processes × 1 thread each, not N processes × M threads each competing for the same cores.
- `build_dataset.py --embedder both` runs harrier (GPU-bound, via `sentence-transformers`/MPS) and
  word2vec (CPU-bound, via `gensim`) concurrently in threads, since they don't contend for the
  same resource.

Different *recipes*, and the projection pool within one recipe, stay sequential relative to each
other on purpose — each already claims every core on its own, so overlapping two of those pools
would cause contention instead of speeding anything up.

Every stage is also independently runnable and argparse-driven (`python kino/embedding/subset.py
--recipe R01_plot`, `python kino/embedding/visualize.py --run-dirs out/embedding_analysis/runs/10000/R01_plot
out/embedding_analysis/runs/10000/R02_taxonomy --embedder harrier --combined-out explorer.html`,
etc.) — `build_dataset.py` is a convenience wrapper, not the only way in.

## Notes / rough edges
- `.env` holds two TMDB credentials (`TMDB_READ`, `TMDB_API_KEY`) — only `TMDB_READ` (a Bearer
  read token) is actually used by the scripts today.
