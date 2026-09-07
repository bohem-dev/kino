# Updating with new movies (no re-download)

The pipeline dedupes by TMDB movie ID at two separate stages, so pulling in newly
released/added movies is cheap — you never re-fetch or re-flatten anything you already have.

- `kino/fetch_movies.py`'s `get_existing_ids()` scans `out/movies_raw.jsonl/` (sharded JSONL, see
  `kino/jsonl_store.py`) and only requests IDs missing from the store.
- `kino/to_parquet.py`'s `load_existing_ids()` reads the `id` column already written to
  `out/movies_parquet/*.parquet` and only flattens JSON files not yet ingested.

So the only thing that's actually stale is the **list of IDs to consider** — that's what
`kino/fetch_id_exports.py` refreshes. Steps:

1. **Pull a fresh ID export.**
   ```
   python main.py fetch-ids
   ```
   Downloads today's `movie_ids_MM_DD_YYYY.json` (TMDB publishes a new one daily) to
   `downloads/tmdb-ids/`, and — as a side effect — writes a diff report telling you how many
   movie IDs are new vs. yesterday's export (`out/report_<date>.txt`,
   `out/data/movie_ids_diff_<date>.csv`). Worth a glance to sanity-check the batch size before
   fetching. (That diff only gets written if *yesterday's* export file also happens to be on
   disk — skip a day and you just lose that one day's comparison, not anything else.)

2. **Fetch.**
   ```
   python main.py fetch-movies
   ```
   Automatically picks the most recently downloaded `movie_ids_*.json` — no file path to edit.
   It loads all IDs from that file, subtracts everything already in `out/movies_raw.jsonl/`, and
   only calls the TMDB API for the difference. With over a million movies already fetched, a
   routine refresh is typically a few hundred to a few thousand new IDs — a couple minutes, not a
   full re-run.

   Steps 1+2 together are `python main.py update`.

3. **Clean, then rebuild derived outputs:**
   ```
   python main.py clean          # drops any TMDB "not found"/error responses saved as records
   python main.py to-parquet     # only ingests the newly-added records
   python main.py export-tsv     # regenerates the TSVs from the full corpus (fast, no API calls)
   ```
   `python main.py schema` only needs re-running if TMDB adds a genuinely new field to the API
   response — not part of the routine refresh.

## Refreshing data on movies you already have (popularity, votes, etc.)

`fetch-movies --refresh` (or `--refresh-only` to skip the new-ID fetch entirely) covers this:
it re-hits bare `/movie/{id}` for every existing id — no `append_to_response`, so it skips the
expensive nested data (cast/keywords/images/etc.) that almost never changes — and patches just
the volatile fields (`popularity`, `vote_average`, `vote_count`, `revenue`, `status`, ...) into
the existing record. Fetches are concurrent across all existing ids; patches are batched one
rewrite per affected shard rather than one per movie (see `kino/jsonl_store.py`).
