import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import requests

from kino import jsonl_store

OUTPUT_DIR = jsonl_store.DEFAULT_DIR
ID_EXPORT_DIR = "downloads/tmdb-ids"
MOVIE_API_URL = "https://api.themoviedb.org/3/movie"
MAX_WORKERS = 20

load_dotenv()
API_KEY = os.getenv("TMDB_READ")
headers = {"accept": "application/json", "Authorization": f"Bearer {API_KEY}"}

# append_records() does one open(...'a')/write() per record; a single write
# is only guaranteed atomic up to PIPE_BUF and these JSON lines run well
# past that, so concurrent fetch threads share this lock around the append.
_append_lock = threading.Lock()


def latest_movie_export(export_dir=ID_EXPORT_DIR):
    # most recently downloaded movie_ids_*.json -- avoids hand-editing a
    # hardcoded filename every time a fresh export is pulled
    candidates = glob.glob(os.path.join(export_dir, "movie_ids_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No movie_ids export found in {export_dir}. "
            "Run `python main.py fetch-ids` first."
        )
    return max(candidates, key=os.path.getmtime)


def get_movie_full(movie_id):
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "append_to_response": (
            "credits,keywords,videos,images,reviews,similar,"
            "recommendations,release_dates,alternative_titles,"
            "external_ids,translations,watch/providers"
        )
    }
    data = requests.get(base_url, headers=headers, params=params).json()

    if data.get("belongs_to_collection"):
        collection_id = data["belongs_to_collection"]["id"]
        coll_resp = requests.get(
            f"https://api.themoviedb.org/3/collection/{collection_id}",
            headers=headers
        )
        data["collection_details"] = coll_resp.json()

    lists_resp = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}/lists",
        headers=headers
    )
    data["movie_lists"] = lists_resp.json()

    return data


def get_movie_full_safe(movie_id, retries=3):
    for attempt in range(retries):
        try:
            return get_movie_full(movie_id)
        except Exception as e:
            print(f"Retry {attempt + 1} for {movie_id}: {e}")
            time.sleep(2 ** attempt)
    return None


def get_existing_ids(output_dir=OUTPUT_DIR):
    return jsonl_store.read_existing_ids(output_dir)


def fetch_and_save(movie_id):
    data = get_movie_full_safe(movie_id)
    if data:
        # id isn't guaranteed to be in the payload on a success:false
        # error response -- clean.py sweeps those, but jsonl_store shards
        # by id, so it needs one to place the record at all.
        data.setdefault("id", movie_id)
        with _append_lock:
            jsonl_store.append_records([data], base_dir=OUTPUT_DIR)
        return data.get("title", movie_id)
    return None


def get_movie_light(movie_id):
    # bare /movie/{id}, no append_to_response -- just the base fields
    # (popularity, vote_average, vote_count, revenue, budget, status)
    # without the expensive nested data that almost never changes
    resp = requests.get(f"{MOVIE_API_URL}/{movie_id}", headers=headers)
    return resp.json()


def get_movie_light_safe(movie_id, retries=3):
    for attempt in range(retries):
        try:
            return get_movie_light(movie_id)
        except Exception as e:
            print(f"Retry {attempt + 1} for {movie_id}: {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_light_for_refresh(movie_id):
    # just the network call, no file I/O -- patching happens afterwards,
    # batched per shard, in refresh_existing
    data = get_movie_light_safe(movie_id)
    if not data or data.get("success") is False:
        status_code = data.get("status_code") if data else None
        return {"status": "failed", "id": movie_id, "status_code": status_code}
    return {"status": "fetched", "id": movie_id, "data": data}


def refresh_existing(workers=MAX_WORKERS, limit=None):
    ids = sorted(get_existing_ids())
    if limit:
        ids = ids[:limit]
    total = len(ids)
    print(f"Refreshing volatile fields (popularity, votes, revenue, status, ...) for {total} existing movies "
          f"using {workers} workers...")

    fetched = {}  # id -> light data
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_light_for_refresh, mid): mid for mid in ids}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            pct = round(i / total * 100, 2)
            if r["status"] == "fetched":
                fetched[r["id"]] = r["data"]
                print(f"[{pct}%] [{i}/{total}] Fetched: id={r['id']}")
            else:
                failed += 1
                print(f"[{pct}%] [{i}/{total}] NOT FOUND on TMDB, will leave local record as-is: "
                      f"id={r['id']}, status_code={r['status_code']}")

    print(f"\nFetched {len(fetched)} updates, patching shards in one rewrite pass each...")

    # Merge each new light payload into its existing record (only the base
    # fields get overwritten; credits/keywords/images/etc. survive) --
    # needs the old record first, so read every affected shard once before
    # handing whole-record replacements to patch_shards.
    merged = {}
    before = {}
    by_shard = {}
    for mid in fetched:
        by_shard.setdefault(jsonl_store.shard_index(mid), []).append(mid)

    for shard_idx, shard_ids in by_shard.items():
        path = os.path.join(OUTPUT_DIR, jsonl_store.shard_filename(shard_idx))
        if not os.path.exists(path):
            continue
        wanted = set(shard_ids)
        for _, mid, existing in jsonl_store.iter_shard_records(path):
            if mid in wanted:
                before[mid] = {"popularity": existing.get("popularity"), "vote_count": existing.get("vote_count")}
                existing.update(fetched[mid])
                merged[mid] = existing

    missing = jsonl_store.patch_shards(merged)

    refreshed = 0
    for mid, rec in merged.items():
        b = before.get(mid, {})
        print(f"Refreshed: {rec.get('title')} (id={mid})  "
              f"popularity {b.get('popularity')} -> {rec.get('popularity')}  "
              f"votes {b.get('vote_count')} -> {rec.get('vote_count')}")
        refreshed += 1

    for mid in missing:
        print(f"SKIPPED, no local record for id={mid} (fetched from TMDB but nothing to patch)")

    print(f"\nRefresh done. refreshed={refreshed}  failed(not-found)={failed}  missing(no local record)={len(missing)}")


def main(refresh=False, refresh_only=False, workers=MAX_WORKERS, limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not refresh_only:
        existing_ids = get_existing_ids()

        movie_file = latest_movie_export()
        print(f"Using ID export: {movie_file}")

        movie_id_list = []
        with open(movie_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                movie_id_list.append(record["id"])

        movie_id_list = [mid for mid in movie_id_list if mid not in existing_ids]

        total = len(existing_ids) + len(movie_id_list)
        current = len(existing_ids)

        print(f"Already fetched: {len(existing_ids)}")
        print(f"Remaining to fetch: {len(movie_id_list)}")
        print(f"Total: {total}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_and_save, mid): mid for mid in movie_id_list}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    current += 1
                    print(f"[{round(current/total*100, 2)}%] [{current}/{total}] Saved: {result}")
                if current % 500 == 0:
                    print(f"\n\n\n\n\n{current} movies processed\n\n\n\n\n")

    if refresh or refresh_only:
        print()
        refresh_existing(workers=workers, limit=limit)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                         help="Also refresh volatile fields (popularity, votes, revenue, status) on already-fetched movies")
    parser.add_argument("--refresh-only", action="store_true",
                         help="Skip fetching new movies; only refresh existing ones")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Refresh at most N existing movies (for testing)")
    args = parser.parse_args()

    main(refresh=args.refresh, refresh_only=args.refresh_only, workers=args.workers, limit=args.limit)
