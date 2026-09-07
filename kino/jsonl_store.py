import json
import os
from collections import defaultdict
from pathlib import Path

SHARD_SIZE = 10_000
DEFAULT_DIR = "out/movies_raw.jsonl"


def shard_index(movie_id):
    return int(movie_id) // SHARD_SIZE


def shard_filename(shard_idx):
    return f"shard_{shard_idx:05d}.jsonl"


def shard_path(base_dir, movie_id):
    return os.path.join(base_dir, shard_filename(shard_index(movie_id)))


def list_shards(base_dir=DEFAULT_DIR):
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(base.glob("shard_*.jsonl"))


def iter_shard_records(path):
    # a malformed line skips rather than blowing up the whole scan --
    # shouldn't happen given atomic writes, but a hand edit could do it
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield line_no, data.get("id"), data


def iter_records(base_dir=DEFAULT_DIR):
    for sp in list_shards(base_dir):
        for _, _, data in iter_shard_records(sp):
            yield data


def count_records(base_dir=DEFAULT_DIR):
    return sum(1 for _ in iter_records(base_dir))


def read_existing_ids(base_dir=DEFAULT_DIR):
    ids = set()
    for sp in list_shards(base_dir):
        for _, movie_id, _ in iter_shard_records(sp):
            if movie_id is not None:
                ids.add(movie_id)
    return ids


def get_record(movie_id, base_dir=DEFAULT_DIR):
    path = shard_path(base_dir, movie_id)
    if not os.path.exists(path):
        return None
    for _, mid, data in iter_shard_records(path):
        if mid == movie_id:
            return data
    return None


def _atomic_write_records(path, records):
    # write-to-tmp-then-replace so a crash mid-write never corrupts the
    # live shard -- worst case you lose the .tmp file
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_path, path)


def append_records(records, base_dir=DEFAULT_DIR):
    # caller's job to make sure these ids aren't already present -- this
    # never dedupes. also not safe across threads without an external
    # lock: a single write() per record can interleave with another
    # thread's write once a line exceeds the OS pipe-buffer guarantee,
    # and these lines run well past that
    if not records:
        return
    os.makedirs(base_dir, exist_ok=True)
    by_shard = defaultdict(list)
    for rec in records:
        by_shard[shard_index(rec["id"])].append(rec)

    for shard_idx, recs in by_shard.items():
        path = os.path.join(base_dir, shard_filename(shard_idx))
        with open(path, "a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")


def patch_shards(updates, base_dir=DEFAULT_DIR):
    # updates: dict[id -> new full record]. never call this once per id --
    # each call rewrites every affected shard in full (~10k records), so
    # batch every update first and call it once. returns ids that had no
    # existing record to patch (use append_records for genuinely new ones)
    if not updates:
        return set()

    by_shard = defaultdict(dict)
    for movie_id, rec in updates.items():
        by_shard[shard_index(movie_id)][movie_id] = rec

    missing = set(updates.keys())
    for shard_idx, shard_updates in by_shard.items():
        path = os.path.join(base_dir, shard_filename(shard_idx))
        if not os.path.exists(path):
            continue
        out_records = []
        for _, mid, data in iter_shard_records(path):
            if mid in shard_updates:
                data = shard_updates[mid]
                missing.discard(mid)
            out_records.append(data)
        _atomic_write_records(path, out_records)

    return missing


def filter_shards(predicate, base_dir=DEFAULT_DIR, workers=None):
    # keep-only rewrite, one shard per task so it's a single pass rather
    # than read-then-delete. returns (kept_count, removed_ids)
    shards = list_shards(base_dir)
    removed_ids = set()
    kept_count = 0

    if workers and workers > 1 and len(shards) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_filter_one_shard, sp, predicate): sp for sp in shards}
            for fut in as_completed(futures):
                kept, removed = fut.result()
                kept_count += kept
                removed_ids.update(removed)
    else:
        for sp in shards:
            kept, removed = _filter_one_shard(sp, predicate)
            kept_count += kept
            removed_ids.update(removed)

    return kept_count, removed_ids


def _filter_one_shard(path, predicate):
    out_records = []
    removed = set()
    for _, mid, data in iter_shard_records(path):
        if predicate(data):
            out_records.append(data)
        else:
            removed.add(mid)
    if removed:
        _atomic_write_records(path, out_records)
    return len(out_records), removed
