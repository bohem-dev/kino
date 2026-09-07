import argparse
import glob
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from kino.jsonl_store import DEFAULT_DIR, SHARD_SIZE, shard_filename, shard_index

OLD_DIR = "out/movies_raw"


def group_by_shard(old_dir):
    by_shard = defaultdict(list)
    skipped_non_numeric = 0
    for fp in glob.glob(os.path.join(old_dir, "*.json")):
        stem = os.path.splitext(os.path.basename(fp))[0]
        if not stem.isdigit():
            skipped_non_numeric += 1
            continue
        by_shard[shard_index(int(stem))].append(fp)
    return by_shard, skipped_non_numeric


def build_shard(shard_idx, filepaths, out_dir):
    records = []
    errors = []
    for fp in filepaths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            errors.append((fp, str(e)))
            continue
        records.append(data)

    # Stable order makes diffing/spot-checking shards sane; id is always
    # present (even on TMDB's success:false error payloads, which is the
    # old id fetch_movies embedded in the filename -- fall back to that).
    records.sort(key=lambda d: d.get("id") or 0)

    out_path = os.path.join(out_dir, shard_filename(shard_idx))
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_path, out_path)

    return shard_idx, len(records), errors


def main(old_dir=OLD_DIR, out_dir=DEFAULT_DIR, workers=None, overwrite=False):
    workers = workers or os.cpu_count() or 1
    os.makedirs(out_dir, exist_ok=True)

    print(f"[scan] grouping ids in {old_dir} by shard (size={SHARD_SIZE})...")
    by_shard, skipped_non_numeric = group_by_shard(old_dir)
    total_files = sum(len(v) for v in by_shard.values())
    print(f"[scan] {total_files} files across {len(by_shard)} shards"
          f" ({skipped_non_numeric} non-numeric filenames skipped)")

    if not overwrite:
        pending = {
            idx: fps for idx, fps in by_shard.items()
            if not os.path.exists(os.path.join(out_dir, shard_filename(idx)))
        }
        already_done = len(by_shard) - len(pending)
        if already_done:
            print(f"[skip] {already_done} shard(s) already exist -- pass --overwrite to redo them")
        by_shard = pending

    if not by_shard:
        print("[done] nothing to migrate.")
        return

    t0 = time.time()
    done_shards = 0
    done_records = 0
    all_errors = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(build_shard, idx, fps, out_dir): idx
            for idx, fps in by_shard.items()
        }
        for fut in as_completed(futures):
            shard_idx, n_records, errors = fut.result()
            done_shards += 1
            done_records += n_records
            all_errors.extend(errors)
            elapsed = time.time() - t0
            print(f"[shard {shard_idx:05d}] {n_records} records "
                  f"({done_shards}/{len(by_shard)} shards, {done_records} records, "
                  f"{elapsed:.1f}s elapsed)")

    print(f"\n[done] wrote {done_records} records across {done_shards} shards -> {out_dir}")
    if all_errors:
        print(f"[warn] {len(all_errors)} unreadable file(s):")
        for fp, err in all_errors[:20]:
            print(f"  {fp}: {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")

    print(
        "\nNext: spot-check counts (e.g. `python -c \"from kino.jsonl_store import "
        "count_records; print(count_records())\"`), run the rest of the pipeline "
        "against the new store, then remove the old out/movies_raw/ directory "
        "yourself once you're confident."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time: out/movies_raw/*.json -> out/movies_raw.jsonl/ shards")
    parser.add_argument("old_dir", nargs="?", default=OLD_DIR)
    parser.add_argument("out_dir", nargs="?", default=DEFAULT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild shards that already exist")
    args = parser.parse_args()

    main(args.old_dir, args.out_dir, workers=args.workers, overwrite=args.overwrite)
