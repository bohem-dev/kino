import os
from multiprocessing import cpu_count

from kino import jsonl_store


def is_valid_record(data):
    return not (isinstance(data, dict) and data.get("success") is False)


def clean_store(base_dir=jsonl_store.DEFAULT_DIR, dry_run=False, workers=None):
    shards = jsonl_store.list_shards(base_dir)
    if not shards:
        print(f"No shards found in {base_dir}.")
        return

    workers = workers or cpu_count()
    print(f"Scanning {len(shards)} shard(s) in {base_dir} using {workers} parallel workers "
          f"({'dry run' if dry_run else 'will delete bad records'})...")

    if dry_run:
        removed_ids = set()
        kept_count = 0
        for sp in shards:
            for _, mid, data in jsonl_store.iter_shard_records(sp):
                if is_valid_record(data):
                    kept_count += 1
                else:
                    removed_ids.add(mid)
                    reason = f"status_code={data.get('status_code')} {data.get('status_message', '')}".strip()
                    print(f"[DRY RUN] Would delete: id={mid} from {sp.name}  ({reason})")
    else:
        kept_count, removed_ids = jsonl_store.filter_shards(is_valid_record, base_dir=base_dir, workers=workers)
        for mid in sorted(removed_ids):
            print(f"Deleted: id={mid}")

    total = kept_count + len(removed_ids)
    pct_deleted = (len(removed_ids) / total * 100) if total else 0
    pct_kept = (kept_count / total * 100) if total else 0

    print("\n--- Summary ---")
    print(f"Total records scanned    : {total}")
    print(f"Deleted (success=false)  : {len(removed_ids)} ({pct_deleted:.3f}%)")
    print(f"Kept (valid movie data)  : {kept_count} ({pct_kept:.3f}%)")

    if dry_run and removed_ids:
        ids_path = os.path.join("out", "failed_ids_dry_run.txt")
        os.makedirs("out", exist_ok=True)
        with open(ids_path, "w") as f:
            f.write("\n".join(str(i) for i in sorted(removed_ids)))
        print(f"\nWrote {len(removed_ids)} candidate ids -> {ids_path} (re-fetch these after cleanup)")


def main(folder=jsonl_store.DEFAULT_DIR, dry_run=False, workers=None):
    clean_store(folder, dry_run=dry_run, workers=workers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drop records that are actually TMDB API error responses")
    parser.add_argument("folder", nargs="?", default=jsonl_store.DEFAULT_DIR, help="JSONL store directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    main(args.folder, dry_run=args.dry_run, workers=args.workers)
