import json
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from kino import jsonl_store

RAW_DIR = jsonl_store.DEFAULT_DIR
SCHEMA_OUT = Path("out/discovered_schema.json")

MAX_STABLE_SUBKEYS = 50
NUM_WORKERS = max(mp.cpu_count() - 1, 1)


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


def scan_shard(shard_path):
    top_level_keys = set()
    nested_dict_subkeys = defaultdict(set)
    nested_dict_types = defaultdict(set)
    valid = 0

    for _, _, d in jsonl_store.iter_shard_records(shard_path):
        valid += 1
        top_level_keys.update(d.keys())
        for key, value in d.items():
            nested_dict_types[key].add(type(value).__name__)
            if isinstance(value, dict):
                nested_dict_subkeys[key].update(value.keys())

    return {
        "top_level_keys": top_level_keys,
        "nested_dict_subkeys": dict(nested_dict_subkeys),
        "nested_dict_types": dict(nested_dict_types),
        "valid": valid,
        "errors": 0,
    }


def merge_results(results):
    top_level_keys = set()
    nested_dict_subkeys = defaultdict(set)
    nested_dict_types = defaultdict(set)
    total_valid = 0
    total_errors = 0

    for r in results:
        top_level_keys.update(r["top_level_keys"])
        for k, v in r["nested_dict_subkeys"].items():
            nested_dict_subkeys[k].update(v)
        for k, v in r["nested_dict_types"].items():
            nested_dict_types[k].update(v)
        total_valid += r["valid"]
        total_errors += r["errors"]

    return top_level_keys, nested_dict_subkeys, nested_dict_types, total_valid, total_errors


def classify_fields(top_level_keys, nested_dict_subkeys, nested_dict_types):
    scalar_fields = []
    fixed_nested = {}
    dynamic_json_fields = []

    for key in sorted(top_level_keys):
        types_seen = nested_dict_types.get(key, set())

        if types_seen <= {"NoneType", "dict"} and key in nested_dict_subkeys:
            subkeys = nested_dict_subkeys[key]
            if len(subkeys) <= MAX_STABLE_SUBKEYS:
                fixed_nested[key] = sorted(subkeys)
            else:
                dynamic_json_fields.append(key)
        elif types_seen <= {"NoneType", "list"}:
            dynamic_json_fields.append(key)
        elif types_seen <= {"NoneType", "str", "int", "float", "bool"}:
            scalar_fields.append(key)
        else:
            dynamic_json_fields.append(key)

    return scalar_fields, fixed_nested, dynamic_json_fields


def main():
    shards = jsonl_store.list_shards(RAW_DIR)
    print(f"[scan] found {len(shards)} shard(s) in {RAW_DIR}")
    print(f"[config] using {NUM_WORKERS} worker processes, one shard per task")

    results = []
    t0 = time.time()
    completed_shards = 0
    completed_files = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(scan_shard, sp): sp for sp in shards}
        for future in as_completed(futures):
            sp = futures[future]
            r = future.result()
            results.append(r)
            completed_shards += 1
            completed_files += r["valid"] + r["errors"]
            elapsed = time.time() - t0
            rate = completed_files / elapsed if elapsed > 0 else 0
            print(f"[shard {sp.name}] done  |  {completed_shards}/{len(shards)} shards, "
                  f"{completed_files} records  |  {rate:8.1f} records/sec  |  {elapsed:6.1f}s elapsed")

    top_level_keys, nested_dict_subkeys, nested_dict_types, total_valid, total_errors = merge_results(results)
    scalar_fields, fixed_nested, dynamic_json_fields = classify_fields(
        top_level_keys, nested_dict_subkeys, nested_dict_types
    )

    elapsed = time.time() - t0
    print(f"\n=== SCAN COMPLETE ===")
    print(f"Valid files scanned      : {total_valid}")
    print(f"Unreadable/errored files : {total_errors}")
    print(f"Total elapsed            : {elapsed:.1f}s  ({total_valid/elapsed:.1f} files/sec avg)")

    print(f"\n=== DISCOVERED SCHEMA ===")
    print(f"Scalar fields       ({len(scalar_fields)}): {scalar_fields}")
    print(f"\nFixed nested dicts  ({len(fixed_nested)}):")
    for k, subkeys in fixed_nested.items():
        print(f"  {k}: {subkeys}")
    print(f"\nDynamic/JSON fields ({len(dynamic_json_fields)}): {dynamic_json_fields}")

    schema = {
        "scalar_fields": scalar_fields,
        "fixed_nested": fixed_nested,
        "dynamic_json_fields": dynamic_json_fields,
        "total_files": total_valid,
        "errors": total_errors,
    }
    with open(SCHEMA_OUT, "w") as f:
        json.dump(schema, f, indent=2)

    size = SCHEMA_OUT.stat().st_size
    print(f"\nSaved schema to {SCHEMA_OUT} ({human(size)})")


if __name__ == "__main__":
    main()