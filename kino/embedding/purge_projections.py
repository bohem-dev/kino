import argparse
from pathlib import Path

RUNS_DIR = Path("out/embedding_analysis/runs")

# Everything here is cheap to regenerate from the existing subset.csv +
# vectors (just re-fits UMAP), unlike subset.csv/vectors/manifest.json
# which cost a full re-embed -- those are left untouched.
PATTERNS = (
    "*_projections_cache_*.npz",
    "*_projections_cache_*.npz.tmp",
    "explorer.html",
    "explorer.csv",
    "explorer.parquet",
)


def find_targets(runs_dir):
    targets = []
    for pattern in PATTERNS:
        targets.extend(runs_dir.rglob(pattern))
    return sorted(set(targets))


def main(runs_dir=RUNS_DIR, dry_run=False):
    runs_dir = Path(runs_dir)
    targets = find_targets(runs_dir)

    if not targets:
        print(f"Nothing to purge under {runs_dir}")
        return

    total_bytes = sum(p.stat().st_size for p in targets)
    for p in targets:
        print(f"  {'[dry-run] would delete' if dry_run else 'deleting'}: {p}")
    print(f"{len(targets)} file(s), {total_bytes / 1e6:.1f} MB")

    if dry_run:
        return

    for p in targets:
        p.unlink()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete stale UMAP/PCA/t-SNE projection caches and combined explorer "
                     "outputs under a runs dir, e.g. after changing which layouts get computed. "
                     "Leaves subset.csv/vectors/manifest.json (and --resume) untouched."
    )
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--dry-run", action="store_true", help="List what would be deleted, delete nothing")
    args = parser.parse_args()

    main(runs_dir=args.runs_dir, dry_run=args.dry_run)
