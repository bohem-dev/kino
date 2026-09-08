#!/usr/bin/env python3
# CLI entry point for the embedding pipeline: subset -> embed -> visualize,
# driven by a named recipe from kino/embedding/recipes/. See ARCHITECTURE.md.

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Running this file directly (`python kino/embedding/build_dataset.py`) puts
# this file's own directory on sys.path, not the repo root -- the kino.*
# imports below would fail without this. No effect when imported normally.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kino.embedding import subset, harrier, visualize, word2vec
from kino.embedding.recipes import all_recipe_keys, resolve_recipe

RUNS_DIR = Path("out/embedding_analysis/runs")
SHARED_CACHE_PATH = Path("out/embedding_analysis/_cache/ranked_full.parquet")


def _run_embedder(embedder, csv_path, run_dir, harrier_model, device,
                   batch_size, w2v_vector_size, w2v_window, w2v_min_count, w2v_epochs):
    if embedder == "harrier":
        vec_out_dir = run_dir / "vectors" / "harrier"
        return harrier.main(
            csv_path=csv_path, out_dir=vec_out_dir,
            model_name=harrier_model, device=device, batch_size=batch_size, tag="run",
        )
    elif embedder == "word2vec":
        vec_out_dir = run_dir / "vectors" / "word2vec"
        model_out_dir = run_dir / "models" / "word2vec"
        return word2vec.main(
            csv_path=csv_path, out_dir=vec_out_dir, model_dir=model_out_dir,
            vector_size=w2v_vector_size, window=w2v_window, min_count=w2v_min_count,
            epochs=w2v_epochs, tag="run",
        )
    raise ValueError(f"Unknown embedder: {embedder}")


def _load_manifest(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Truncated/corrupt from a run that died mid-write -- treat as absent.
        return None


def _completed_embedder_runs(manifest, embedders, run_dir):
    # Returns {embedder: (vectors_path, ids_path)} for embedders whose
    # manifest entry AND on-disk files both check out. A dead process can
    # leave a manifest entry pointing at files that were never finished
    # writing, or a subset.csv that's gone missing -- re-verify existence
    # rather than trusting the manifest blindly.
    if manifest is None:
        return {}
    csv_path = Path(manifest.get("csv_path", ""))
    if not csv_path.exists():
        return {}
    done = {}
    for embedder in embedders:
        entry = manifest.get("embedders", {}).get(embedder)
        if not entry:
            continue
        vectors_path = Path(entry["vectors_path"])
        ids_path = Path(entry["movie_ids_path"])
        if vectors_path.exists() and ids_path.exists():
            done[embedder] = (vectors_path, ids_path)
    return done


def run_recipe(
    recipe_key,
    top_n,
    input_dir,
    runs_dir,
    cache_path,
    workers,
    rebuild_cache,
    cast_n,
    embedders,
    harrier_model,
    device,
    batch_size,
    w2v_vector_size,
    w2v_window,
    w2v_min_count,
    w2v_epochs,
    resume=False,
):
    # runs subset + every requested embedder for one recipe, returns a
    # list of run descriptors (one per embedder) for visualize.main_combined()
    recipe = resolve_recipe(recipe_key)
    run_dir = runs_dir / str(top_n) / recipe_key
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}\nRecipe: {recipe_key}  ({recipe['description']})\ngroups={recipe['groups']}\nrun_dir={run_dir}\n{'=' * 72}")

    existing_manifest = _load_manifest(run_dir) if resume else None
    done = _completed_embedder_runs(existing_manifest, embedders, run_dir) if resume else {}
    pending = [e for e in embedders if e not in done]

    if resume and done:
        print(f"--resume: already have {', '.join(done)} for {recipe_key} -- reusing.")

    if not pending:
        print(f"--resume: {recipe_key} fully done ({', '.join(embedders)}) -- skipping subset+embed.")
        csv_path = Path(existing_manifest["csv_path"])
        runs = []
        for embedder, (vectors_path, ids_path) in done.items():
            runs.append({
                "label": f"{recipe_key} / {embedder}",
                "csv_path": csv_path,
                "vectors_path": vectors_path,
                "movie_ids_path": ids_path,
            })
        return runs

    subset.main(
        input_dir=input_dir,
        workers=workers,
        top_n=top_n,
        output_dir=str(run_dir),
        output_name="subset.csv",
        cache_path=str(cache_path),
        rebuild_cache=rebuild_cache,
        recipe=recipe_key,
        cast_n=cast_n,
    )
    csv_path = run_dir / "subset.csv"

    manifest = {
        "recipe": recipe_key,
        "description": recipe["description"],
        "groups": recipe["groups"],
        "top_n": top_n,
        "csv_path": str(csv_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedders": {
            embedder: {"vectors_path": str(v), "movie_ids_path": str(i)}
            for embedder, (v, i) in done.items()
        },
    }

    embedder_kwargs = dict(
        csv_path=csv_path, run_dir=run_dir, harrier_model=harrier_model, device=device,
        batch_size=batch_size, w2v_vector_size=w2v_vector_size, w2v_window=w2v_window,
        w2v_min_count=w2v_min_count, w2v_epochs=w2v_epochs,
    )

    if len(pending) > 1:
        # harrier (GPU-bound) and word2vec (CPU-bound) don't contend for the
        # same resource -- safe to run concurrently, unlike the process
        # pools inside subset.py/visualize.py which already claim every core.
        print(f"Running {', '.join(pending)} concurrently...")
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {embedder: pool.submit(_run_embedder, embedder, **embedder_kwargs) for embedder in pending}
            results = {embedder: future.result() for embedder, future in futures.items()}
    else:
        results = {pending[0]: _run_embedder(pending[0], **embedder_kwargs)}

    runs = []
    for embedder, (vectors_path, ids_path) in done.items():
        runs.append({
            "label": f"{recipe_key} / {embedder}",
            "csv_path": csv_path,
            "vectors_path": vectors_path,
            "movie_ids_path": ids_path,
        })
    for embedder, (vectors_path, ids_path) in results.items():
        manifest["embedders"][embedder] = {
            "vectors_path": str(vectors_path),
            "movie_ids_path": str(ids_path),
        }
        runs.append({
            "label": f"{recipe_key} / {embedder}",
            "csv_path": csv_path,
            "vectors_path": vectors_path,
            "movie_ids_path": ids_path,
        })

    # Write the manifest right after this recipe's embedders finish (rather
    # than only at the very end of the whole invocation) so a shutdown
    # between recipes loses nothing already completed -- that's the
    # checkpoint --resume reads back.
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest -> {run_dir / 'manifest.json'}")
    return runs


def main():
    parser = argparse.ArgumentParser(description="Run subset -> embed -> visualize for one or more recipes")
    parser.add_argument("--top-n", type=int, default=subset.TOP_N_DEFAULT)
    parser.add_argument("--recipe", default=None,
                         help="Recipe name (JSON file in kino/embedding/recipes/, without .json), "
                              "'full' (every field group), or 'all' (every recipe, then 'full'). "
                              "Required unless --list-recipes.")
    parser.add_argument("--embedder", default="harrier", choices=["harrier", "word2vec", "both"])
    parser.add_argument("--input-dir", default=subset.RAW_DIR)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--cache-path", default=str(SHARED_CACHE_PATH),
                         help="Shared full-ranked-list cache, reused across every recipe/embedder at this --top-n")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=subset.MAX_WORKERS)
    parser.add_argument("--cast-n", type=int, default=5)
    parser.add_argument("--skip-visualize", action="store_true", help="Only run subset + embed")
    parser.add_argument("--sample", type=int, default=None, help="Passed through to visualize.py --sample")
    parser.add_argument("--viz-workers", type=int, default=None,
                         help="Worker processes for the UMAP pool (default: every core)")

    parser.add_argument("--harrier-model", default=harrier.MODEL_NAME)
    parser.add_argument("--device", default=harrier.DEVICE)
    parser.add_argument("--batch-size", type=harrier.parse_batch_size, default=harrier.BATCH_SIZE,
                         help="Fixed batch size, or 'auto' to grow/shrink around OOMs "
                              "and fit as much as the device can handle")

    parser.add_argument("--w2v-vector-size", type=int, default=word2vec.VECTOR_SIZE)
    parser.add_argument("--w2v-window", type=int, default=word2vec.WINDOW)
    parser.add_argument("--w2v-min-count", type=int, default=word2vec.MIN_COUNT)
    parser.add_argument("--w2v-epochs", type=int, default=word2vec.EPOCHS)

    parser.add_argument("--list-recipes", action="store_true", help="Print available recipes and exit")
    parser.add_argument("--resume", action="store_true",
                         help="Skip recipe/embedder combinations already completed in a prior interrupted "
                              "invocation (checked via each run's manifest.json + the files it points at), "
                              "and reuse cached PCA/UMAP/t-SNE fits when rebuilding the explorer")
    args = parser.parse_args()

    if args.list_recipes:
        from kino.embedding.recipes import load_all_recipes
        for key, recipe in load_all_recipes().items():
            print(f"  {key}: {recipe['description']}  (groups={recipe['groups']})")
        raise SystemExit(0)

    if not args.recipe:
        parser.error("--recipe is required (or pass --list-recipes)")

    embedders = ["harrier", "word2vec"] if args.embedder == "both" else [args.embedder]
    recipe_keys = all_recipe_keys(include_full=True) if args.recipe == "all" else [args.recipe]
    runs_dir = Path(args.runs_dir)

    common_kwargs = dict(
        top_n=args.top_n,
        input_dir=args.input_dir,
        runs_dir=runs_dir,
        cache_path=Path(args.cache_path),
        workers=args.workers,
        rebuild_cache=args.rebuild_cache,
        cast_n=args.cast_n,
        embedders=embedders,
        harrier_model=args.harrier_model,
        device=args.device,
        batch_size=args.batch_size,
        w2v_vector_size=args.w2v_vector_size,
        w2v_window=args.w2v_window,
        w2v_min_count=args.w2v_min_count,
        w2v_epochs=args.w2v_epochs,
        resume=args.resume,
    )

    all_runs = []
    for recipe_key in recipe_keys:
        all_runs.extend(run_recipe(recipe_key, **common_kwargs))

    print(f"\n{'=' * 72}\n{len(recipe_keys)} recipe(s), {len(all_runs)} run(s) total.")

    if not args.skip_visualize:
        explorer_path = runs_dir / str(args.top_n) / "explorer.html"
        print(f"\nBuilding combined explorer covering all {len(all_runs)} run(s)...")
        visualize.main_combined(all_runs, explorer_path, sample=args.sample, max_workers=args.viz_workers,
                                 resume=args.resume)
    else:
        print("(--skip-visualize: no explorer built)")

    print(f"\n{'=' * 72}\nDone.")
    for run in all_runs:
        print(f"  {run['label']}: {run['csv_path'].parent}")


if __name__ == "__main__":
    main()
