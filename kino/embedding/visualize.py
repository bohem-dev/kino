import os

# Must happen before numpy/sklearn/umap are imported anywhere in this
# process (and, more importantly, before they're imported in each spawned
# worker -- os.environ is inherited by spawned children at process
# creation, regardless of import order). Each worker fits one view; it
# doesn't need its own internal thread pool on top of the process pool.
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap.umap_ as umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------
# Defaults (all overridable via CLI -- see bottom of file)
# ---------------------------------------------------------------------

DATA_CSV = Path("out/embedding_analysis/top_10000_by_revenue.csv")
VECTORS_PATH = Path("out/embedding_analysis/vectors/harrier_0.6b_vectors_10k.npy")
MOVIE_IDS_PATH = Path("out/embedding_analysis/vectors/harrier_0.6b_movie_ids_10k.npy")
OUT_DIR = Path("out/embedding_analysis/projections")

RANDOM_STATE = 42

POINT_SIZE_2D = 7
POINT_SIZE_3D = 4

PLOTLY_COLOUR_SEQUENCE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#8DD3C7", "#FFFFB3",
]

TSNE_PERPLEXITIES = [5, 10, 20, 30, 50, 75]

# Each named preset is one UMAP layout.
# n_neighbors: low = tiny local neighbourhoods, high = more global structure.
# min_dist: low = tight visible clusters, high = more diffuse spread.
UMAP_PRESETS = {
    "Very local / tight": {"n_neighbors": 5, "min_dist": 0.01},
    "Local / tight": {"n_neighbors": 10, "min_dist": 0.03},
    "Balanced": {"n_neighbors": 20, "min_dist": 0.08},
    "Broad / balanced": {"n_neighbors": 40, "min_dist": 0.15},
    "Global / diffuse": {"n_neighbors": 75, "min_dist": 0.40},
}


# ---------------------------------------------------------------------
# Basic data helpers
# ---------------------------------------------------------------------

def primary_genre(value):
    # first genre from a pipe-separated TMDB genre column
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text.split("|")[0].strip() if text else "Unknown"


def slugify(value):
    result = value.lower()
    for old, new in {" ": "_", "/": "_", "\\": "_", ":": "", ";": "", ",": "", ".": "", "(": "", ")": ""}.items():
        result = result.replace(old, new)
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def build_hover_text(rows):
    # one hover string per row, computed once per genre group and reused
    # across every view -- only the x/y/z coordinates change between views

    def fmt(row):
        parts = [f"<b>{row.get('title', '')}</b>"]
        if pd.notna(row.get("year")):
            parts.append(f"Year: {int(row['year'])}")
        if row.get("genres"):
            parts.append(f"Genres: {row['genres']}")
        if row.get("directors"):
            parts.append(f"Director: {row['directors']}")
        if row.get("writers"):
            parts.append(f"Writers: {row['writers']}")
        if row.get("top_cast"):
            parts.append(f"Cast: {row['top_cast']}")
        if pd.notna(row.get("vote_average")):
            votes = row.get("vote_count", 0)
            votes = int(votes) if pd.notna(votes) else 0
            parts.append(f"Rating: {row['vote_average']:.1f} ({votes:,} votes)")
        if pd.notna(row.get("popularity")):
            parts.append(f"Popularity: {row['popularity']:.1f}")
        if pd.notna(row.get("runtime")):
            parts.append(f"Runtime: {int(row['runtime'])} min")
        return "<br>".join(parts)

    return rows.apply(fmt, axis=1).tolist()


# ---------------------------------------------------------------------
# View definitions + parallel computation
# ---------------------------------------------------------------------

def build_view_defs():
    defs = [{"key": "pca", "label": "PCA"}]
    for name, params in UMAP_PRESETS.items():
        key = f"umap_{slugify(name)}"
        defs.append({
            "key": key,
            "label": f"UMAP: {name} (nn={params['n_neighbors']}, md={params['min_dist']})",
        })
    for perplexity in TSNE_PERPLEXITIES:
        defs.append({"key": f"tsne_{perplexity}", "label": f"t-SNE perplexity={perplexity}"})
    return defs


def _build_jobs(view_defs):
    # flat (dims, key, method, params) job list -- every job is
    # independent, so it can be handed to any worker in any order
    jobs = []
    for dims in (2, 3):
        for view in view_defs:
            if view["key"] == "pca":
                jobs.append((dims, "pca", "pca", {}))
            elif view["key"].startswith("umap_"):
                name = view["key"][len("umap_"):]
                params = next(p for n, p in UMAP_PRESETS.items() if slugify(n) == name)
                jobs.append((dims, view["key"], "umap", params))
            else:  # tsne_<perplexity>
                perplexity = int(view["key"].split("_")[1])
                jobs.append((dims, view["key"], "tsne", {"perplexity": perplexity}))
    return jobs


_WORKER_EMBEDDINGS = None


def _init_worker(embeddings):
    # runs once per worker process, not once per job -- the embedding
    # matrix is pickled across the process boundary a fixed number of
    # times (= worker count), not once per fit
    global _WORKER_EMBEDDINGS
    _WORKER_EMBEDDINGS = embeddings


def _fit_view(job):
    dims, key, method, params = job
    X = _WORKER_EMBEDDINGS
    variance = None

    if method == "pca":
        model = PCA(n_components=dims, random_state=RANDOM_STATE)
        coords = model.fit_transform(X)
        variance = model.explained_variance_ratio_.sum()
    elif method == "umap":
        reducer = umap.UMAP(
            n_components=dims,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        coords = reducer.fit_transform(X)
    elif method == "tsne":
        tsne = TSNE(
            n_components=dims,
            perplexity=params["perplexity"],
            learning_rate="auto",
            init="pca",
            metric="cosine",
            max_iter=1500,
            random_state=RANDOM_STATE,
        )
        coords = tsne.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    return dims, key, coords, variance


def compute_all_views(embeddings, view_defs, max_workers=None):
    # fits every (dims, method) combination across a process pool sized
    # to the whole machine -- these 26 fits are fully independent
    jobs = _build_jobs(view_defs)
    max_workers = max_workers or (os.cpu_count() or 1)
    max_workers = min(max_workers, len(jobs))

    coords_by_dims = {2: {}, 3: {}}
    print(f"  Computing {len(jobs)} layouts across {max_workers} worker processes "
          f"(1 thread each -- {max_workers} total, matching the machine's core count)...")

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(embeddings,)) as pool:
        futures = {pool.submit(_fit_view, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            dims, key, coords, variance = future.result()
            coords_by_dims[dims][key] = coords
            done += 1
            if variance is not None:
                print(f"    [{done}/{len(jobs)}] PCA {dims}D explained variance: {variance:.2%}")
            else:
                print(f"    [{done}/{len(jobs)}] {dims}D {key} done")

    return coords_by_dims


# ---------------------------------------------------------------------
# Figure assembly: one dropdown-switchable figure per dimensionality
# ---------------------------------------------------------------------

def build_dropdown_figure(movies, coords_by_key, view_defs, dims, title_prefix):
    category_order = sorted(movies["primary_genre"].dropna().unique().tolist())
    color_map = {g: PLOTLY_COLOUR_SEQUENCE[i % len(PLOTLY_COLOUR_SEQUENCE)] for i, g in enumerate(category_order)}

    genre_masks = {g: (movies["primary_genre"] == g).to_numpy() for g in category_order}
    hover_text_by_genre = {g: build_hover_text(movies.loc[mask]) for g, mask in genre_masks.items()}

    marker_size = POINT_SIZE_2D if dims == 2 else POINT_SIZE_3D
    trace_cls = go.Scattergl if dims == 2 else go.Scatter3d

    fig = go.Figure()
    view_trace_slices = {}
    idx = 0

    for view in view_defs:
        coords = coords_by_key[view["key"]]
        start = idx
        is_first = view is view_defs[0]
        for genre in category_order:
            mask = genre_masks[genre]
            kwargs = dict(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=genre,
                legendgroup=genre,
                marker=dict(size=marker_size, color=color_map[genre], line=dict(width=0)),
                text=hover_text_by_genre[genre],
                hoverinfo="text",
                opacity=0.78,
                visible=is_first,
                showlegend=is_first,
            )
            if dims == 3:
                kwargs["z"] = coords[mask, 2]
            fig.add_trace(trace_cls(**kwargs))
            idx += 1
        view_trace_slices[view["key"]] = (start, idx)

    total_traces = idx
    buttons = []
    for view in view_defs:
        start, end = view_trace_slices[view["key"]]
        visible = [False] * total_traces
        for i in range(start, end):
            visible[i] = True
        buttons.append(dict(
            label=view["label"],
            method="update",
            args=[
                {"visible": visible, "showlegend": visible},
                {"title": f"{title_prefix}: {view['label']}"},
            ],
        ))

    layout_kwargs = dict(
        template="plotly_dark",
        title=f"{title_prefix}: {view_defs[0]['label']}",
        height=900 if dims == 2 else 950,
        legend_title_text="Primary genre",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.0,
            xanchor="left",
            y=1.1,
            yanchor="top",
            showactive=True,
        )],
        margin=dict(l=20, r=20, t=110, b=20) if dims == 2 else dict(l=0, r=0, t=110, b=0),
    )
    if dims == 3:
        layout_kwargs["scene"] = dict(
            xaxis_title="Dimension 1", yaxis_title="Dimension 2", zaxis_title="Dimension 3",
        )
    fig.update_layout(**layout_kwargs)
    return fig


# ---------------------------------------------------------------------
# Loading + one run's worth of figures
# ---------------------------------------------------------------------

def load_run(csv_path, vectors_path, movie_ids_path, sample=None):
    movies = pd.read_csv(csv_path)
    embeddings = np.load(vectors_path)
    movie_ids = np.load(movie_ids_path)

    assert embeddings.ndim == 2, "Expected a 2D embedding matrix."
    assert len(movies) == len(embeddings), "Movie dataframe row count does not match embedding matrix row count."
    assert len(movie_ids) == len(embeddings), "Movie ID count does not match embedding matrix row count."
    assert np.array_equal(movies["id"].to_numpy(), movie_ids), (
        "Movie IDs in CSV order do not match the saved ID array -- vector rows "
        "may be attached to the wrong movies."
    )

    if sample is not None and sample < len(movies):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(len(movies), size=sample, replace=False))
        movies = movies.iloc[idx].reset_index(drop=True)
        embeddings = embeddings[idx]
        print(f"  Subsampled to {sample:,} movies for a faster preview run.")

    movies = movies.copy()
    movies["primary_genre"] = movies["genres"].map(primary_genre) if "genres" in movies.columns else "Unknown"
    return movies, embeddings


def _projections_cache_path(vectors_path, sample):
    vectors_path = Path(vectors_path)
    suffix = f"sample{sample}" if sample else "full"
    return vectors_path.parent / f"{vectors_path.stem}_projections_cache_{suffix}.npz"


def _load_cached_views(cache_path):
    if not cache_path.exists():
        return None
    try:
        npz = np.load(cache_path)
        coords_by_dims = {2: {}, 3: {}}
        for name in npz.files:
            dims_str, key = name.split("__", 1)
            coords_by_dims[int(dims_str)][key] = npz[name]
        return coords_by_dims
    except (OSError, ValueError, EOFError):
        # Partial/corrupt file from a run that died mid-write -- recompute.
        return None


def _save_cached_views(cache_path, coords_by_dims):
    arrays = {
        f"{dims}__{key}": coords
        for dims, coords_by_key in coords_by_dims.items()
        for key, coords in coords_by_key.items()
    }
    # Write under a temp name then rename -- np.savez isn't atomic on its
    # own, and a kill mid-write would otherwise leave a corrupt cache file
    # that _load_cached_views has to detect and discard instead of reusing.
    tmp_path = cache_path.with_suffix(".npz.tmp")
    # Pass an open file handle rather than tmp_path directly -- np.savez
    # auto-appends ".npz" to string paths that don't already end in ".npz",
    # which would silently write to "*.npz.tmp.npz" instead of tmp_path.
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    tmp_path.rename(cache_path)


def build_run_figures(csv_path, vectors_path, movie_ids_path, title_prefix, sample=None, max_workers=None,
                       resume=False):
    # loads one run's data, computes every layout, returns its 2D/3D
    # figures plus a long-format coordinate dataframe
    movies, embeddings = load_run(csv_path, vectors_path, movie_ids_path, sample=sample)
    print(f"  Movies: {len(movies):,}   Embedding matrix: {embeddings.shape}   "
          f"Distinct primary genres: {movies['primary_genre'].nunique()}")

    view_defs = build_view_defs()
    cache_path = _projections_cache_path(vectors_path, sample)
    coords_by_dims = _load_cached_views(cache_path) if resume else None
    if coords_by_dims is not None:
        print(f"  --resume: loaded cached projections from {cache_path}")
    else:
        coords_by_dims = compute_all_views(embeddings, view_defs, max_workers=max_workers)
        _save_cached_views(cache_path, coords_by_dims)

    fig_2d = build_dropdown_figure(movies, coords_by_dims[2], view_defs, dims=2, title_prefix=title_prefix)
    fig_3d = build_dropdown_figure(movies, coords_by_dims[3], view_defs, dims=3, title_prefix=title_prefix)

    records = []
    for dims, coords_by_key in coords_by_dims.items():
        for view in view_defs:
            coords = coords_by_key[view["key"]]
            d = movies[["id", "title"]].copy()
            d["dims"] = dims
            d["view"] = view["key"]
            d["view_label"] = view["label"]
            d["x"] = coords[:, 0]
            d["y"] = coords[:, 1]
            d["z"] = coords[:, 2] if dims == 3 else np.nan
            records.append(d)
    coordinates = pd.concat(records, ignore_index=True)

    return fig_2d, fig_3d, coordinates


# ---------------------------------------------------------------------
# HTML assembly: single-run (Dimensions only) and multi-run (Run + Dimensions)
# ---------------------------------------------------------------------

_PAGE_STYLE = """
  body { background:#111111; color:#eeeeee; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:0; }
  #controls { padding:14px 24px; border-bottom:1px solid #333; display:flex; gap:24px; align-items:center; flex-wrap:wrap; }
  #controls label { font-size:15px; margin-right:8px; }
  #controls select { font-size:14px; padding:5px 10px; background:#222; color:#eee; border:1px solid #444; border-radius:4px; }
  .panel { display:none; }
  .panel.active { display:block; }
"""


def write_combined_html(fig_2d, fig_3d, out_path):
    # single-run page: one Dimensions selector toggling between two
    # embedded figures (each with its own Method dropdown already built in)
    div_2d = fig_2d.to_html(full_html=False, include_plotlyjs="cdn", div_id="plot-2d")
    div_3d = fig_3d.to_html(full_html=False, include_plotlyjs=False, div_id="plot-3d")

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Operation Kino: Projection Explorer</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<div id="controls">
  <label for="dimSelect">Dimensions:</label>
  <select id="dimSelect" onchange="showPanel(this.value)">
    <option value="plot-2d">2D</option>
    <option value="plot-3d">3D</option>
  </select>
</div>
<div id="panel-2d" class="panel active">{div_2d}</div>
<div id="panel-3d" class="panel">{div_3d}</div>
<script>
function showPanel(plotId) {{
  document.querySelectorAll('.panel').forEach(function(p) {{ p.classList.remove('active'); }});
  var panelId = plotId === 'plot-2d' ? 'panel-2d' : 'panel-3d';
  document.getElementById(panelId).classList.add('active');
  var plotDiv = document.getElementById(plotId);
  if (window.Plotly) {{ Plotly.Plots.resize(plotDiv); }}
}}
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def write_global_html(runs, out_path):
    # multi-run page: a Run selector on top of the Dimensions selector, so
    # every recipe/embedder combination lives in one pane -- no more
    # opening a separate tab per run to compare them.
    # runs = [{"label": ..., "fig_2d": ..., "fig_3d": ...}, ...]
    options = "\n".join(f'    <option value="{i}">{r["label"]}</option>' for i, r in enumerate(runs))

    panels = []
    for i, run in enumerate(runs):
        include_js = "cdn" if i == 0 else False
        panels.append(run["fig_2d"].to_html(full_html=False, include_plotlyjs=include_js, div_id=f"plot-{i}-2"))
        panels.append(run["fig_3d"].to_html(full_html=False, include_plotlyjs=False, div_id=f"plot-{i}-3"))

    panel_divs = "\n".join(
        f'<div id="panel-{i}-{dims}" class="panel{" active" if i == 0 and dims == 2 else ""}">{div}</div>'
        for i, run in enumerate(runs)
        for dims, div in ((2, panels[2 * i]), (3, panels[2 * i + 1]))
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Operation Kino: Projection Explorer</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<div id="controls">
  <label for="runSelect">Run:</label>
  <select id="runSelect" onchange="updatePanel()">
{options}
  </select>
  <label for="dimSelect">Dimensions:</label>
  <select id="dimSelect" onchange="updatePanel()">
    <option value="2">2D</option>
    <option value="3">3D</option>
  </select>
</div>
{panel_divs}
<script>
function updatePanel() {{
  var run = document.getElementById('runSelect').value;
  var dims = document.getElementById('dimSelect').value;
  document.querySelectorAll('.panel').forEach(function(p) {{ p.classList.remove('active'); }});
  var panelId = 'panel-' + run + '-' + dims;
  var panel = document.getElementById(panelId);
  panel.classList.add('active');
  var plotDiv = document.getElementById('plot-' + run + '-' + dims);
  if (window.Plotly && plotDiv) {{ Plotly.Plots.resize(plotDiv); }}
}}
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------
# Run-directory resolution (build_dataset.py output layout)
# ---------------------------------------------------------------------

def resolve_run_paths(run_dir, embedder):
    # locates subset.csv + the newest vectors/ids .npy pair inside a
    # build_dataset.py run directory, without needing to know the exact
    # filenames a given embedder uses (harrier vs. word2vec name theirs
    # differently)
    csv_path = run_dir / "subset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No subset.csv in {run_dir}")

    vec_dir = run_dir / "vectors" / embedder
    vector_matches = sorted(vec_dir.glob("*_vectors_*.npy"), key=lambda p: p.stat().st_mtime)
    id_matches = sorted(vec_dir.glob("*_movie_ids_*.npy"), key=lambda p: p.stat().st_mtime)
    if not vector_matches or not id_matches:
        raise FileNotFoundError(f"No {embedder} vectors found in {vec_dir}")

    return csv_path, vector_matches[-1], id_matches[-1]


# ---------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------

def main(csv_path=DATA_CSV, vectors_path=VECTORS_PATH, movie_ids_path=MOVIE_IDS_PATH,
         out_dir=OUT_DIR, tag=None, sample=None, max_workers=None, resume=False):
    # single-run mode: one HTML with a Dimensions selector
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or Path(vectors_path).stem

    print("Loading movies and embeddings...")
    fig_2d, fig_3d, coordinates = build_run_figures(
        csv_path, vectors_path, movie_ids_path,
        title_prefix="Operation Kino", sample=sample, max_workers=max_workers, resume=resume,
    )

    html_path = out_dir / f"{tag}_projection_explorer.html"
    write_combined_html(fig_2d, fig_3d, html_path)

    coordinates.to_parquet(out_dir / f"{tag}_all_projections.parquet", index=False)
    coordinates.to_csv(out_dir / f"{tag}_all_projections.csv", index=False)

    print("\nFinished.")
    print(f"Open -> {html_path}")
    print(f"Coordinates -> {out_dir / f'{tag}_all_projections.parquet'} / .csv")
    return html_path


def main_combined(runs, out_path, sample=None, max_workers=None, resume=False):
    # multi-run mode: one HTML with Run + Dimensions selectors.
    # runs = [{"label": ..., "csv_path": ..., "vectors_path": ..., "movie_ids_path": ...}, ...]
    # computed one run at a time -- each run's own 26-fit computation
    # already saturates the machine via compute_all_views's process pool,
    # so running multiple runs' pools concurrently would oversubscribe
    # instead of speeding anything up. With --resume, each run's fitted
    # projections are cached to disk (see _save_cached_views) so a
    # shutdown partway through this loop only costs the in-progress run.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    figures = []
    all_coords = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {run['label']}")
        fig_2d, fig_3d, coordinates = build_run_figures(
            run["csv_path"], run["vectors_path"], run["movie_ids_path"],
            title_prefix=run["label"], sample=sample, max_workers=max_workers, resume=resume,
        )
        figures.append({"label": run["label"], "fig_2d": fig_2d, "fig_3d": fig_3d})
        coordinates["run"] = run["label"]
        all_coords.append(coordinates)

    write_global_html(figures, out_path)

    combined = pd.concat(all_coords, ignore_index=True)
    combined.to_parquet(out_path.with_suffix(".parquet"), index=False)
    combined.to_csv(out_path.with_suffix(".csv"), index=False)

    print("\nFinished.")
    print(f"Open -> {out_path}")
    print(f"Coordinates -> {out_path.with_suffix('.parquet')} / {out_path.with_suffix('.csv')}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA/UMAP/t-SNE projection explorer, single or multi-run")
    parser.add_argument("--run-dir", default=None,
                         help="A build_dataset.py run directory (out/embedding_analysis/runs/<top_n>/<recipe>) "
                              "-- resolves --csv/--vectors/--movie-ids/--out-dir/--tag automatically.")
    parser.add_argument("--run-dirs", nargs="+", default=None,
                         help="Multiple run directories to combine into ONE explorer with a Run selector, "
                              "instead of a separate file per run. Overrides --run-dir/--csv/--vectors/--movie-ids.")
    parser.add_argument("--combined-out", default=None,
                         help="Output HTML path for --run-dirs (default: <common parent>/explorer.html)")
    parser.add_argument("--embedder", default="harrier", choices=["harrier", "word2vec"],
                         help="Which embedder's output to load when using --run-dir/--run-dirs")
    parser.add_argument("--csv", default=str(DATA_CSV), help="subset.py output CSV")
    parser.add_argument("--vectors", default=str(VECTORS_PATH), help="Embedding vectors .npy")
    parser.add_argument("--movie-ids", default=str(MOVIE_IDS_PATH), help="Aligned movie ids .npy")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default=None, help="Filename prefix (default: vectors filename stem)")
    parser.add_argument("--sample", type=int, default=None,
                         help="Subsample to N movies for a faster preview run (e.g. while iterating)")
    parser.add_argument("--workers", type=int, default=None,
                         help="Worker processes for the PCA/UMAP/t-SNE pool (default: every core)")
    parser.add_argument("--resume", action="store_true",
                         help="Reuse cached per-run PCA/UMAP/t-SNE fits from a prior interrupted run "
                              "instead of recomputing them")
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
        runs = []
        for run_dir in run_dirs:
            csv_path, vectors_path, movie_ids_path = resolve_run_paths(run_dir, args.embedder)
            runs.append({
                "label": f"{run_dir.name} / {args.embedder}",
                "csv_path": csv_path,
                "vectors_path": vectors_path,
                "movie_ids_path": movie_ids_path,
            })
        combined_out = Path(args.combined_out) if args.combined_out else run_dirs[0].parent / "explorer.html"
        main_combined(runs, combined_out, sample=args.sample, max_workers=args.workers, resume=args.resume)
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
            csv_path, vectors_path, movie_ids_path = resolve_run_paths(run_dir, args.embedder)
            out_dir = run_dir / "projections" / args.embedder
            tag = args.tag or args.embedder
        else:
            csv_path = Path(args.csv)
            vectors_path = Path(args.vectors)
            movie_ids_path = Path(args.movie_ids)
            out_dir = Path(args.out_dir)
            tag = args.tag

        main(
            csv_path=csv_path,
            vectors_path=vectors_path,
            movie_ids_path=movie_ids_path,
            out_dir=out_dir,
            tag=tag,
            sample=args.sample,
            max_workers=args.workers,
            resume=args.resume,
        )
