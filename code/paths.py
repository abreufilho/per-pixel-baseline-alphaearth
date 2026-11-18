#!/usr/bin/env python3
"""Caminhos canonicos dos artefatos do experimento.

Por padrao os dados ficam em ``data/`` na raiz do repo. Use
``ALPHAEARTH_MG_DATA=/outro/caminho`` para apontar os scripts para outra arvore.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ALPHAEARTH_MG_DATA", ROOT / "data")).expanduser().resolve()

REFERENCE_DIR = DATA_ROOT / "reference"
TILES_DIR = DATA_ROOT / "tiles"
RASTER_DIR = DATA_ROOT / "rasters" / "gee"
LABELS_DIR = DATA_ROOT / "labels"
PROB_RF_DIR = DATA_ROOT / "probabilities" / "rf"
PROB_LIN_DIR = DATA_ROOT / "probabilities" / "lin"
METRICS_RF_DIR = DATA_ROOT / "metrics" / "rf"
METRICS_LIN_DIR = DATA_ROOT / "metrics" / "lin"
VIEW_DIR = DATA_ROOT / "view"
FIGURES_DIR = ROOT / "manuscript" / "figures"

REF = REFERENCE_DIR / "ref_cafe_mg_2017.parquet"
TILES = TILES_DIR / "tiles_eval.geojson"
SPLIT = TILES_DIR / "split.json"

SOURCES = ("s1", "s2", "ae")
EXPECTED_BANDS = {"s1": 8, "s2": 16, "ae": 64}
EXPERIMENTS = ("E1", "E2", "E3", "E4")


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _make_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    tmp.unlink(missing_ok=True)
    return tmp


def _cleanup_temp(path: Path) -> None:
    path.unlink(missing_ok=True)
    for sidecar in path.parent.glob(f"{path.name}.*"):
        if sidecar.is_file():
            sidecar.unlink(missing_ok=True)


@contextmanager
def staged_outputs():
    """Stage multiple outputs and atomically publish them together on success."""
    staged: list[tuple[Path, Path]] = []
    committed = False

    def stage(path: Path) -> Path:
        final = Path(path)
        tmp = _make_temp_path(final)
        staged.append((tmp, final))
        return tmp

    try:
        yield stage
        missing = [tmp for tmp, _ in staged if not tmp.exists()]
        if missing:
            missing_list = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"saida temporaria nao criada: {missing_list}")
        for tmp, final in staged:
            os.replace(tmp, final)
        committed = True
    finally:
        if not committed:
            for tmp, _ in staged:
                _cleanup_temp(tmp)


@contextmanager
def atomic_output(path: Path):
    """Write one output through a temp file and replace the final path only on success."""
    with staged_outputs() as stage:
        yield stage(path)


def atomic_write_text(path: Path, text: str) -> None:
    with atomic_output(path) as tmp:
        tmp.write_text(text)


def raster_path(tile_id: str, source: str) -> Path:
    return RASTER_DIR / f"{tile_id}_{source}.tif"


def label_path(tile_id: str) -> Path:
    return LABELS_DIR / f"labels_{tile_id}.tif"


def prob_path(tile_id: str, experiment: str, variant: str = "rf") -> Path:
    if variant == "rf":
        return PROB_RF_DIR / f"prob_{tile_id}_{experiment}.tif"
    if variant == "lin":
        return PROB_LIN_DIR / f"prob_{tile_id}_{experiment}_lin.tif"
    raise ValueError(f"variant invalida: {variant}")


def prob_path_for_experiment(tile_id: str, experiment: str) -> Path:
    if experiment.endswith("_lin"):
        return prob_path(tile_id, experiment.removesuffix("_lin"), "lin")
    return prob_path(tile_id, experiment, "rf")


def metrics_dir(variant: str = "rf") -> Path:
    if variant == "rf":
        return METRICS_RF_DIR
    if variant == "lin":
        return METRICS_LIN_DIR
    raise ValueError(f"variant invalida: {variant}")


def metrics_suffix(variant: str = "rf") -> str:
    return "" if variant == "rf" else "_lin"


def metrics_path(kind: str, variant: str = "rf") -> Path:
    suffix = metrics_suffix(variant)
    names = {
        "summary": f"metrics_E1_E4{suffix}.csv",
        "per_tile": f"metrics_E1_E4{suffix}_per_tile.csv",
        "tests": f"metrics_E1_E4{suffix}_tests.csv",
    }
    if kind not in names:
        raise ValueError(f"kind invalido: {kind}")
    return metrics_dir(variant) / names[kind]


def thresholds_path(variant: str = "rf") -> Path:
    return metrics_dir(variant) / f"thresholds{metrics_suffix(variant)}.json"


# --- Controles de contexto espacial (E1c-E4c) e tiles confirmatórios -------------------------
# Protocolo: docs/protocol/confirmatory.md. data/ctx/ guarda E1c-E4c nos 80 tiles originais;
# data/confirm/ é uma árvore completa (tiles, rasters, labels) para os tiles confirmatórios.

CTX_ROOT = DATA_ROOT / "ctx"
CONFIRM_ROOT = DATA_ROOT / "confirm"
MIN_FREE_GB = 80.0
SCOPES = ("original", "confirm")


@dataclass(frozen=True)
class Layout:
    """Árvore de insumos (tiles, rasters, labels) com a mesma estrutura de data/."""

    root: Path

    @property
    def tiles(self) -> Path:
        return self.root / "tiles" / "tiles_eval.geojson"

    @property
    def split(self) -> Path:
        return self.root / "tiles" / "split.json"

    def raster(self, tile_id: str, source: str) -> Path:
        return self.root / "rasters" / "gee" / f"{tile_id}_{source}.tif"

    def label(self, tile_id: str) -> Path:
        return self.root / "labels" / f"labels_{tile_id}.tif"


ORIGINAL = Layout(DATA_ROOT)
CONFIRM = Layout(CONFIRM_ROOT)


def layout(scope: str) -> Layout:
    if scope not in SCOPES:
        raise ValueError(f"scope invalido: {scope}")
    return ORIGINAL if scope == "original" else CONFIRM


def scope_root(scope: str) -> Path:
    return CTX_ROOT if scope == "original" else CONFIRM_ROOT


VARIANT_SUFFIXES = {"lin": "_lin", "rf8": "_rf8"}


def exp_name(base: str, variant: str) -> str:
    """('E3c', 'lin') -> 'E3c_lin'; ('E1', 'rf') -> 'E1'."""
    return base + VARIANT_SUFFIXES.get(variant, "")


def split_variant(experiment: str) -> tuple[str, str]:
    """'E3c_lin' -> ('E3c', 'lin'); 'E3c_rf8' -> ('E3c', 'rf8'); 'E1' -> ('E1', 'rf')."""
    for variant, suf in VARIANT_SUFFIXES.items():
        if experiment.endswith(suf):
            return experiment.removesuffix(suf), variant
    return experiment, "rf"


def scoped_prob_path(scope: str, tile_id: str, experiment: str) -> Path:
    """Probabilidade de um experimento (sufixo _lin/_rf8 opcional) num escopo.

    original: E1-E4 (rf/lin) vêm de data/probabilities (passo 4); o resto de data/ctx/probabilities.
    confirm: tudo em data/confirm/probabilities.
    """
    base, variant = split_variant(experiment)
    if scope == "original" and not base.endswith("c") and variant in ("rf", "lin"):
        return prob_path(tile_id, base, variant)
    return scope_root(scope) / "probabilities" / variant / f"prob_{tile_id}_{base}.tif"


def scoped_metrics_dir(scope: str) -> Path:
    return scope_root(scope) / "metrics"


def check_disk(min_free_gb: float = MIN_FREE_GB) -> float:
    """Aborta se o volume de data/ tiver menos que min_free_gb livres. Retorna GB livres."""
    probe = DATA_ROOT if DATA_ROOT.exists() else ROOT
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < min_free_gb:
        raise SystemExit(f"espaco livre {free_gb:.1f} GB < {min_free_gb:.0f} GB; abortando")
    return free_gb
