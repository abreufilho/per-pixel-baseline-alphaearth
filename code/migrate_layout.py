#!/usr/bin/env python3
"""Migra artefatos de scratch/data para o layout canonico data/.

Por padrao faz dry-run. Use --apply para mover os arquivos. O script nao recalcula
nada e ignora backups historicos em scratch/data/_bak_*.
"""
import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from paths import (
    DATA_ROOT,
    LABELS_DIR,
    METRICS_LIN_DIR,
    METRICS_RF_DIR,
    PROB_LIN_DIR,
    PROB_RF_DIR,
    RASTER_DIR,
    REF,
    ROOT,
    SPLIT,
    TILES,
    VIEW_DIR,
)

OLD_DATA = ROOT / "scratch" / "data"


@dataclass
class Move:
    src: Path
    dst: Path


def collect_moves() -> list[Move]:
    moves: list[Move] = []

    fixed = {
        OLD_DATA / "ref_cafe_mg_2017.parquet": REF,
        OLD_DATA / "tiles_eval.geojson": TILES,
        OLD_DATA / "split.json": SPLIT,
    }
    moves.extend(Move(src, dst) for src, dst in fixed.items() if src.exists())

    for src in sorted((OLD_DATA / "insumos").glob("*.tif")):
        moves.append(Move(src, RASTER_DIR / src.name))
    for src in sorted((OLD_DATA / "insumos").glob("*.tif.aux.xml")):
        moves.append(Move(src, RASTER_DIR / src.name))
    for src in sorted(OLD_DATA.glob("labels_*.tif")):
        moves.append(Move(src, LABELS_DIR / src.name))
    for src in sorted(OLD_DATA.glob("prob_*.tif")):
        dst_dir = PROB_LIN_DIR if src.stem.endswith("_lin") else PROB_RF_DIR
        moves.append(Move(src, dst_dir / src.name))
    for src in sorted(OLD_DATA.glob("metrics_E1_E4*.csv")):
        dst_dir = METRICS_LIN_DIR if "_lin" in src.name else METRICS_RF_DIR
        moves.append(Move(src, dst_dir / src.name))
    for src in sorted(OLD_DATA.glob("thresholds*.json")):
        dst_dir = METRICS_LIN_DIR if src.name == "thresholds_lin.json" else METRICS_RF_DIR
        moves.append(Move(src, dst_dir / src.name))
    for src in sorted((OLD_DATA / "view").glob("*.gpkg")):
        moves.append(Move(src, VIEW_DIR / src.name))

    return moves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="move os arquivos; sem isso faz dry-run")
    args = ap.parse_args()

    moves = collect_moves()
    if not moves:
        print(f"nenhum artefato a migrar em {OLD_DATA}")
        return

    collisions = [m for m in moves if m.dst.exists()]
    if collisions:
        print("destinos ja existem; nao vou sobrescrever:")
        for move in collisions[:20]:
            print(f"  {move.dst}")
        if len(collisions) > 20:
            print(f"  ... +{len(collisions) - 20}")
        raise SystemExit(1)

    total_bytes = sum(m.src.stat().st_size for m in moves)
    action = "MOVE" if args.apply else "DRY"
    print(f"{action}: {len(moves)} arquivos, {total_bytes / 1e9:.2f} GB")
    for move in moves:
        print(f"  {move.src} -> {move.dst}")

    if not args.apply:
        return

    manifest_path = DATA_ROOT / "layout_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for move in moves:
        move.dst.parent.mkdir(parents=True, exist_ok=True)
        size = move.src.stat().st_size
        shutil.move(str(move.src), str(move.dst))
        manifest.append({"src": str(move.src), "dst": str(move.dst), "bytes": size})

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"✓ migrado para {DATA_ROOT}")
    print(f"✓ manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
