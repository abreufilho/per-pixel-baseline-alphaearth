#!/usr/bin/env python3
"""Passo 4 — probe supervisionado leve: RF idêntico por experimento prevê P(fronteira).

Desenho v2 (docs/legacy/STUDY-DESIGN.md): a referência CONAB vira supervisão. Por tile, rasterizamos os
talhões inteiramente no interior do tile em duas classes — fronteira (borda ±1 px) e interior.
Treino SÓ dentro dos talhões (referência esparsa: o exterior não é rotulável). Um Random Forest
com hiperparâmetros fixos e idênticos p/ E1-E4 (invariante a escala de features) é treinado nos
pixels das células de TREINO e prevê P(fronteira) nos tiles de validação/teste.

Split espacial por célula de região (0,5°): células inteiras → treino (~50% dos tiles),
validação (~20%, escolhe o limiar no passo 5) e teste (~30%), seed fixo. Sem vazamento.

Além do RF, um probe LINEAR (LogisticRegression, hiperparâmetros fixos idênticos p/ E1-E4)
é treinado nos MESMOS pixels, com padronização por banda usando estatísticas do TREINO —
responde se a informação de fronteira é linearmente decodificável do embedding.

Saídas (não-versionadas): data/tiles/split.json, data/labels/labels_{tile}.tif
(0=fora, 1=interior, 2=fronteira), probabilidades RF/linear em data/probabilities/, float32.

Uso: python code/04_probe.py
"""
import json
import random

import numpy as np
import pandas as pd
import rasterio

from paths import (
    LABELS_DIR,
    ORIGINAL,
    PROB_LIN_DIR,
    PROB_RF_DIR,
    REF,
    SPLIT,
    TILES,
    label_path,
    prob_path,
    raster_path,
    staged_outputs,
)
from objects import load_ref
from probing import EXP_SOURCES, SEED, load_stack, prob_profile, rasterize_labels, train_experiment

SPLIT_FRACS = {"train": 0.5, "val": 0.2}  # resto = test


def make_split(tiles: list[dict]) -> dict[str, str]:
    """Células inteiras por partição; greedy até atingir a fração de tiles."""
    by_cell: dict[str, list[str]] = {}
    for ft in tiles:
        p = ft["properties"]
        by_cell.setdefault(p["region"], []).append(p["tile_id"])
    cells = sorted(by_cell)
    random.Random(SEED).shuffle(cells)
    total = sum(len(v) for v in by_cell.values())
    split, done = {}, 0
    for cell in cells:
        if done < SPLIT_FRACS["train"] * total:
            part = "train"
        elif done < (SPLIT_FRACS["train"] + SPLIT_FRACS["val"]) * total:
            part = "val"
        else:
            part = "test"
        for tid in by_cell[cell]:
            split[tid] = part
        done += len(by_cell[cell])
    return split


def make_labels(tid: str, ref_all, stage) -> np.ndarray | None:
    """0=fora do footprint, 1=interior de talhão, 2=fronteira. Grava labels_{tid}.tif."""
    labels, profile = rasterize_labels(raster_path(tid, "s2"), ref_all)
    if labels is None:
        return None
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stage(label_path(tid)), "w", **profile) as dst:
        dst.write(labels, 1)
    return labels


def main() -> None:
    tiles = json.loads(TILES.read_text())["features"]
    tiles = [ft for ft in tiles
             if all(raster_path(ft["properties"]["tile_id"], s).exists()
                    for s in ("s1", "s2", "ae"))]
    split = make_split(tiles)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    counts = pd.Series(split).value_counts().to_dict()
    print(f"· split por célula: {counts} ({len(tiles)} tiles com insumos)")

    ref_all = load_ref(REF)
    labels_by_tile = {}
    with staged_outputs() as stage:
        for ft in tiles:
            tid = ft["properties"]["tile_id"]
            lab = make_labels(tid, ref_all, stage)
            if lab is None:
                split[tid] = "skip"  # sem talhão inteiro no interior — fora de todas as partições
                print(f"  {tid}: sem talhões inteiros no interior (skip)")
            else:
                labels_by_tile[tid] = lab
        stage(SPLIT).write_text(json.dumps(split, indent=1))

        train_ids = [t for t, p in split.items() if p == "train"]
        predict_ids = [t for t, p in split.items() if p in ("val", "test")]
        for exp in EXP_SOURCES:
            probes, shape_ = train_experiment(ORIGINAL, exp, train_ids, labels_by_tile)
            print(f"· {exp}: RF + linear treinados ({shape_[0]} px, {shape_[1]} features, "
                  f"{len(train_ids)} tiles de treino)")
            PROB_RF_DIR.mkdir(parents=True, exist_ok=True)
            PROB_LIN_DIR.mkdir(parents=True, exist_ok=True)
            for tid in predict_ids:
                stack = load_stack(ORIGINAL, tid, exp)
                profile = prob_profile(raster_path(tid, "s2"), *stack.shape[1:])
                for variant, prob in probes.predict(stack).items():
                    with rasterio.open(stage(prob_path(tid, exp, variant)), "w", **profile) as dst:
                        dst.write(prob, 1)
            print(f"  {exp}: P(fronteira) RF+lin gerada p/ {len(predict_ids)} tiles val/teste")


if __name__ == "__main__":
    main()
