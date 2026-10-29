#!/usr/bin/env python3
"""Esporta il checkpoint finale di v7 Pyramid in formato Hugging Face (safetensors + vocab)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent / "geodesic_mesh_nn"

import torch
from safetensors.torch import save_file

def export_checkpoint(ckpt_path: Path, out_dir: Path):
    print(f"Caricamento checkpoint da: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("model", raw)

    # Rimuovi eventuali prefissi inutili
    clean_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean_state_dict[k] = v.clone().contiguous()

    out_file = out_dir / "model.safetensors"
    print(f"Salvataggio safetensors in: {out_file}")
    save_file(clean_state_dict, str(out_file))

    # Copia file del vocabolario
    vocab_src = REPO_ROOT / "vocab_stages"
    vocab_dest = out_dir / "vocab_stages"
    if vocab_src.exists():
        print(f"Copia vocab_stages da {vocab_src} a {vocab_dest}")
        if vocab_dest.exists():
            shutil.rmtree(vocab_dest)
        shutil.copytree(vocab_src, vocab_dest)

    print("\n✅ Esportazione completata con successo!")
    print(f"Cartella pronta per Hugging Face e GitHub: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "geodesic_stages_attn_v7_chinchilla_fast" / "step_120000.pt",
        help="Percorso del checkpoint .pt da esportare",
    )
    args = parser.parse_args()
    export_checkpoint(args.ckpt, ROOT)
