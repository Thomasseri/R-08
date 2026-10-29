#!/usr/bin/env python3
"""Verifica inferenza di Sentinel Geodesic R-08 da safetensors."""
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "geodesic_mesh_nn"
sys.path.insert(0, str(REPO))

import pezzi_tokenizer as pt
from vocab_stages import load_bundle, align_stages_to_pezzi
from eval_stages_smoke import _realign_from_pezzi
from geodesic_stages_model import CascadeMeshModel
from geodesic_network import WalkConfig

print("1. Inizializzazione architettura CascadeMeshModel...")
walk = WalkConfig(max_hops=1, exit_threshold=0.85)
model = CascadeMeshModel(
    vocab_pezzi=1251,
    vocab_parole=17908,
    vocab_unioni=20006,
    d_model=256,
    frequency=2,
    walk=walk,
    n_attn_layers=2,
    n_heads=8,
    max_seq_len=512,
    compass_spiral=True,
    fast_walk=True,
    router_v6=True,
    hash_routing=True,
    no_walk=True,
    pyramid_center=True,
)

print("2. Caricamento pesi safetensors...")
weights = load_file(str(ROOT / "model.safetensors"))
model.load_state_dict(weights, strict=False)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f"Modello caricato su: {device}")

bundle = load_bundle(ROOT / "vocab_stages")
pezzi = bundle.pezzi
eos = int(getattr(pezzi, "eos_id", 2))
pad = int(getattr(pezzi, "pad_id", 3))

test_prompts = [
    "Di che colore è il cielo di giorno?",
    "Utente: Ciao, come stai?\nAssistente:",
]

print("\n--- TEST GENERAZIONE INFERENZA CON SAFETENSORS ---")
for prompt in test_prompts:
    enc = pt.encode_cascade(prompt, bundle)
    p, w, u = align_stages_to_pezzi(enc, bundle, causal=True)
    p, w, u = list(p), list(w), list(u)
    new = []
    with torch.inference_mode():
        for _ in range(32):
            logits, _ = model(
                torch.tensor([p[-256:]], dtype=torch.long, device=device),
                torch.tensor([w[-256:]], dtype=torch.long, device=device),
                torch.tensor([u[-256:]], dtype=torch.long, device=device),
                record_path=False,
            )
            nxt = int(logits[0, -1].argmax())
            if nxt in (eos, pad):
                break
            new.append(nxt)
            p.append(nxt)
            p, w, u = _realign_from_pezzi(p, bundle, causal_align=True)
            p, w, u = list(p), list(w), list(u)
    print(f"\nPrompt: {prompt}")
    print(f"Risposta: {pt.decode(new, pezzi)}")
print("\n✅ Test safetensors completato con successo!")
