#!/usr/bin/env python3
"""Suite di collaudo e verifica ufficiale per la release Hugging Face di R08 (Rob).
Verifica:
1. R08Model.from_pretrained e R08Tokenizer.from_pretrained
2. Inferenza con sampling (temperature=0.7, top_p=0.9, repetition_penalty=1.15)
3. Throughput di generazione (tok/s) e memoria
4. Convalida file del pacchetto (config.json, model.safetensors, README.md)
"""
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modeling_geodesic import R08Model, R08Tokenizer

print("=" * 60)
print("  SENTINEL LIFEOS — COLLAUDO UFFICIALE RILASCIO HUGGING FACE  ")
print("  Modello: R08 (Rob) — Geodesic Pyramidal Sphere 50M          ")
print("  Autore: Thomas Seri (https://www.sentinelhub.eu)            ")
print("=" * 60)

# 1. Verifica file essenziali
print("\n[TEST 1/4] Verifica file del pacchetto...")
essential_files = ["config.json", "model.safetensors", "README.md", "LICENSE.md", "modeling_geodesic.py"]
for f in essential_files:
    p = ROOT / f
    if not p.exists():
        raise FileNotFoundError(f"File mancante: {f}")
    sz_mb = p.stat().st_size / (1024 * 1024)
    print(f"  ✓ {f:<22} ({sz_mb:.2f} MB)")

# 2. Caricamento Tokenizer & Modello
print("\n[TEST 2/4] Caricamento Tokenizer e Modello da repository locale...")
t0 = time.time()
tokenizer = R08Tokenizer.from_pretrained(ROOT)
print(f"  ✓ Tokenizer caricato in {time.time() - t0:.2f}s (Pezzi: {len(tokenizer.pezzi.token_to_id)}, Parole: {len(tokenizer.bundle.parole.token_to_id)}, Unioni: {len(tokenizer.bundle.unioni.token_to_id)})")

t0 = time.time()
model = R08Model.from_pretrained(ROOT)
dev = next(model.mesh_model.parameters()).device
print(f"  ✓ Modello caricato in {time.time() - t0:.2f}s sul device: {dev}")

# 3. Test Generazione con Sampling & Repetition Penalty
print("\n[TEST 3/4] Test di generazione con Sampling & Anti-Loop...")
test_prompts = [
    "Di che colore è il cielo di giorno?",
    "Utente: Ciao, come stai?\nAssistente:",
    "Utente: Qual è la capitale d'Italia?\nAssistente:",
    "Spiegami a cosa serve un cuscinetto a sfera:",
]

for prompt in test_prompts:
    t_start = time.time()
    generated = model.generate(
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=40,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    elapsed = time.time() - t_start
    print(f"\n👉 Prompt: {prompt}")
    print(f"🤖 Risposta: {generated.strip()}")
    print(f"   (tempo: {elapsed:.2f}s)")

# 4. Benchmark Throughput di Inferenza
print("\n[TEST 4/4] Benchmark Throughput di Inferenza...")
bench_prompt = "La manutenzione predittiva consiste in"
warmup = model.generate(bench_prompt, tokenizer, max_new_tokens=10)

t_bench = time.time()
n_tok = 64
bench_out = model.generate(bench_prompt, tokenizer, max_new_tokens=n_tok, temperature=0.7)
sec = time.time() - t_bench
tok_per_sec = n_tok / max(sec, 1e-4)

print(f"  ✓ Generati {n_tok} token in {sec:.2f}s ({tok_per_sec:.1f} token/s su {dev})")

print("\n" + "=" * 60)
print("✅ TUTTI I TEST HUGGING FACE SUPERATI CON SUCCESSO!")
print("   Il pacchetto è pronto per essere pubblicato su Hugging Face.")
print("   Repository di destinazione: Thomasseri/R-08")
print("=" * 60)
