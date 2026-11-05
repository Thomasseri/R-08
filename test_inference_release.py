#!/usr/bin/env python3
"""Test di inferenza ufficiale per il pacchetto standalone R08 (Rob) su Hugging Face / GitHub."""
from pathlib import Path
from modeling_geodesic import R08Model, R08Tokenizer

ROOT = Path(__file__).resolve().parent

def main():
    print("1. Caricamento tokenizer da vocab_stages...")
    tokenizer = R08Tokenizer.from_pretrained(ROOT)

    print("2. Caricamento modello R08 da model.safetensors...")
    model = R08Model.from_pretrained(ROOT, device="cpu")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modello caricato con successo! Parametri: {total_params:,}")

    prompts = [
        "Utente: Ciao, come stai?\nAssistente:",
        "La capitale d'Italia è",
        "Per riparare un cilindro pneumatico Festo occorre",
    ]

    for p in prompts:
        print(f"\n--- PROMPT: {p} ---")
        out = model.generate(p, tokenizer, max_new_tokens=40, temperature=0.7, top_p=0.9, repetition_penalty=1.15)
        print(f"RISPOSTA:\n{out.strip()}")

    print("\n✅ Verifica completata con successo!")

if __name__ == "__main__":
    main()
