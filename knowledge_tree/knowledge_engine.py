#!/usr/bin/env python3
"""Sentinel Knowledge Engine — Motore di interrogazione gerarchica deterministica ad albero."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "machinery_maintenance_tree.json"

class SentinelKnowledgeTree:
    def __init__(self, data_path: Optional[Path] = None):
        self.path = data_path or DEFAULT_DATA
        with open(self.path, "r", encoding="utf-8") as f:
            self.raw = json.load(f)
        self.nodes: Dict[str, Any] = self.raw.get("nodes", {})

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Cerca nodi pertinenti nell'albero in base a parole chiave."""
        q = query.lower()
        results = []
        for node_id, data in self.nodes.items():
            # Calcola affinità per percorso e contenuto
            text_dump = json.dumps(data, ensure_ascii=False).lower()
            if any(term in text_dump for term in q.split()):
                score = sum(text_dump.count(term) for term in q.split())
                results.append({
                    "node_id": node_id,
                    "score": score,
                    "argomento": data.get("argomento"),
                    "categoria": data.get("categoria"),
                    "componente": data.get("componente"),
                    "dati": data
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_maintenance_facts(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Restituisce la scheda tecnica deterministica a zero allucinazioni."""
        return self.nodes.get(node_id)

if __name__ == "__main__":
    tree = SentinelKnowledgeTree()
    print("=== TEST SENTINEL KNOWLEDGE ENGINE ===")
    query = "cuscinetto pompa linea 1"
    print(f"Interrogazione: '{query}'\n")
    matches = tree.search(query)
    for m in matches:
        print(f"📍 Trovato nodo: {m['node_id']}")
        print(f"   Argomento:   {m['argomento']} > {m['categoria']}")
        print(f"   Componente:  {m['componente']}")
        dati = m['dati']
        if "componenti_critici" in dati:
            print(f"   Ricambio:    {dati['componenti_critici']}")
        if "procedura_intervento" in dati:
            print("   Procedura:")
            for step in dati["procedura_intervento"][:2]:
                print(f"     {step}")
        print(f"   Certezza Dati: {dati.get('affidabilita_dati_pct', 100)}%\n")
