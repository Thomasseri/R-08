# Sentinel Hierarchical Knowledge Tree
### *Motore Ontologico Deterministico a 4 Livelli (Zero-Allucinazioni)*

Questo modulo implementa la componente **simbolica e deterministica** di **Sentinel LifeOS** ([www.sentinelhub.eu](https://www.sentinelhub.eu)), operando in simbiosi con il modello neurale **R08 (Rob)**.

---

## 🏗️ La Gerarchia a 4 Livelli

A differenza dei sistemi RAG convenzionali basati su ricerca vettoriale approssimativa, il **Knowledge Tree di Sentinel** organizza la conoscenza industriale su un'alberatura rigorosa:

```text
LIVELLO 1: ARGOMENTO (Topic)
└── "Manutenzione Industriale" / "Automazione & Elettrico"

    LIVELLO 2: CATEGORIA (Category)
    └── "Meccanica & Idraulica" / "PLC" / "Sensoristica"

        LIVELLO 3: SUB-ARGOMENTO & COMPONENTE (Entity)
        └── "Pompa Linea 1" / "Siemens LOGO! 0BA8" / "Sensore IFM VSA001"

            LIVELLO 4: RISPOSTE, DATI CERTI & PROCEDURE (%)
            ├── Codici ricambio certificati (es. SKF 6204-2RSH)
            ├── Ore garantite dal costruttore (MTBF)
            ├── Soglie fisiche d'allarme (ISO 10816)
            ├── Coppie di serraggio (Nm) e tolleranze
            └── Checklist di intervento sequenziale
```

---

## ⚡ Caratteristiche Chiave

1. **Zero Allucinazioni:** I dati tecnici critici (coppie di serraggio, codici ricambi, pinout PLC) non vengono inventati dalla rete neurale: vengono prelevati direttamente dai nodi certificati dell'albero.
2. **Tempo di Risposta Sub-Millisecondo:** Navigazione in memoria senza consumo di GPU, eseguibile anche su microprocessori a bordo quadro.
3. **Fascicolo Macchinario Modulare:** Permette di aggiornare o sostituire il manuale di un singolo sensore o componente mantenendo lo storico intatto.

---

## 💻 Utilizzo Rapido

```python
from knowledge_engine import SentinelKnowledgeTree

# Inizializza l'albero
tree = SentinelKnowledgeTree()

# Cerca specifiche di un componente
risultati = tree.search("cuscinetto pompa linea 1")
nodo = risultati[0]

print(f"Componente: {nodo['componente']}")
print(f"Codice Ricambio: {nodo['dati']['componenti_critici']['cuscinetto_lato_motore']['codice_ricambio']}")
print(f"Coppia di Serraggio: {nodo['dati']['componenti_critici']['cuscinetto_lato_motore']['coppia_serraggio_nm']} Nm")
```

---

**Autore:** Thomas Seri  
**Piattaforma:** [Sentinel LifeOS](https://www.sentinelhub.eu)  
**Licenza:** Sentinel R08 Non-Commercial Research License & Patent Reservation  

> **Nota Legale sui Marchi Registrati:**  
> Tutti i marchi, loghi e denominazioni commerciali di terzi citati a scopo di esempio o compatibilità (quali Siemens, SKF, Grundfos, IFM Electronic) appartengono ai rispettivi legittimi proprietari. Sono menzionati esclusivamente in conformità al principio del *fair use* descrittivo (Art. 21 CPI / Art. 14 EU Trade Mark Regulation) per fini di interoperabilità tecnica, manutenzione e identificazione dei ricambi.
