# R08 (Rob) — Geodesic Mesh Neural Network (Pyramidal Sphere 50M)
### *Il Cuore Neurale Geometrico di SENTINEL LifeOS*

[![Model Name](https://img.shields.io/badge/Model-R08%20(Rob)-orange.svg)](https://github.com/Thomasseri/R-08)
[![Ecosystem](https://img.shields.io/badge/Ecosystem-Sentinel%20LifeOS-purple.svg)](https://github.com/Thomasseri/R-08)
[![Parameters](https://img.shields.io/badge/Parameters-45M-blue.svg)](https://github.com/Thomasseri/R-08)
[![Architecture](https://img.shields.io/badge/Architecture-Geodesic%20Pyramidal%20Sphere-brightgreen.svg)](https://github.com/Thomasseri/R-08)
[![Training Speed](https://img.shields.io/badge/Training%20Speed-10%2C200%20tok%2Fs%20(MPS)-red.svg)](https://github.com/Thomasseri/R-08)
[![Language](https://img.shields.io/badge/Language-Italian%20(IT)-yellow.svg)](https://github.com/Thomasseri/R-08)
[![License](https://img.shields.io/badge/License-Non--Commercial%20Research-lightgrey.svg)](LICENSE.md)

**Nome Ufficiale del Modello:** **R08** *(per gli amici **Rob**, da Robot)*  
**Progetto Padre:** **Sentinel LifeOS** *(Il Sistema Operativo di Vita e Gestione)*  
**Autore & Ideatore:** Thomas Seri  
**Hardware di Sviluppo:** Apple Silicon (Metal Performance Shaders - MPS)

---

## 🧭 Cos'è Sentinel LifeOS e perché nasce R08

**Sentinel** è più di un software: è un **LifeOS (Sistema Operativo di Vita)** progettato per unificare in un unico ecosistema gli **obiettivi personali**, il **controllo economico**, la **gestione aziendale**, l'**automazione industriale (PLC)** e la **manutenzione predittiva dei macchinari**.

Mentre i sistemi tradizionali sono frammentati tra decine di app ed Excel, Sentinel rimette la persona e l'azienda al centro. Per dare a Sentinel una vera intelligenza operativa autonoma, locale e non dipendente da cloud esterni, è nato **R08 (Rob)**: un modello neurale compatto da 45 Milioni di parametri basato su una geometria interamente nuova.

---

## 🌟 L'Innovazione Architetturale di R08

Mentre i Large Language Model convenzionali si affidano al solito Transformer "piatto" a matrici dense con complessità quadratica $O(N^2)$, **R08** introduce una topologia sferico-piramidale non-euclidea:

1. **Topologia a Sfera Geodetica (Buckminster Fuller)**:
   - I pesi e le rappresentazioni concettuali sono distribuiti sulle **80 facce triangolari di un icosaedro geodetico regolarizzato** a 42 vertici proiettato su una sfera 3D.
   - Ogni token viene instradato nello spazio geometrico sferico, favorendo un clustering naturale dei concetti affini e riducendo drasticamente le allucinazioni.

2. **Centro Piramidale (Pyramidal Center Hub a 1-Hop)**:
   - Al centro della sfera risiede un super-nodo piramidale connesso a tutte le 80 facce triangolari (753.408 parametri dedicati).
   - Consente lo scambio istantaneo di informazioni globali polo-polo in un solo passaggio, eliminando i colli di bottiglia.

3. **Efficienza Estrema (2 Strati di Attenzione vs 14 Strati del Transformer Standard)**:
   - **R08** compete testa a testa con Transformer standard a **14 blocchi pesanti di attenzione**, impiegando **soli 2 strati di attenzione causale densa**.

4. **Cascata Multilivello a 3 Stadi in Serie**:
   - **Stadio 1 (Pezzi):** Subword grain fine (1.251 token).
   - **Stadio 2 (Parole):** Lessico esteso (17.908 token).
   - **Stadio 3 (Unioni):** Sintagmi e unioni concettuali (20.006 token).

---

## 📊 Benchmark Diretti: R08 vs Transformer Baseline (~45M)

Test condotti a parità rigorosa di parametri (~45M), stesso dataset e stesso hardware Apple Silicon:

| Metrica / Compito | Transformer Standard (14 blocchi) | **R08 Geodesic Pyramid (2 strati)** | Risultato R08 |
|---|---|---|---|
| **Strati di Attenzione Densa** | 14 blocchi | **2 strati** | **-85% complessità attenzione** |
| **Loss di Training (stesso stream)** | 1.9751 | **1.7906** | **-0.18 nats (R08 vince nel 68% dei batch)** |
| **MultiBLiMP (Comprensione complessa IT)** | 54.9% | **56.8%** | **+1.85 punti a favore di R08** |
| **BLiMP-IT (Grammatica italiana)** | 79.7% | **79.2%** | Sostanziale parità ($\pm 0.5\%$) |
| **Throughput di Training su Mac** | ~9.800 tok/s | **~10.600 tok/s** | Throughput superiore su memoria unificata |

---

## 🚀 Utilizzo Rapido (Inference Quickstart)

```python
import torch
from modeling_geodesic import R08Model, R08Tokenizer

# 1. Carica modello R08 e tokenizer
model = R08Model.from_pretrained("Thomasseri/R-08")
tokenizer = R08Tokenizer.from_pretrained("Thomasseri/R-08")

# 2. Genera testo
prompt = "Utente: Come si effettua la manutenzione predittiva del cuscinetto?\nAssistente:"
input_ids = tokenizer.encode(prompt)

output_ids = model.generate(
    input_ids,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(output_ids))
```

---

## 🔒 Proprietà Intellettuale & Licenza

Il modello **R08** e il sistema **Sentinel LifeOS** sono distribuiti sotto la licenza **Sentinel R08 Non-Commercial Research License & Patent Reservation**.

- ✅ **Consentito:** Studio personale, ricerca accademica, test di riproducibilità e citazione scientifica.
- ❌ **Vietato:** Qualsiasi utilizzo commerciale, vendita, sublicenza o incorporazione in prodotti for-profit senza licenza commerciale scritta.
- ⚖️ **Nota Brevettuale:** L'autore (Thomas Seri) si riserva esplicitamente tutti i diritti brevettuali presenti e futuri relativi alla topologia di instradamento geodetico, al centro piramidale e all'architettura R08.

Per partnership aziendali, licenze commerciali o integrazione in Sentinel: **Thomas Seri**.
