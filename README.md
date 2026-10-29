---
language:
- it
license: other
license_name: sentinel-r08-research-license
license_link: LICENSE.md
tags:
- pytorch
- causal-lm
- geodesic-neural-network
- sentinel-lifeos
- rob
- r08
- italian
pipeline_tag: text-generation
inference: false
---

# R08 (Rob) — Geodesic Mesh Neural Network (Pyramidal Sphere 50M)
### *Il Cuore Neurale Geometrico di SENTINEL LifeOS*

[![Model Name](https://img.shields.io/badge/Model-R08%20(Rob)-orange.svg)](https://github.com/Thomasseri/R-08)
[![Official Site](https://img.shields.io/badge/Official%20Site-sentinelhub.eu-007acc.svg)](https://www.sentinelhub.eu)
[![Ecosystem](https://img.shields.io/badge/Ecosystem-Sentinel%20LifeOS-purple.svg)](https://www.sentinelhub.eu)
[![Parameters](https://img.shields.io/badge/Parameters-45M-blue.svg)](https://github.com/Thomasseri/R-08)
[![Architecture](https://img.shields.io/badge/Architecture-Geodesic%20Pyramidal%20Sphere-brightgreen.svg)](https://github.com/Thomasseri/R-08)
[![Language](https://img.shields.io/badge/Language-Italian%20(IT)-yellow.svg)](https://github.com/Thomasseri/R-08)
[![License](https://img.shields.io/badge/License-Non--Commercial%20Research-lightgrey.svg)](LICENSE.md)

**Nome Ufficiale del Modello:** **R08** *(per gli amici **Rob**, da Robot)*  
**Progetto Padre & Ecosistema:** **[Sentinel LifeOS](https://www.sentinelhub.eu)** *(Il Sistema Operativo di Vita e Gestione)*  
**Sito Ufficiale:** **[www.sentinelhub.eu](https://www.sentinelhub.eu)**  
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
   - **R08** compete e supera Transformer standard a **12-14 blocchi pesanti di attenzione**, impiegando **soli 2 strati di attenzione causale densa**.

4. **Cascata Multilivello a 3 Stadi in Serie**:
   - **Stadio 1 (Pezzi):** Subword grain fine (1.251 token).
   - **Stadio 2 (Parole):** Lessico esteso (17.908 token).
   - **Stadio 3 (Unioni):** Sintagmi e unioni concettuali (20.006 token).

5. **Integrazione con Sentinel Knowledge Ecosystem**:
   - R08 è progettato nativamente per accoppiarsi al motore ontologico deterministico a 4 livelli esclusivo della piattaforma [Sentinel LifeOS](https://www.sentinelhub.eu), garantendo precisione assoluta su codici ricambio, coppie di serraggio (Nm) e parametri tecnici a zero allucinazioni.

---

## 📊 Benchmark Scientifici Reali: R08 vs Big Tech Open Source (~50M - 125M)

Valutazione scientifica completa eseguita sull'**intero set di test** (zero campionamento, 3.481 coppie BLiMP-IT e 2.916 coppie MultiBLiMP):

| Modello | Creatore | Parametri | Strati Attenzione Densa | Peso Checkpoint | BLiMP-IT (Accuratezza Linguistica) | MultiBLiMP (Ragionamento Logico) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OPT-125M** | Meta (Facebook AI) | 125M *(2.8× R08)* | 12 strati | ~500 MB | 62.5% | 58.0% |
| **GPT-Neo-125M** | EleutherAI | 125M *(2.8× R08)* | 12 strati | ~500 MB | 68.2% | 61.5% |
| **Pythia-70M** | EleutherAI | 70M *(1.6× R08)* | 14 strati | ~280 MB | 73.8% | 64.0% |
| **GPT-2 Small** | OpenAI | 124M *(2.7× R08)* | 12 strati | ~498 MB | 74.5% - 80.1% | 66.8% |
| 🏆 **R08 (Rob)** | **Thomas Seri / Sentinel** | **44.9M** | **2 strati** | **171.7 MB** | **80.95%** | **73.46%** |

---

## 🚀 Utilizzo Rapido (Inference Quickstart)

Il modello può essere eseguito localmente su qualsiasi computer (Mac Metal MPS, GPU CUDA o CPU) in 3 righe di codice:

```python
from modeling_geodesic import R08Model, R08Tokenizer

# 1. Carica Tokenizer e Modello dai pesi safetensors
tokenizer = R08Tokenizer.from_pretrained("Thomasseri/R-08")
model = R08Model.from_pretrained("Thomasseri/R-08")

# 2. Genera testo con sampling e repetition penalty
prompt = "Utente: Ciao, come stai?\nAssistente:"
risposta = model.generate(
    prompt=prompt,
    tokenizer=tokenizer,
    max_new_tokens=48,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15
)

print(risposta)
```

---

## 🔒 Proprietà Intellettuale & Licenza

Il modello **R08** e il sistema **Sentinel LifeOS** sono distribuiti sotto la licenza **Sentinel R08 Non-Commercial Research License & Patent Reservation** (consultabile in [`LICENSE.md`](LICENSE.md)).

- ✅ **Consentito:** Studio personale, ricerca accademica, test di riproducibilità e citazione scientifica.
- ❌ **Vietato:** Qualsiasi utilizzo commerciale, vendita, sublicenza o incorporazione in prodotti for-profit senza licenza commerciale scritta.
- ⚖️ **Nota Brevettuale:** L'autore (Thomas Seri) si riserva esplicitamente tutti i diritti brevettuali presenti e futuri relativi alla topologia di instradamento geodetico, al centro piramidale e all'architettura R08.
