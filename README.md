> **NOTA: Questo file è il template ufficiale per il README tecnico della vostra repository.**  
> Prima di iniziare, assicuratevi di aver letto attentamente le **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.  
> Questo file deve contenere **esclusivamente gli aspetti tecnici** del progetto (Setup, Run, Risultati base). La relazione testuale e teorica va inserita nel file **[`docs/REPORT.md`](docs/REPORT.md)**.
> *Cancellate questo blocco nota prima della consegna.*

# [Titolo del Progetto Assegnato]

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Informazioni su Gruppo e Progetto
- **Group ID**: [Es. G07]
- **ID Progetto**: [Es. 1]

## 📝 Descrizione del Progetto
Breve paragrafo (3-4 righe) che descrive visivamente e sinteticamente il progetto, il modello principale implementato e il task affrontato. 
*(Immaginate che questo sia l'Abstract tecnico della vostra repo GitHub).*

> 📖 **Relazione Ufficiale**: Per tutti i dettagli teorici, l'analisi delle performance, l'architettura utilizzata e i contributi del gruppo, fate riferimento al nostro elaborato formale: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Riproducibilità Tecnica

### 1. Dati e Setup dell'Ambiente

**Prerequisiti:**
Spiegate come chi legge può installare l'ambiente per far girare il vostro codice.

```bash
git clone https://github.com/vostronomer/vostro-repo.git
cd vostro-repo
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
Spiegate in 2 righe da dove si scaricano i dati e in quale cartella devono risiedere (es. `data/raw/`).

### 2. Training delle Reti
Fornite i **comandi esatti** per far partire l'addestramento.

**Training Baseline:**
```bash
python -m src.training.train --config experiments/configs/baseline.yaml
```

**Training Modello Migliorato:**
```bash
python -m src.training.train --config experiments/configs/model_v1.yaml
```

### 3. Valutazione (Evaluation)
Fornite i comandi per riprodurre i numeri della tabella riassuntiva.

```bash
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```

---

*Per la dichiarazione dei task individuali e l'uso di AI, riferirsi a `docs/REPORT.md`.*
