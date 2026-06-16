# [Graph-based Metric Learning for Scene Understanding]

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: [G18]
- **Project ID**: [3]

## 📝 Project Description
L'obiettivo principale è verificare se è possibile superare il limite delle architetture CNN classiche (che riducono l'immagine ad un singolo vettore piatto ignorando le relazioni spaziali e semantiche) modellando esplicitamente le scene complesse sotto forma di grafi (Scene Graphs). Utilizzando modelli di Graph Neural Network, nello specifico reti GCN (Graph Convolutional Network) e reti GINE (Graph Isomorphism Network with Edge features), il sistema mappa nodi (oggetti) ed archi (relazioni) in uno spazio latente ottimizzato tramite Supervised Contrastive Loss per raggruppare scenari semanticamente vicini e permettere un recupero accurato.

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**


```bash
git clone[ https://github.com/yourusername/your-repo.git](https://github.com/LenionK/dl26-projects.git)
cd dl26-projects
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
Il progetto utilizza il dataset GQA (disponibile sul [sito ufficiale](https://cs.stanford.edu/people/dorarad/gqa/download.html)). Deve eseese inserito nella cartella .data/

**Etichettatura dataset**
Prima di poter passare al training è necessario eseguire la seguente istruzione per creare i file .csv richiesti nelle prossime istruzioni in quanto il dataset raw non contiene etichette.
```bash
python src/datasets/dataset_preprocessing.py
```



### 2. Network Training
Il repository mette a disposizione script dedicati sia per l'addestramento della baseline CNN, sia per l'addestramento dei modelli grafici basati su grafi (GCN e GINE). Ogni script accetta la propria configurazione tramite file YAML in cui è possibile definire i parametri di rete e la tipologia di loss (es. Supervised Contrastive Loss).

**Baseline Training (CNN standard basata su immagini):**
```bash
python baseline_train.py --config baseline_config.yaml
```

**GCN Model Training:**
```bash
python gcn_train.py --config gcn_config.yaml --loss supervised_contrastive
```

**GINE Model Training:**
```bash
python gine_train.py --config gine_config.yaml --loss supervised_contrastive
```

### 3. Evaluation
Provide the commands to reproduce the numbers in your summary table.

```bash
python baseline_eval.py --config baseline_config.yaml
python gcn_eval.py --config gcn_config.yaml
python gine_eval.py --config gine_config.yaml
```

---

*For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`.*
