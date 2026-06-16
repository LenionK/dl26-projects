# Graph-based Metric Learning for Scene Understanding

- **Gruppo ID**: G18
- **Progetto ID**: 3
- **Modulo di riferimento**: Metric Learning
- **Dataset**: GQA (Graph Question Answering)

---

## 1. Introduzione e Obiettivo

Le architetture CNN classiche elaborano un'immagine come un singolo vettore piatto, perdendo le relazioni spaziali e semantiche tra gli oggetti che compongono una scena. L'obiettivo di questo progetto è verificare se modellare esplicitamente le scene come **Scene Graphs** — grafi in cui i nodi rappresentano oggetti e gli archi le relazioni tra di essi — permetta di superare questo limite e di ottenere embedding più informativi per il **retrieval semantico di scene simili**.

L'ipotesi di partenza è che due scene strutturalmente simili (stessi oggetti e relazioni, indipendentemente dall'aspetto visivo pixel-level) debbano mappare in regioni vicine dello spazio latente. Per addestrare tale spazio metrico si utilizza la **Supervised Contrastive Loss**, che raggruppa embedding appartenenti alla stessa classe scenica e separa quelli di classi diverse.

Il sistema implementato comprende:

1. Una **baseline CNN** (ResNet-18) che ignora la struttura a grafo.
2. Un encoder **GCN** (Graph Convolutional Network) che elabora la topologia del grafo.
3. Un encoder **GINE** (Graph Isomorphism Network with Edge features) che incorpora anche le etichette semantiche degli archi.
4. Un pipeline di **valutazione retrieval** basato su FAISS.
5. Obiettivi extra: **analisi di robustezza** alle perturbazioni strutturali, **interpretabilità** tramite Grad-CAM e GNNExplainer, e **focal point analysis** sulle relazioni critiche.

---

## 2. Struttura del Repository

```
dl26-projects/
├── data/                          # Dataset GQA e CSV preprocessati
├── experiments/
│   ├── configs/                   # File YAML di configurazione
│   └── checkpoints/               # Pesi dei modelli addestrati
├── notebooks/                     # Esplorazione dati, visualizzazioni, analisi extra
├── src/
│   ├── datasets/                  # Dataset e preprocessing
│   ├── models/                    # Architetture neurali
│   ├── training/                  # Loop di training e loss
│   ├── evaluation/                # Metriche retrieval
│   └── utils/                     # Grad-CAM, perturbazioni, interpretabilità
├── environment.yml                # Ambiente Conda
└── README.md                      # Istruzioni di riproducibilità
```

Il codice di produzione risiede interamente in `src/`; i notebook servono per esplorazione, visualizzazione e analisi sperimentali aggiuntive.

---

## 3. Dati Utilizzati

### 3.1 Fonte

Il progetto utilizza il dataset **[GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)**, che fornisce per ogni immagine:

- Gli **oggetti** rilevati con nome, bounding box (`x`, `y`, `w`, `h`) e relazioni verso altri oggetti.
- Metadati di scena (`location`, `weather`, dimensioni immagine).

I file raw utilizzati sono:

- `data/sceneGraphs/train_sceneGraphs.json`
- `data/sceneGraphs/val_sceneGraphs.json`
- `data/images/` — immagini `.jpg`

### 3.2 Subset e Split

Per limitare i tempi di addestramento si utilizza il **30%** del dataset GQA, campionato con seed fisso (`random.seed(42)`):

| Split | Record totali GQA | Subset (30%) |
| :--- | ---: | ---: |
| Train | 74.942 | ~22.482 |
| Val   | 10.696 | ~3.208 |

I CSV risultanti (`data/df_train.csv`, `data/df_val.csv`) contengono le colonne `image_id`, `objects`, `labels` e metadati associati.

### 3.3 Etichettatura delle Scene

Poiché GQA non fornisce direttamente etichette di classe scenica adatte al metric learning, le etichette sono generate automaticamente con **CLIP (ViT-B/32)** confrontando ogni immagine con un set di prompt testuali. Le 14 classi sceniche definite sono:

| Classe | Esempio prompt CLIP |
| :--- | :--- |
| Interni | *a kitchen interior*, *a bedroom interior*, *a bathroom interior*, *a living room interior* |
| Urbano / trasporti | *a street scene in a city*, *a railway scene*, *an airplane flying in the sky* |
| Attività / natura | *a person playing a sport*, *a wild animal in nature*, *a beach scene*, *a countryside landscape* |
| Cibo / edifici | *a food scene*, *a building exterior*, *a restaurant dining scene* |

Il notebook `notebooks/analisi_dataset.ipynb` documenta l'analisi della distribuzione delle classi e la pulizia del dataset. La distribuzione risulta **sbilanciata** (es. *a bedroom interior*: ~417 campioni train vs *a kitchen interior*: ~298), motivo per cui si adotta un campionatore bilanciato durante il training.

### 3.4 Preprocessing dei Grafi

Il modulo `src/datasets/gqa_graph_dataset.py` converte ogni riga del DataFrame in un oggetto `torch_geometric.data.Data`:

**Nodi:**
- Tipo di oggetto codificato tramite vocabolario (`node_vocab`) costruito sul train set.
- Opzionalmente, 4 feature di bounding box normalizzate rispetto alle dimensioni massime della scena (`use_bbox=True`, `normalize_bbox=True`).

**Archi:**
- Derivati dalle relazioni GQA (`obj["relations"]`).
- Tipo di relazione codificato tramite `rel_vocab` (utilizzato da GINE).
- Grafi resi **bidirezionali** per favorire il flusso di informazione.

**Vocabolari:** `build_node_vocab()` e `build_rel_vocab()` estraggono rispettivamente tutti i nomi di oggetti e tipi di relazione presenti nel train set, garantendo coerenza tra train e validation.

---

## 4. Metodologia e Architetture

### 4.1 Baseline — EmbeddingNet (ResNet-18)

**File:** `src/models/baseline.py`

La baseline utilizza una **ResNet-18 pre-addestrata** su ImageNet come encoder visivo. L'ultimo layer fully-connected viene sostituito da una proiezione lineare che mappa le feature (512-d) in uno spazio embedding di dimensione 128, normalizzato con L2.

```
Immagine RGB (224×224) → ResNet-18 → FC(512→128) → L2-normalize → embedding
```

**Dataset:** `SceneDataset` in `src/datasets/datasets.py` carica le immagini con trasformazioni standard (resize, flip orizzontale, normalizzazione ImageNet).

### 4.2 GCN — GCNEmbeddingNet

**File:** `src/models/gnn.py`

Encoder a grafo basato su **Graph Convolutional Network**:

| Componente | Dettaglio |
| :--- | :--- |
| Node embedding | `nn.Embedding(num_node_types, 64)` + bbox opzionale |
| Node encoder | Linear → ReLU → LayerNorm → hidden_dim (256) |
| Convoluzioni | 3 layer `GCNConv(256→256)` con LayerNorm, ReLU, Dropout |
| Pooling | Concatenazione di **global mean pool** e **global max pool** |
| Proiettore | MLP → embedding 128-d, L2-normalizzato |

Il GCN aggrega informazioni strutturali ma **non utilizza esplicitamente le etichette semantiche degli archi** — solo la topologia.

### 4.3 GINE — GINEEmbeddingNet

**File:** `src/models/gnn.py`

Variante più espressiva che incorpora le **feature degli archi**:

| Componente | Dettaglio |
| :--- | :--- |
| Node embedding | Identico al GCN |
| Edge encoder | `nn.Embedding(edge_vocab_size, hidden_dim)` |
| Convoluzioni | 4 layer `GINEConv` con MLP interno ed `edge_dim=hidden_dim` |
| Pooling / Proiettore | Identici al GCN |

GINE può distinguere relazioni semanticamente diverse (es. *on*, *under*, *to the left of*) anche tra la stessa coppia di oggetti, modellando meglio la compositionalità della scena.

### 4.4 Funzioni di Loss

**File:** `src/training/contrastiveloss.py`

| Loss | Utilizzo | Descrizione |
| :--- | :--- | :--- |
| `supervised_contrastive_loss` | Baseline CNN | Contrastive loss supervisionata con temperatura τ; per ogni campione, massimizza la similarità con i positivi (stessa classe) nel batch |
| `supervised_contrastive_loss_GNN` | GCN / GINE | Variante dedicata ai modelli a grafo con normalizzazione esplicita e gestione dei batch senza positivi |
| `ProxyTripletLoss` | GCN (opzionale) | Ogni classe ha un proxy apprendibile; triplet loss con hardest negative proxy |

La loss principale è la **Supervised Contrastive Loss** con temperatura:
- Baseline: τ = 0.1
- GCN/GINE: τ = 0.07

### 4.5 Training

**File:** `src/training/train_model.py`, script dedicati in `src/training/`

| Parametro | Baseline | GCN / GINE |
| :--- | :--- | :--- |
| Optimizer | Adam (lr=1e-4) | Adam (lr=3e-4, weight_decay=1e-4) |
| Scheduler | — | CosineAnnealingLR |
| Epochs | 10 | 20 |
| Grad clipping | — | max_norm=1.0 |
| Batch sampling | BalancedBatchSampler (6 classi × 4 campioni) | BalancedGraphBatchSampler (tutte le classi × 4 campioni) |

Il **BalancedBatchSampler** campiona equamente le classi presenti nel batch, essenziale per la contrastive loss che richiede almeno un positivo per campione. I checkpoint salvano `model_state_dict`, `optimizer_state_dict`, `loss_history`, e per i modelli a grafo anche `node_vocab` e `rel_vocab`.

**Comandi di training:**

```bash
python src/training/baseline_train.py --config baseline_config.yaml
python src/training/gcn_train.py --config experiments/configs/gcn_config.yaml --loss supcon
python src/training/gine_train.py
```

---

## 5. Valutazione

### 5.1 Protocollo di Retrieval

**File:** `src/evaluation/retrieval.py`

La valutazione segue un protocollo di **retrieval instance-based**:

1. Si calcolano gli embedding di tutto il validation set.
2. Si costruisce un indice **FAISS** (Inner Product su vettori L2-normalizzati = similarità coseno).
3. Per ogni query si recuperano i k vicini più simili (escludendo la query stessa).
4. Si calcola se il vicino più prossimo appartiene alla stessa classe.

**Metriche implementate:**

| Metrica | Descrizione |
| :--- | :--- |
| acc@1 | Accuracy al primo vicino (Retrieval@1) |
| precision_macro / recall_macro | Macro-averaged sulle classi |
| recall@K / precision@K | Con K = 1, 5, 10 |
| MRR | Mean Reciprocal Rank |
| mAP | Mean Average Precision |

Sono disponibili anche funzioni di visualizzazione qualitativa (`plot_retrieval`, `plot_graph_retrieval`, `plot_tsne`).

**Comandi di valutazione:**

```bash
python src/evaluation/baseline_eval.py --config baseline_config.yaml
python src/evaluation/gcn_eval.py --config experiments/configs/gcn_config.yaml
python src/evaluation/gine_eval.py
```

---

## 6. Obiettivi Extra

### 6.1 Robustezza alle Perturbazioni Strutturali

**File:** `src/utils/perturb.py`, notebook `notebooks/perturbation_comparison.ipynb`

Si simula l'incompletezza o l'errore di un eventuale scene graph extractor rimuovendo casualmente nodi (e le relative relazioni) con probabilità crescente `drop_prob ∈ {0%, 30%, 50%, 70%}`.

La funzione `perturb_graph_objects()` mantiene almeno un nodo per grafo e filtra le relazioni verso nodi eliminati.

### 6.2 Interpretabilità — Grad-CAM

**File:** `src/utils/GradCAM.py`, `src/utils/GraphGradCAM.py`

| Metodo | Target | Descrizione |
| :--- | :--- | :--- |
| GradCAM++ | Baseline CNN | Heatmap sulle regioni dell'immagine rilevanti per l'embedding |
| Graph Grad-CAM | GCN / GINE | Importanza per nodo calcolata rispetto al **prototipo di classe** (centroide degli embedding) |
| GNNExplainer | GCN / GINE | Maschera di importanza sugli archi per tipo di relazione |

Il notebook `notebooks/GraphGradCAM.ipynb` implementa:
- Confronto GCN vs GINE per importanza nodi per classe.
- Studio globale delle relazioni più influenti (GINE: per tipo semantico; GCN: per coppia di tipi di nodo).
- **Focal Point Analysis**: misura quanto la rimozione di un tipo di relazione altera il ranking dei vicini (Ranking Disruption).

### 6.3 Analisi Focal Point

La **Ranking Disruption** quantifica l'impatto strutturale di ciascuna relazione:

```
Disruption = 1 − |top-k(base) ∩ top-k(perturbato)| / k
```

Un valore alto indica che quella relazione è un *single point of failure* per il retrieval. L'analisi su GINE ha identificato relazioni spaziali come *to the left of* tra le più critiche.

---

## 7. Risultati

### 7.1 Retrieval su Validation Set (grafo intatto, drop_prob = 0%)

Risultati ottenuti dal notebook `perturbation_comparison.ipynb` sul validation set (~3.208 campioni):

| Modello | acc@1 | precision_macro | recall_macro | mAP |
| :--- | :---: | :---: | :---: | :---: |
| **GCN** | **59.66%** | **55.98%** | **55.38%** | **65.94%** |
| **GINE** | 49.91% | 48.78% | 48.92% | 57.78% |

Il **GCN supera GINE** su tutte le metriche di retrieval nel setup sperimentale attuale. Una possibile spiegazione è che GINE, pur essendo più espressivo, richiede più dati o tuning iperparametrico per convergere; inoltre il vocabolario delle relazioni è ampio e sparsamente popolato, rendendo l'edge embedding più difficile da apprendere su un subset ridotto.

### 7.2 Robustezza alle Perturbazioni

Degradazione delle performance al crescere della probabilità di rimozione nodi:

| drop_prob | GCN acc@1 | GINE acc@1 | GCN mAP | GINE mAP |
| :---: | :---: | :---: | :---: | :---: |
| 0% | 59.66% | 49.91% | 65.94% | 57.78% |
| 30% | 55.92% | 44.98% | 62.36% | 53.15% |
| 50% | 51.03% | 41.52% | 58.21% | 50.47% |
| 70% | 45.14% | 35.82% | 52.99% | 45.54% |

**Osservazioni:**
- Il GCN degrada in modo più graduale ed mantiene un vantaggio consistente su GINE a ogni livello di perturbazione.
- Anche con il 70% dei nodi rimossi, il GCN mantiene acc@1 > 45%, suggerendo che la struttura residua del grafo conserva informazione discriminativa.
- GINE risulta più sensibile alle perturbazioni, coerentemente con la maggiore dipendenza dalle feature degli archi che vengono eliminate insieme ai nodi.

### 7.3 Baseline CNN

La baseline ResNet-18 è addestrata e valutabile tramite gli script dedicati. Essendo basata esclusivamente sull'aspetto visivo, ci si aspetta che eccella su classi fortemente correlate al contenuto pixel (es. *a beach scene*) ma fatichi a generalizzare su scene strutturalmente simili con appearance diversa — limite che i modelli a grafo mirano a superare.

---

## 8. Discussione

### Perché modellare i grafi?

Le scene non sono semplici collezioni di pixel: sono **composizioni di entità e relazioni**. Un GCN/GINE può riconoscere che due scene contengono {stove, refrigerator, cabinet} con relazioni spaziali simili, anche se illuminazione e colori differiscono. La baseline CNN deve inferire implicitamente tale struttura dai pixel, un compito più difficile con dati limitati.

### GCN vs GINE

| Aspetto | GCN | GINE |
| :--- | :--- | :--- |
| Feature archi | Solo topologia | Tipo semantico della relazione |
| Layer convoluzioni | 3 | 4 |
| Performance retrieval | Migliore nel setup attuale | Inferiore, più sensibile a perturbazioni |
| Interpretabilità relazioni | Per coppia tipi-nodo | Per tipo semantico (es. *on*, *wearing*) |

### Limiti attuali

1. **Subset ridotto**: solo il 30% di GQA, con classi sbilanciate.
2. **Etichette CLIP**: pseudo-label automatiche, non ground truth; possibili errori di classificazione.
3. **Scene graph ground truth**: si assume che GQA fornisca grafi perfetti; in produzione servirebbe un extractor automatico.
4. **Nessun confronto numerico baseline vs GNN** nel notebook principale (valutabile eseguendo `baseline_eval.py`).
5. **`dataset_preprocessing.py` vuoto**: il preprocessing è documentato nel notebook `analisi_dataset.ipynb`.

---

## 9. Conclusioni

Il progetto dimostra la fattibilità di un sistema di **metric learning su scene graphs** per il retrieval semantico di scene. L'encoder **GCN** raggiunge quasi il **60% di acc@1** sul validation set, superando GINE e mostrando buona robustezza alle perturbazioni strutturali.

Gli obiettivi extra sono stati affrontati con:
- Test di robustezza sistematico (perturbazione nodi).
- Visualizzazione dell'importanza di nodi e archi (Grad-CAM, GNNExplainer).
- Identificazione delle relazioni critiche (focal point analysis).

**Sviluppi futuri:**
- Addestrare su dataset completo GQA.
- Confronto diretto baseline CNN vs GCN vs GINE con tabella unificata.
- Scene graph extractor automatico (VLM / object detector).
- Fine-tuning iperparametri GINE (numero layer, edge embedding dim).
- Integrazione di augmentations strutturali durante il training.

---

## 10. Riproducibilità

### Ambiente

```bash
git clone https://github.com/LenionK/dl26-projects.git
cd dl26-projects
conda env create -f environment.yml
conda activate dl-project
```

Dipendenze principali: PyTorch, torchvision, PyTorch Geometric, FAISS, scikit-learn, pandas, matplotlib, CLIP (per etichettatura).

### Dataset

1. Scaricare GQA dal [sito ufficiale](https://cs.stanford.edu/people/dorarad/gqa/download.html).
2. Posizionare i file in `data/`.
3. Eseguire il preprocessing descritto in `notebooks/analisi_dataset.ipynb` per generare `df_train.csv` e `df_val.csv`.

### Pipeline completa

```bash
# Training
python src/training/baseline_train.py --config baseline_config.yaml
python src/training/gcn_train.py --config experiments/configs/gcn_config.yaml --loss supcon
python src/training/gine_train.py

# Valutazione
python src/evaluation/baseline_eval.py --config baseline_config.yaml
python src/evaluation/gcn_eval.py --config experiments/configs/gcn_config.yaml
python src/evaluation/gine_eval.py
```

---

## 11. Mappa del Codice Sorgente

| Modulo | File | Responsabilità |
| :--- | :--- | :--- |
| Dataset immagini | `src/datasets/datasets.py` | `SceneDataset`, `BalancedBatchSampler` |
| Dataset grafi | `src/datasets/gqa_graph_dataset.py` | `GQAGraphDataset`, vocabolari, `BalancedGraphBatchSampler` |
| Baseline | `src/models/baseline.py` | `EmbeddingNet` (ResNet-18) |
| Modelli GNN | `src/models/gnn.py` | `GCNEmbeddingNet`, `GINEEmbeddingNet` |
| Loss | `src/training/contrastiveloss.py` | Supervised Contrastive, Proxy Triplet |
| Training loop | `src/training/train_model.py` | `train_model`, `train_graph_model`, checkpoint |
| Script train | `src/training/baseline_train.py`, `gcn_train.py`, `gine_train.py` | Entry point con config YAML |
| Retrieval | `src/evaluation/retrieval.py` | FAISS, metriche, visualizzazioni |
| Eval script | `src/evaluation/baseline_eval.py`, `gcn_eval.py`, `gine_eval.py` | Valutazione end-to-end |
| Grad-CAM CNN | `src/utils/GradCAM.py` | GradCAM++ per baseline |
| Grad-CAM grafo | `src/utils/GraphGradCAM.py` | Graph Grad-CAM, GNNExplainer, focal point |
| Perturbazioni | `src/utils/perturb.py` | Rimozione stocastica nodi |
| Config | `experiments/configs/*.yaml` | Iperparametri per ogni modello |

---

### 12 Utilizzo dell'Intelligenza Artificiale

Strumenti di IA generativa (es. GitHub Copilot, ChatGPT, Cursor) sono stati utilizzati per:

- Scrittura di boilerplate code.
- Debugging e refactoring.
- Documentazione e redazione di parti di questa relazione.

Le scelte architetturali (GCN vs GINE, supervised contrastive loss, protocollo di valutazione retrieval, analisi focal point) e la responsabilità dei risultati sono del gruppo.

---

*Relazione generata a partire dall'analisi del codice sorgente e dei notebook del repository `dl26-projects`.*
