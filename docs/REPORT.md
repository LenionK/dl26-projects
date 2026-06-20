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
| Batch sampling | BalancedBatchSampler (6 classi × 4 campioni) | BalancedGraphBatchSampler (6 classi × 4 campioni)|

Il **BalancedBatchSampler** campiona equamente le classi presenti nel batch, essenziale per la contrastive loss che richiede almeno un positivo per campione. I checkpoint salvano `model_state_dict`, `optimizer_state_dict`, `loss_history`, e per i modelli a grafo anche `node_vocab` e `rel_vocab`.

### Motivazioni delle Scelte di Ottimizzazione

Le scelte di ottimizzatore, scheduler e gradient clipping non sono casuali ma 
rispondono a un problema comune: il dataset di training contiene **label rumorose**, 
generate automaticamente da CLIP. Un'etichetta errata introduce un segnale di 
supervisione sbagliato che può destabilizzare il training in modi diversi, 
e ciascuna delle scelte adottate mitiga uno specifico aspetto di questo problema.

**Adam con Weight Decay**

Adam è stato scelto come ottimizzatore per la sua capacità di adattare il learning 
rate per ogni parametro in base alla storia dei gradienti, rendendolo più stabile 
rispetto a SGD vanilla in presenza di gradienti irregolari — condizione frequente 
quando alcune label sono errate. Il `weight_decay=1e-4` aggiunge una penalità 
sui pesi grandi, limitando la capacità del modello di memorizzare esempi rumorosi 
e spingendolo verso soluzioni più generali. In presenza di label sbagliate, un 
modello senza regolarizzazione tenderebbe ad overfittare il rumore anziché 
apprendere la struttura semantica sottostante.

**CosineAnnealingLR**

Lo scheduler riduce il learning rate seguendo una curva a coseno, da `lr_max` 
fino a `lr * 0.05` (5% del valore iniziale). Questo comportamento ha due effetti 
complementari in presenza di rumore:

- Nella fase iniziale, il learning rate alto permette al modello di **esplorare 
  ampiamente** lo spazio dei parametri, evitando di convergere prematuramente 
  su minimi locali indotti da label sbagliate.
- Nella fase finale, il learning rate basso permette di **affinare la convergenza** 
  su strutture corrette, riducendo l'influenza degli outlier rumorosi che 
  producono gradienti instabili.

Un learning rate fisso non offre questo compromesso: o è troppo alto e non converge, 
o è troppo basso e rimane bloccato in minimi locali causati dal rumore.

**Gradient Clipping (max_norm=1.0)**

Un esempio con label errata può generare un gradiente molto grande: il modello
riceve un segnale di supervisione sbagliato e tenta di correggere aggressivamente
i parametri per soddisfarlo. Questo problema è amplificato dalla natura della
Supervised Contrastive Loss, che calcola il gradiente **su tutto il batch
simultaneamente**. Se un batch contiene più esempi con label errate concentrati
sulla stessa classe — condizione plausibile con il BalancedGraphBatchSampler che
forza 4 campioni per classe e con classi visivamente simili come *bedroom* vs
*living room* — si verificano due errori che si sommano nel gradiente finale:

- **Falsi positivi**: esempi di classi diverse vengono trattati come positivi,
  il modello cerca di avvicinarli nello spazio embedding quando non dovrebbe.
- **Falsi negativi**: esempi della stessa classe vengono trattati come negativi,
  il modello cerca di allontanarli quando non dovrebbe.

Il risultato è un gradiente molto grande e nella direzione sbagliata. Senza
clipping, questo produce un passo enorme che sposta i parametri lontano da una
buona soluzione; il batch successivo (più pulito) deve correggere il danno,
rendendo il training instabile. Clippare la norma L2 a `1.0` limita il danno
che un cluster di esempi rumorosi può arrecare in un singolo step, lasciando
al training successivo la possibilità di correggere gradualmente.


**Sinergia tra le tre scelte**

| Tecnica | Problema affrontato |
| :--- | :--- |
| Adam + weight decay | Evita memorizzazione del rumore, regolarizza i pesi |
| CosineAnnealingLR | Esplora prima, converge su strutture corrette dopo |
| Gradient clipping | Limita il danno di singoli esempi con label errate |

Le tre tecniche agiscono su livelli diversi dello stesso problema: 
il weight decay interviene sulla **capacità** del modello, lo scheduler 
sulla **traiettoria** di ottimizzazione, il gradient clipping sulla 
**stabilità** dei singoli aggiornamenti. La loro combinazione rende 
il training robusto al rumore introdotto dall'etichettatura automatica via CLIP.

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

#### GradCAM++ — Baseline CNN

**File:** `src/utils/GradCAM.py`

Per la baseline ResNet-18 si utilizza **GradCAM++**, una variante migliorata di
GradCAM classico che pesa i gradienti con termini al secondo e terzo ordine,
producendo heatmap più precise per oggetti multipli nella stessa immagine.

Il meccanismo è il seguente:

1. Forward pass → embedding dell'immagine.
2. Score scalare calcolato come **prodotto scalare tra l'embedding e il prototipo della classe target** — il centroide degli embedding di tutti i campioni della stessa classe nel training set.
3. Backward pass → accumulo dei gradienti sull'ultimo layer convoluzionale.
4. Pesi α calcolati combinando gradienti al secondo e terzo ordine:

```python
score   = dot(embedding, prototype)           # proiezione sul prototipo
alpha   = dY² / (2·dY² + sum_A·dY³)
weights = (alpha * ReLU(dY)).sum(dim=(H,W))
cam     = ReLU((weights * A).sum(dim=C))
```

Questa scelta dello score è fondamentale per la coerenza con il paradigma di metric learning: GradCAM++ classico è progettato per la classificazione e usa il logit della classe target come score scalare, che non esiste in un sistema metric learning. Sostituirlo con il prodotto scalare `dot(embedding, prototipo)` misura direttamente quanto l'embedding dell'immagine è orientato verso il centroide della classe, che è esattamente il segnale ottimizzato dalla Supervised Contrastive Loss durante il training. Questa scelta rende inoltre GradCAM++ sulla baseline **semanticamente coerente** con il Graph Grad-CAM sui modelli a grafo, che usa lo stesso score scalare.

I prototipi vengono calcolati sul **training set** tramite `compute_prototypes()` in `src/utils/GradCAM.py`, in modo analogo a `compute_class_prototypes()` in `GraphGradCAM.py`, garantendo che non ci sia data leakage dalla gallery di validazione.

**Output:** overlay heatmap colorata sull'immagine originale — le zone rosse
corrispondono alle regioni visive più rilevanti per la costruzione dell'embedding.

#### Graph Grad-CAM — Importanza Nodi (GCN / GINE)

**File:** `src/utils/GraphGradCAM.py` — classe `GraphGradCAM`

Per i modelli a grafo, GradCAM viene adattato per operare sui **nodi** anziché
sui pixel. La differenza chiave rispetto alla baseline è lo **score**: invece
della norma dell'embedding si usa il **prodotto scalare con il prototipo della
classe target**, ovvero il centroide degli embedding di tutti i campioni della
stessa classe nel validation set:

```python
prototype = gallery_embs[classe_target].mean(dim=0)   # centroide della classe
score     = dot(emb, prototype)                        # quanto il grafo è vicino alla classe
score.backward()
```

Questa scelta è fondamentale per il metric learning: non esiste un logit di classe
su cui fare backprop, ma la distanza dal prototipo è il segnale più diretto per
capire cosa spinge un embedding verso la sua classe.

L'importanza per nodo è calcolata con la formula Grad-CAM standard adattata
alla dimensione dei nodi:

```python
alpha            = gradients.mean(dim=0)              # [Feature]
node_importance  = ReLU(sum(alpha * activations, dim=-1))  # [Num_Nodi]
node_importance /= node_importance.max()              # normalizzazione [0,1]
```

I prototipi di classe vengono calcolati tramite `compute_class_prototypes()`,
che fa la media degli embedding su tutti i campioni del validation set per classe.
L'analisi viene aggregata su più campioni per classe tramite
`aggregate_node_importance_by_class()`, producendo un ranking medio dei tipi di
nodo più discriminativi per ciascuna classe scenica.

**Output:** barplot affiancati GCN vs GINE per classe — permette di confrontare
quali oggetti i due modelli considerano più rilevanti (es. per *kitchen*:
*stove*, *refrigerator*, *cabinet*).

#### GNNExplainer — Importanza Archi (GCN / GINE)

**File:** `src/utils/GraphGradCAM.py` — funzioni `analyze_global_edge_importance_with_explainer`,
`analyze_global_edge_importance_gcn`

Approccio complementare a Grad-CAM: invece di usare i gradienti, **GNNExplainer
ottimizza una maschera continua sugli archi** `m_e ∈ [0,1]` per trovare il sottografo minimale
che preserva l'output originale. La maschera entra moltiplicando i messaggi sugli archi durante la forward pass mascherata:

```
h_v = AGG( { m_{u→v} · W · h_u  ∀u ∈ N(v) } )
```

minimizzando la loss:

```
L = − MI(Y, G_S) + λ · Ω(M)
```

dove il primo termine massimizza la fedeltà dell'output sul sottografo mascherato e il secondo penalizza maschere dense (sparsità).

A differenza di una classificazione standard, il task è metric learning e l'output è un vettore embedding — non un logit di classe. Per questo motivo lo score scalare che guida l'ottimizzazione della maschera è il **prodotto scalare tra embedding e prototipo della classe target**, identico a quello usato in Graph Grad-CAM:

```python
# GINEWithPrototype / GCNWithPrototype
score = dot(emb, prototype)   # [1, 1] — proiezione sul centroide della classe
```

I modelli sono wrappati in `GCNWithPrototype` e `GINEWithPrototype`, che registrano il prototipo come buffer non-learnable e restituiscono lo score scalare. L'explainer viene costruito tramite `build_explainer_with_prototype()`:

```python
build_explainer_with_prototype(model, prototype, model_type="gine", epochs=200)
```

Questa scelta garantisce che GNNExplainer ottimizzi la maschera rispetto all'**intera geometria dello spazio latente** — non a una singola coordinata dell'embedding come nel caso della modalità `regression` standard — producendo spiegazioni semanticamente coerenti con il Graph Grad-CAM e con la Supervised Contrastive Loss.

Se il prototipo non è disponibile, le funzioni ricadono automaticamente sul comportamento precedente (`GCNWrapper` / `GINEWrapper` senza prototipo) emettendo un `UserWarning`.

L'aggregazione delle importanze avviene in modo diverso per i due modelli:

- **GINE**: raggruppa per **tipo semantico della relazione** (`on`, `wearing`,
  `to the left of`, ecc.) usando `rel_vocab`.
- **GCN**: raggruppa per **coppia di tipi di nodo sorgente → destinazione**
  (`person → chair`, `table → plate`, ecc.) poiché non ha vocabolario semantico
  degli archi. Viene applicato un filtro `min_count` per escludere coppie viste
  meno di 3 volte, evitando che outlier rari falsino la classifica.

**Output:** barplot delle top-K relazioni più importanti nello spazio latente,
ordinato per importanza media su tutti i campioni analizzati.

---

### 6.3 Analisi Focal Point

**File:** `src/utils/GraphGradCAM.py` — funzioni `focal_point_analysis`,
`focal_point_analysis_gcn`, `robustness_by_relation_type`, `robustness_by_relation_type_gcn`

L'analisi focal point è lo strumento più originale del modulo di interpretabilità.
Mentre GNNExplainer misura l'importanza degli archi rispetto all'embedding,
la focal point analysis misura l'importanza rispetto al **retrieval** — ovvero
quanto cambia il ranking dei vicini più prossimi quando si rimuove un tipo di
relazione.

#### Ranking Disruption

Per ogni tipo di relazione nel vocabolario si calcola:

```
Disruption = 1 − |top-k(base) ∩ top-k(perturbato)| / k
```

dove `top-k(base)` sono i k vicini più prossimi con il grafo intatto e
`top-k(perturbato)` sono i k vicini dopo aver rimosso tutti gli archi di quel
tipo di relazione. La query viene esclusa dalla gallery prima del confronto per
evitare che la distanza nulla da se stessa distorca il ranking.

- `0.0` → la relazione è irrilevante per il retrieval
- `1.0` → la relazione ridetermina completamente il ranking (single point of failure)

Vengono incluse solo relazioni con almeno 5 campioni validi (grafi che contengono
quella relazione e che dopo la rimozione hanno ancora almeno 3 archi residui),
per evitare stime instabili su relazioni rarissime.

#### Delta Embedding

La funzione `robustness_by_relation_type()` fornisce una misura complementare:
invece del ranking, misura il **delta in norma L2 sull'embedding** dopo la
rimozione di ciascuna relazione. Un delta alto indica che quella relazione
contribuisce significativamente alla costruzione del vettore embedding.

Le due misure sono complementari e catturano fenomeni diversi:

| Scenario | Delta L2 | Disruption | Interpretazione |
| :--- | :---: | :---: | :--- |
| Alto | Alta | Relazione davvero critica in tutto |
| Alto | Bassa | Spostamento in direzione non discriminativa |
| Basso | Alta | Piccolo spostamento ma in zona di confine tra classi |
| Basso | Bassa | Relazione irrilevante |

#### Risultati

**GINE** — le due metriche sono fortemente allineate, con la stessa gerarchia di relazioni in entrambe le classifiche:

| Relazione | Delta L2 | Disruption |
| :--- | :---: | :---: |
| to the left of | 0.102 | 0.543 |
| to the right of | 0.091 | 0.486 |
| on | 0.051 | 0.246 |
| wearing | 0.020 | 0.129 |
| in | 0.023 | 0.121 |

L'allineamento tra le due metriche in GINE indica che lo spazio latente è regolare: ogni spostamento significativo dell'embedding si traduce in un cambio reale del ranking. Le relazioni critiche sono relazioni spaziali generiche (`to the left of`, `to the right of`) che appaiono in centinaia di grafi e strutturano la geometria complessiva della scena.

**GCN** — le due metriche sono invece **dissociate**:

| Top Delta L2 | Top Disruption |
| :--- | :--- |
| airplane → fan (0.850) | leaves → tree (0.517) |
| ottoman → chair (0.445) | person → shirt (0.400) |
| wall → tiles (0.436) | sky → clouds (0.367) |
| sailboat → lighthouse (0.399) | shirt → person (0.320) |
| lighthouse → sailboat (0.378) | clouds → sky (0.300) |

Le relazioni con alto Delta L2 sono specifiche e rare — producono grandi spostamenti dell'embedding in direzioni isolate dello spazio latente dove non ci sono altri grafi vicini, quindi il retrieval non cambia. Le relazioni con alta Disruption sono comuni e strutturalmente pervasive (`person → shirt`, `sky → clouds`) — spostano poco l'embedding in valore assoluto ma lo portano verso zone affollate dello spazio latente, cambiando i vicini recuperati.

Questo comportamento è coerente con il fatto che GCN non ha accesso ai tipi di arco espliciti: deve inferire il significato della relazione dalla coppia di tipi di nodo agli estremi. Le coppie rare identificano univocamente una scena ma la posizionano in una regione isolata; le coppie comuni organizzano i cluster condivisi tra molti grafi.

La tabella GCN mostra anche che le relazioni critiche per il Disruption compaiono in entrambe le direzioni (`leaves → tree` e `tree → leaves`, `sky → clouds` e `clouds → sky`), con importanze asimmetriche — confermando che GCN tratta il grafo come diretto e la direzione porta informazione indipendente.

**Output:** `pd.Series` / `pd.DataFrame` ordinati per metrica decrescente, con filtraggio per numero minimo di campioni validi.

---

## 7. Risultati

### 7.1 Confronto Baseline vs GCN vs GINE su Validation Set (grafo intatto, drop_prob = 0%)

Gli script `src/evaluation/baseline_eval.py`, `gcn_eval.py` e `gine_eval.py` sono stati eseguiti sul validation set (~3.208 campioni) per ottenere un confronto numerico diretto tra i tre modelli su tutte le metriche di retrieval implementate (risultati in `confronto_metriche_retrieval.csv`):

| Metrica | Baseline (CNN) | GCN | GINE |
| :--- | :---: | :---: | :---: |
| acc@1 | **62.31%** | 59.66% | 49.91% |
| precision_macro | **58.55%** | 56.73% | 48.77% |
| recall_macro | 56.68% | **56.61%** | 48.92% |
| recall@5 | **83.95%** | 82.04% | 78.83% |
| recall@10 | **88.81%** | 87.91% | 85.72% |
| precision@5 | **61.95%** | 60.32% | 50.02% |
| precision@10 | **61.52%** | 60.46% | 49.78% |
| MRR | **71.63%** | 69.49% | 62.10% |
| mAP | **68.00%** | 66.31% | 57.78% |

Contrariamente all'ipotesi di partenza, la **baseline CNN risulta il modello con le performance di retrieval più alte** su (quasi) tutte le metriche, seguita da GCN e poi da GINE. Il GCN resta comunque il migliore dei due encoder a grafo, confermando il pattern GCN > GINE già osservato in fase di analisi. Tra Baseline e GCN il margine è contenuto (circa 2-3 punti percentuali su acc@1 e mAP), mentre GINE rimane sensibilmente più indietro.

Questo risultato completa il confronto numerico baseline vs GNN che nella prima stesura della relazione era indicato come mancante (cfr. Sez. 8, Limiti), e viene discusso in dettaglio in Sez. 7.3 e Sez. 8.

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

Come riportato in Sez. 7.1, la baseline ResNet-18 ottiene in pratica le metriche di retrieval più alte tra i tre modelli (acc@1 = 62.31%, mAP = 68.00%). Questo risultato è in parte spiegabile dal modo in cui sono state generate le etichette di classe (Sez. 3.3): le 14 classi sceniche derivano da prompt CLIP basati sull'**aspetto visivo complessivo** della scena (es. *a beach scene*, *a kitchen interior*), un segnale che una CNN pre-addestrata su ImageNet è naturalmente molto efficace a catturare. I modelli a grafo, al contrario, ricevono solo informazione strutturale (oggetti e relazioni) e devono inferire la classe scenica da quella sola struttura, perdendo per costruzione l'informazione di colore, texture e illuminazione che ha contribuito a definire l'etichetta stessa.

Questo non invalida l'ipotesi di partenza sul valore dei scene graph, ma suggerisce che, con etichette di classe definite in base all'aspetto visivo, il confronto è strutturalmente favorevole alla baseline. Il vantaggio dei modelli a grafo è più plausibile su task dove la similarità "vera" è definita dalla composizione semantica (stessi oggetti/relazioni) indipendentemente dall'appearance — condizione che andrebbe verificata con etichette di classe costruite direttamente sulla struttura del grafo piuttosto che su CLIP visivo (cfr. Sez. 8).

### 7.4 Interpretabilità — Sintesi dei Risultati

L'analisi di interpretabilità ha prodotto tre tipi di evidenza complementari.

**Graph Grad-CAM (importanza nodi):** i prototipi di classe vengono calcolati sul training set tramite `compute_class_prototypes()` e usati come score scalare per il backward pass, in modo coerente con la Supervised Contrastive Loss. Lo stesso schema è adottato da GradCAM++ sulla baseline CNN tramite `compute_prototypes()` in `GradCAM.py`, garantendo confrontabilità tra i tre modelli.

**GNNExplainer (importanza archi):** l'ottimizzazione della maschera è guidata dal prodotto scalare `dot(emb, prototipo)` invece che da una singola coordinata dell'embedding. Questo è implementato nei wrapper `GCNWithPrototype` e `GINEWithPrototype` e garantisce che GNNExplainer operi sull'intera geometria dello spazio latente.

**Focal Point Analysis:** il confronto tra Delta L2 e Ranking Disruption ha rivelato un comportamento strutturalmente diverso tra i due modelli a grafo:

- **GINE**: le due metriche sono allineate (le stesse relazioni spaziali dominano entrambe le classifiche). Lo spazio latente è regolare — ogni spostamento significativo si traduce in un cambio reale del retrieval. Le relazioni critiche sono relazioni spaziali generiche (`to the left of`, `to the right of`) che strutturano la geometria complessiva della scena.

- **GCN**: le due metriche sono dissociate. Le relazioni con alto Delta L2 sono specifiche e rare (producono grandi spostamenti in direzioni isolate dello spazio latente, senza cambiare i vicini); le relazioni con alta Disruption sono comuni e part-whole (`person → shirt`, `leaves → tree`, `sky → clouds`), che spostano poco l'embedding ma lo portano verso zone affollate cambiando il ranking.

Questo risultato è interpretabile alla luce delle architetture: GCN, non avendo accesso ai tipi di arco espliciti, usa le coppie di tipi di nodo come proxy per il significato della relazione. Le coppie rare identificano univocamente una scena ma la posizionano in regioni isolate; le coppie comuni organizzano i cluster condivisi. GINE invece, avendo i tipi di arco come feature esplicite, costruisce uno spazio latente più regolare dove Delta L2 e Disruption sono naturalmente allineati.

---

## 8. Discussione

### Perché modellare i grafi?

Le scene non sono semplici collezioni di pixel: sono **composizioni di entità e relazioni**. Un GCN/GINE può riconoscere che due scene contengono {stove, refrigerator, cabinet} con relazioni spaziali simili, anche se illuminazione e colori differiscono. La baseline CNN deve inferire implicitamente tale struttura dai pixel, un compito più difficile con dati limitati.

### Baseline vs GCN vs GINE

Il confronto numerico completo (Sez. 7.1) mostra l'ordinamento **Baseline > GCN > GINE** su quasi tutte le metriche, con la baseline CNN davanti per 2-3 punti percentuali rispetto al GCN. Questo risultato va letto alla luce di come sono state costruite le etichette di classe (pseudo-label CLIP basate sull'aspetto visivo, Sez. 3.3): premia per costruzione un modello che, come la CNN, lavora direttamente sui pixel. I modelli a grafo restano comunque competitivi pur ignorando completamente l'informazione visiva, e tra di essi il GCN si conferma superiore al GINE.

### GCN vs GINE

| Aspetto | GCN | GINE |
| :--- | :--- | :--- |
| Feature archi | Solo topologia | Tipo semantico della relazione |
| Layer convoluzioni | 3 | 4 |
| Performance retrieval | Migliore tra i due encoder a grafo | Inferiore, più sensibile a perturbazioni |
| Interpretabilità relazioni | Per coppia tipi-nodo | Per tipo semantico (es. *on*, *wearing*) |

### Limiti attuali

1. **Subset ridotto**: solo il 30% di GQA, con classi sbilanciate.
2. **Etichette CLIP**: pseudo-label automatiche, non ground truth; possibili errori di classificazione. Inoltre, essendo basate sull'aspetto visivo della scena, favoriscono per costruzione un confronto con la baseline CNN (cfr. Sez. 7.3): un'etichettatura basata sulla struttura del grafo (oggetti/relazioni) renderebbe il confronto con GCN/GINE più equo.
3. **Scene graph ground truth**: si assume che GQA fornisca grafi perfetti; in produzione servirebbe un extractor automatico.
4. **`dataset_preprocessing.py` vuoto**: il preprocessing è documentato nel notebook `analisi_dataset.ipynb`.

---

## 9. Conclusioni

Il progetto dimostra la fattibilità di un sistema di **metric learning su scene graphs** per il retrieval semantico di scene. L'encoder **GCN** raggiunge quasi il **60% di acc@1** sul validation set, superando GINE e mostrando buona robustezza alle perturbazioni strutturali. Il confronto numerico completo con la baseline CNN (Sez. 7.1) mostra però che quest'ultima resta il modello con le metriche di retrieval più alte (acc@1 = 62.31%, mAP = 68.00%), un risultato in larga parte attribuibile al fatto che le etichette di classe sono derivate dall'aspetto visivo della scena (Sez. 3.3, 7.3). I modelli a grafo restano comunque competitivi pur non avendo accesso ai pixel, e tra i due il GCN è sistematicamente superiore al GINE.

Gli obiettivi extra sono stati affrontati con:
- Test di robustezza sistematico (perturbazione nodi).
- Visualizzazione dell'importanza di nodi e archi (Grad-CAM, GNNExplainer).
- Identificazione delle relazioni critiche (focal point analysis).

**Sviluppi futuri:**
- Addestrare su dataset completo GQA.
- Definire etichette di classe basate sulla struttura del grafo (anziché su CLIP visivo) per un confronto baseline vs GNN più equo.
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
- Revisione sintattica di questo documento.

Le scelte architetturali (GCN vs GINE, supervised contrastive loss, protocollo di valutazione retrieval, analisi focal point) e la responsabilità dei risultati sono del gruppo.

---