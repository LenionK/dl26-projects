# Istruzioni del Corso e Progetti

## 📚 Istruzioni Iniziali per gli Studenti

Questo repository è il punto di partenza ufficiale per tutti i progetti del corso. Di seguito i passi per iniziare:
1. **Scegliete il progetto**: Consultate [l'elenco dei progetti qui sotto](#elenco-progetti) per leggere le tracce disponibili e verificare quali sono libere o già assegnate. Comunicate quindi via mail il progetto scelto al docente.
2. **Fork**: Fate un **fork** di questa repository nel vostro account GitHub personale. Sebbene sia preferibile che usiate il tasto Fork in alto a destra (in modo da mantenere la cronologia visibile per valutarla), potete anche creare un repository slegato (o anche indipendentemente da GitHib) e tenerlo privato se preferite.
3. **Clone**: Clonate il vostro fork in locale.
4. **Lavorate in questa root**: Consultate [CONTRIBUTING.md](CONTRIBUTING.md) per le convenzioni richieste su come strutturare le cartelle (`src/`, `data/`, `notebooks/`), come scrivere codice pulito, e come usare Git professionalmente in gruppo. Sostituite i placeholder nel file `README.md` con le informazioni tecniche sul repository e `docs/REPORT.md` con la descrizione del progetto e del lavoro svolto.
5. **Policy sull'Uso dell'IA**: L'utilizzo di strumenti di AI generativa (ChatGPT, GitHub Copilot, Claude, ecc.) è **permesso, ma regolato**. L'utilizzo di questi strumenti è incoraggiato per velocizzare la scrittura di codice boilerplate, per il debugging o come supporto alla documentazione. Tuttavia, **non delegate mai all'IA il pensiero strategico e le scelte architetturali**. Elaborate la vostra strategia, scrivete o generate il codice, e assumetevi la piena responsabilità di ogni riga. L'uso di tali strumenti andrà esplicitamente dichiarato nella relazione finale.
6. **Licenza**: È buona norma rilasciare il vostro lavoro open source. Trovate un file `LICENSE` (preimpostato su licenza MIT). Aprite il file, sostituite `[Anno]` e `[Nome Cognome]` con l'anno corrente e i membri del vostro team. Ricordate di sceglierne una diversa se non volete condividere liberamente il codice.
7. **Consegna**: Il vostro fork GitHub è il **deliverable finale** del progetto. Assicuratevi che il codice sia riproducibile seguendo le istruzioni sotto e che le slide della presentazione d'esame siano depositate all'interno della cartella `docs/`. Nel caso in cui abbiate optato per un repository privato, la valutazione potrà avvenire rendendo il repo visibile al docente (handle `antoninofurnari`) o spedendo il sorgente del repo via email.

---

Questo file contiene l'elenco dei progetti disponibili, i dettagli completi per ogni progetto e i gruppi formati.

## Elenco Progetti

| ID | Titolo | Modulo | Difficoltà | Assegnato |
| :---: | :--- | :--- | :--- | :--- |
| 3 | [Metric Learning per Face Recognition Egocentrico](#traccia-3) | Metric Learning | Base | Libero |
| 4 | [Few-shot Learning per Gesture Recognition](#traccia-4) | Metric Learning | Intermedio | Libero |
| 5 | [Graph-based Metric Learning per Scene Understanding](#traccia-5) | Metric Learning | Avanzato | Libero |
| 6 | [Knowledge Distillation per Action Recognition Mobile](#traccia-6) | Knowledge Distillation | Base | Libero |
| 7 | [Domain Adaptation per Action Recognition – Egocentric → Exocentric](#traccia-7) | Domain Adaptation | Intermedio | Libero |
| 8 | [Domain Adaptation con Image-to-Image Translation (CycleGAN)](#traccia-8) | Domain Adaptation | Intermedio | Libero |
| 9 | [Multi-source Domain Adaptation per Riconoscimento Azioni](#traccia-9) | Domain Adaptation | Avanzato | Libero |
| 10 | [Contrastive Learning per Video Representation (SimCLR Video)](#traccia-10) | Self-Supervised Learning | Base | Libero |
| 11 | [Masked Video Modeling (MAE-style) per Egocentric Video](#traccia-11) | Self-Supervised Learning | Intermedio | Libero |
| 12 | [Clustering-based Self-Supervised Learning per Action Discovery](#traccia-12) | Self-Supervised Learning | Intermedio | Libero |
| 13 | [Temporal Action Localization con 1D CNN](#traccia-13) | Video Understanding | Base | Libero |
| 14 | [Action Recognition con Vision Transformer (ViT-based)](#traccia-14) | Video Understanding | Intermedio | Libero |
| 15 | [Vision-Language Alignment con CLIP per Video](#traccia-15) | Vision & Language | Intermedio | Libero |
| 16 | [Multimodal Action Recognition – Video + Audio + Text](#traccia-16) | Vision & Language | Avanzato | Libero |
| 17 | [Egocentric Video + Gaze per Procedural Understanding](#traccia-17) | Video Understanding | Intermedio | Libero |
| 18 | [State-Space Models (Mamba) per Sequenze Lunghe](#traccia-18) | Advanced Sequential Modeling | Avanzato | Libero |
| 19 | [Transformer vs RNN per Procedural Video Understanding](#traccia-19) | Advanced Sequential Modeling | Intermedio | Libero |
| 20 | [Diffusion Models per Trajectory/Motion Generation](#traccia-20) | Advanced Sequential Modeling | Intermedio | Libero |
| 21 | [Deep Q-Learning per Frame Selection in Video](#traccia-21) | Reinforcement Learning | Base | Libero |
| 22 | [Policy Gradient per Gesture Control](#traccia-22) | Reinforcement Learning | Intermedio | Libero |
| 23 | [Multi-agent RL per Coordinazione Compiti](#traccia-23) | Reinforcement Learning | Avanzato | Libero |
| 24 | [Task Graphs Differenziabili (Yao method) – Group A](#traccia-24) | Research Topic (Grafi/Procedural) | Avanzato | Libero |
| 25 | [Task Graphs – Softmax vs Sum Feasibility – Group B](#traccia-25) | Research Topic (Grafi/Procedural) | Avanzato | Libero |
| 26 | [Rilevamento Errori Procedurali con Gaze – Group A](#traccia-26) | Research Topic (Egocentric/Multimodal) | Intermedio | Libero |
| 27 | [Rilevamento Errori – Progress-Aware Model – Group B](#traccia-27) | Research Topic (Egocentric/Multimodal) | Intermedio | Libero |
| 28 | [Graph Autoencoder per Rappresentazioni Geometriche](#traccia-28) | Research Topic (Grafi/Representation) | Avanzato | Libero |
| 29 | [Hyperbolic Embeddings per Gerarchia Azioni](#traccia-29) | Research Topic (Rappresentazioni avanzate) | Intermedio | Libero |
| 30 | [Generative Models per Data Augmentation in Egocentric Domain](#traccia-30) | Research Topic (Egocentric/Generative) | Intermedio | Libero |
| 31 | [Online Episodic Memory per Action Anticipation](#traccia-31) | Research Topic (Memory/Anticipation) | Avanzato | Libero |

---

## Descrizioni Dettagliate dei Progetti

<a id='traccia-3'></a>
### Traccia 3: Metric Learning per Face Recognition Egocentrico
**Difficoltà**: Base  
**Modulo**: Metric Learning  
**Quando iniziare**: Dopo la lezione su Metric Learning (24/03/26)

#### Descrizione del problema
Riconoscere volti visti da prospettiva egocentric (es. in video da occhiali intelligenti). La sfida è che i volti sono spesso di parte, con occhi o profilo, e le condizioni di illuminazione sono estreme.

#### Dataset
- **EGTEA Gaze+** (subset con volti annotati) o sintetico
- ~100–200 identità, 5–10 esemplari per identità
- Frame a risoluzione bassa (egocentric ≈ 720p)

#### Obiettivi minimi
1. CNN backbone (ResNet-18 fine-tuned)
2. Triplet loss con hard negative mining: dato un volto anchor, trovare positivi (stessa persona) e negativi (altre persone)
3. Retrieval evaluation: mAP @1, 5, 10 (se mostro un volto, il modello recupera la stessa persona nei top-10 risultati più frequentemente?)
4. Analisi cluster nel latent space: volti della stessa persona sono vicini?

#### Obiettivi extra
- Confronto triplet loss vs ArcFace (margine-basato)
- Strategie di sampling per triplet (online mining vs offline)
- Robustezza a occhiali/maschere (aggiungere occlusione ai test)

---

<a id='traccia-4'></a>
### Traccia 4: Few-shot Learning per Gesture Recognition
**Difficoltà**: Intermedio  
**Modulo**: Metric Learning  
**Quando iniziare**: Dopo la lezione su Metric Learning (24/03/26)

#### Descrizione del problema
Riconoscere gesti (mani o corpo) dagli ultimi 1–5 esemplari visti. Questo è utile quando il gesto è raro o nuovo.

#### Dataset
- **miniImageNet** adattato a skeleton 2D (o video di gesti brevi)
- Oppure dataset DIY: 10 gesti, 5–10 esemplari per gesto
- Coordinate skeleton (e.g. da MediaPipe)

#### Obiettivi minimi
1. Feature encoder: CNN 1D su coordinatequadri scheletro (o CNN 2D su immagini gesti)
2. Prototypical Networks: calcolare prototipo per ogni classe (media degli embedding dei support set)
3. Query classification: confrontare query embedding con prototipi (distanza L2)
4. Metrica: Accuracy 5-way 1-shot, 5-way 5-shot
5. Report: come cambia performance con più esempi?

#### Obiettivi extra
- Relation Networks (imparare metrica di distanza invece di usare L2)
- Domain adaptation: pre-trainare su gesti di una persona, fine-tuning su un'altra
- Analisi failure cases

---

<a id='traccia-5'></a>
### Traccia 5: Graph-based Metric Learning per Scene Understanding
**Difficoltà**: Avanzato  
**Modulo**: Metric Learning  
**Quando iniziare**: Dopo la lezione su Metric Learning (24/03/26) + approfondimento

#### Descrizione del problema
Rappresentare scene (es. cucina, ufficio) come grafi (oggetti = nodi, relazioni spaziali = edge) e imparare embedding robusti per retrieval scene-to-scene.

#### Dataset
- **Visual Genome** (scene graphs: ~100k immagini con annotazioni oggetti e relazioni)
- Sottoinsiemme: ~500 scene con grafi non complessi (5–15 nodi)

#### Obiettivi minimi
1. Scene graph encoder: GCN/GraphSAGE che processa grafo e produce embedding
2. Contrastive loss: coppie di scene simili (stesso luogo, stessa attività) devono avere embedding vicini
3. Retrieval: dato un grafo query, trovare i scene graph più simili nel dataset
4. Metrica: mAP su retrieval, cluster purity

#### Obiettivi extra
- Grafo dinamico: estrarre scene graph dai video (nodi = tracce oggetti, edge = interazioni temporali)
- Robustezza a perturbazioni (rimuovere nodi/edge dal grafo di test, vedere se retrieval degradata)
- Interpretabilità: quali edge sono critiche per similarità?

---

<a id='traccia-6'></a>
### Traccia 6: Knowledge Distillation per Action Recognition Mobile
**Difficoltà**: Base  
**Modulo**: Knowledge Distillation  
**Quando iniziare**: Dopo la lezione su Knowledge Distillation (07/04/26)

#### Descrizione del problema
Comprimere un modello video pesante (es. 3D ResNet-50) in uno leggero (es. MobileNet) mantenendo performance, per deployment su dispositivi mobili.

#### Dataset
- **HMDB-51** o **UCF-101** (azioni sportive/quotidiane)
- ~1000–2000 video, 51 classi
- Feature video disponibili online

#### Obiettivi minimi
1. **Teacher**: 3D ResNet-50 pre-addestrato (baseline accuratezza sul test set)
2. **Student**: MobileNet 3D (versione leggera, es. 5–10x meno parametri)
3. **KD Loss**: L_KD = α * L_CE(student, hard_labels) + β * L_KL(student, teacher)
4. Training loop: student apprende dall'output soft del teacher
5. Metriche:
   - Accuracy confronto (teacher vs student no KD vs student + KD)
   - Model size (MB)
   - Inference time (ms)

#### Obiettivi extra
- Temperature tuning: come cambia performance con T = 1, 5, 10, 20?
- Attention transfer: non solo logit, ma anche mappe di attivazione intermedie
- Visualizzazione cosa il teacher trasmette al student (t-SNE del latent space)

---

<a id='traccia-7'></a>
### Traccia 7: Domain Adaptation per Action Recognition – Egocentric → Exocentric
**Difficoltà**: Intermedio  
**Modulo**: Domain Adaptation  
**Quando iniziare**: Dopo la lezione su Domain Adaptation (16/04/26)

#### Descrizione del problema
Un modello addestrato su video egocentric (da occhiali intelligenti) non funziona su video esocentric (video di terze persone). Usare DA per trasferire.

#### Dataset
- **Source (egocentric)**: EPIC-Kitchens (~1000 video)
- **Target (exocentric)**: Kinetics subset (~500 video simili, es. "chopping")
- Mapping: scegliere 20–30 azioni comuni

#### Obiettivi minimi
1. **Baseline fine-tuning**: addestrare su target, vedere accuratezza
2. **Adversarial DA**: gradient reversal layer
   - Encoder (CNN) condiviso
   - Classification head (predice azione su target)
   - Domain discriminator (predice se egocentric=0 o exocentric=1)
   - Backprop: loss_class - λ*loss_domain (adversarial)
3. Metriche: accuracy target, loss domain discriminator
4. Report: il modello riesce a confondere il discriminator? L'accuracy migliora con DA vs fine-tuning?

#### Obiettivi extra
- Maximum Mean Discrepancy (MMD) loss
- Visualizzazione feature alignment (t-SNE source vs target)
- Analisi per-classe: quali azioni sono facili/difficili da adattare?

---

<a id='traccia-8'></a>
### Traccia 8: Domain Adaptation con Image-to-Image Translation (CycleGAN)
**Difficoltà**: Intermedio  
**Modulo**: Domain Adaptation  
**Quando iniziare**: Dopo la lezione su Domain Adaptation (16/04/26)

#### Descrizione del problema
Tradurre immagini da dominio A a dominio B senza coppie allineate (es. sketch → foto). Usare la traduzione come pre-processing per migliorare classificatore su target.

#### Dataset
- **Office-31** (source: Amazon, target: DSLR)
- Oppure **VisDA** (syn → real)
- Alternativa: dati vostri con due domini naturali

#### Obiettivi minimi
1. **CycleGAN**: due generatori (A→B, B→A) e due discriminatori
2. Loss: adversarial (discriminator convince) + cycle-consistency (G_AB(G_BA(x)) ≈ x)
3. Pipeline: 
   - Addestrare CycleGAN per tradurre source → target
   - Usare immagini tradotte + originali per addestrare classifier
4. Metriche:
   - FID score (Frechet Inception Distance) su immagini tradotte
   - Accuracy classificatore su target
   - Qualità visiva (umana)

#### Obiettivi extra
- DANN contemporaneo: mentre CycleGAN traduce, un domain discriminator adavisuarsa
- Analisi ibrida: quando la traduzione aiuta e quando no?

---

<a id='traccia-9'></a>
### Traccia 9: Multi-source Domain Adaptation per Riconoscimento Azioni
**Difficoltà**: Avanzato  
**Modulo**: Domain Adaptation  
**Quando iniziare**: Dopo la lezione su Domain Adaptation (16/04/26)

#### Descrizione del problema
Invece di un solo dominio source, usare informazioni da 3 sorgenti diverse per migliorare su target.

#### Dataset
- **Source 1**: HMDB-51
- **Source 2**: UCF-101
- **Source 3**: Kinetics subset
- **Target**: Action-localization custom o subset diinterest

#### Obiettivi minimi
1. Modello con: shared encoder + 3 source classifiers + target classifier
2. Domain discriminator per ogni source (o globale)
3. Weighted ensemble: assegnare peso a ogni source basato su similarità con target
4. Training loop: ottimizzare tutti contemporaneamente
5. Metriche: accuracy target, per-source contribution analysis

#### Obiettivi extra
- Meta-learning per domain weighting: imparare quali source pesare
- Incomplete batch simulation: cosa succede se manca un source durante training?
- Analogy study: come performance varia con numero di source?

---

<a id='traccia-10'></a>
### Traccia 10: Contrastive Learning per Video Representation (SimCLR Video)
**Difficoltà**: Base  
**Modulo**: Self-Supervised Learning  
**Quando iniziare**: Dopo la lezione su Self-Supervised Learning (21/04/26)

#### Descrizione del problema
Pre-addestrare un encoder video senza etichette, usando contrastive loss su coppie di frame/clip augmentati dallo stesso video.

#### Dataset
- **Kinetics-400** (sottoinsiemme ~50k video) oppure **UCF-101** se volete partire piccolo
- Unlabeled (etichette NON usate per pre-training)

#### Obiettivi minimi
1. Augmentazioni video: crop spaziale, temporal sampling, color jitter, rotation
2. Encoder: 3D ResNet mini (es. ResNet-18 3D)
3. Projection head: encoder → 128-dim vector
4. Contrastive loss: SimCLR
   - Batch di N video
   - Per ogni video: due augmentazioni → due embedding
   - Loss: massimizzare similarità augmentazioni dello stesso video, minimizzare col resto del batch
5. Pre-training per K epoch (es. 100)
6. **Linear probe**: congelate encoder, allena solo FC layer su dataset etichettato (es. 10% HMDB), misura accuracy
7. Metrica: confrontare linear probe accuracy vs training supervised from scratch

#### Obiettivi extra
- Temperature in contrastive loss: T=0.1, 0.5, 1.0, effetto su convergen
- Visualizzazione: t-SNE embedding di video simili dovrebbero clusterizzare
- Momentum contrast (MoCo) per batch size più grande

---

<a id='traccia-11'></a>
### Traccia 11: Masked Video Modeling (MAE-style) per Egocentric Video
**Difficoltà**: Intermedio  
**Modulo**: Self-Supervised Learning  
**Quando iniziare**: Dopo la lezione su Self-Supervised Learning (21/04/26)

#### Descrizione del problema
Mascherare frame casuali in un video e addestrare modello a ricostruirli (autoencoder-style). Utile per egocentric perché il modello impara cosa succede nello spazio procedurale.

#### Dataset
- **EPIC-Kitchens** (1000 video ~30 frame ciascuno)
- Oppure dati procedurali vostri

#### Obiettivi minimi
1. **Input**: sequenza di frame [1, 2, 3, 4, 5] (es. 5 frame, 1 ogni 0.2 sec)
2. **Masking**: mascherare 50% dei frame (es. [X, 2, X, 4, X])
3. **Encoder**: ViT 3D o CNN 3D su frame non mascherati
4. **Decoder**: ricostruisce frame mascherati da embedding encoder
5. **Loss**: MSE tra frame ricostruiti e originali
6. **Evaluation**:
   - Reconstruction MSE/PSNR
   - Linear probe accuracy (downstream task: action classification)
   - Confronto con supervised pre-training

#### Obiettivi extra
- Strategie masking: random vs patch-based vs temporale
- Decoder asymmetrico (piccolo) per efficienza
- Visualizzazione frame ricostruiti

---

<a id='traccia-12'></a>
### Traccia 12: Clustering-based Self-Supervised Learning per Action Discovery
**Difficoltà**: Intermedio  
**Modulo**: Self-Supervised Learning  
**Quando iniziare**: Dopo la lezione su Self-Supervised Learning (21/04/26)

#### Descrizione del problema
Scoprire azioni ricorrenti in video **non etichettati** usando clustering iterativo. Utile quando non avete annotation ma il video ha pattern ripetitivi.

#### Dataset
- Unlabeled procedural videos (es. YouTube DIY subset, ~500 video)
- Oppure video shopping, cooking, ecc.

#### Obiettivi minimi
1. Feature extraction: backbone pre-trained (es. CLIP, TimeSformer)
2. Feature pooling per clip (es. clip di 30 frame → 1 vector 512-dim)
3. K-means clustering (iniziate con k=10, poi sperimentate)
4. Pseudo-labels: assegnate cluster a ogni clip
5. Fine-tuning: addestrate classifier su pseudo-labels
6. Valutazione: 
   - Cluster purity (quanti video in cluster 0 sono veramente simili?)
   - Downstream task accuracy se avete gt labels (opzionale)
7. Iterazione: ripetete clustering su embedding fine-tuned

#### Obiettivi extra
- Hierarchical clustering: scoprire gerarchia (macro-azioni vs micro-passi)
- Temporal consistency: clip dello stesso video dovrebbero restare nello stesso cluster

---

<a id='traccia-13'></a>
### Traccia 13: Temporal Action Localization con 1D CNN
**Difficoltà**: Base  
**Modulo**: Video Understanding  
**Quando iniziare**: Dopo la lezione su Video Understanding (30/04/26)

#### Descrizione del problema
Localizzare **quando** un'azione accade nel video (find start/end frame). Es. in un video di 2 minuti, trovare che l'azione "chopping" inizia al frame 120 e finisce al frame 350.

#### Dataset
- **ActivityNet-1.3** (sottoinsiemme 20 classi, ~100 video)
- Feature video pre-estratte (C3D/SlowFast)
- Annotazioni: frame start/end per ogni azione

#### Obiettivi minimi
1. Pre-processing: converte video → sequenza di feature (es. feature ogni 0.5 sec)
2. 1D CNN encoder: studia feature lungo dimensione temporale
3. Head regressione: predice (start_frame, end_frame) per ogni finestra
4. Loss: MSE + IoU loss
5. Metrica: mAP @IoU=0.5 (quante predizioni sono entro 50% del ground truth?)
6. Report: errori sistematici (es. predice sempre action troppo corta/lunga)?

#### Obiettivi extra
- Soft-NMS post-processing: fondere detection overlapping
- Class-agnostic detection: trovare confini azioni senza sapere classe

---

<a id='traccia-14'></a>
### Traccia 14: Action Recognition con Vision Transformer (ViT-based)
**Difficoltà**: Intermedio  
**Modulo**: Video Understanding  
**Quando iniziare**: Dopo la lezione su Video Understanding (30/04/26)

#### Descrizione del problema
Usare self-attention (Vision Transformer) per classificare azioni in video. Transformer vede il video come sequenza di patch spazio-temporali.

#### Dataset
- **HMDB-51** o **Kinetics-400 subset**
- Video frame o feature pre-estratte

#### Obiettivi minimi
1. **Patch embedding**: dividere video in patch 16x16x4 (spazio x tempo), project a 768-dim
2. **Positional encoding**: posizioni spazio-temporali
3. **Transformer**: stack di attention layers
4. **Classification**: CLS token → FC head → logit per 51 classi
5. **Training**: standard supervised CE loss
6. **Evaluation**: Top-1 accuracy, confronto con CNN 3D baseline
7. Metriche comparativi: latenza, # param

#### Obiettivi extra
- Attention visualization: quali patch il modello guarda?
- Confronto ViT vs TimeSformer (versione video-specific)
- Efficienza: ridurre # layer, # attention heads

---

<a id='traccia-15'></a>
### Traccia 15: Vision-Language Alignment con CLIP per Video
**Difficoltà**: Intermedio  
**Modulo**: Vision & Language  
**Quando iniziare**: Dopo la lezione su Vision & Language (07/05/26)

#### Descrizione del problema
Allineare feature video con testo usando contrastive loss (stile CLIP). Permette query testuale per video (es. "person chopping vegetables" → trovare video simili).

#### Dataset
- **MSR-VTT** (sottoinsiemme ~500 video con caption)
- Oppure **COCO-Captions** adattato con didascalie video

#### Obiettivi minimi
1. **Video encoder**: pre-trained (TimeSformer, SlowFast)
2. **Text encoder**: pre-trained (BERT, DistilBERT)
3. **Contrastive loss**: 
   - Batch di N (video, caption) coppie
   - Massimizzare similarità coppie corrette
   - Minimizzare similarità coppie scorrette
4. Training e valutazione:
   - Text-to-video retrieval: dato testo, trovare video simile
   - Metrica: R@1, R@5, R@10 (recall top-K)
5. Report: funziona la ricerca zero-shot?

#### Obiettivi extra
- Fine-tuning encoder (vs frozen)
- Zero-shot action recognition: assegnare label testuale agli action, top-5 accuracy
- Analisi failure: quali coppie il modello confonde?

---

<a id='traccia-16'></a>
### Traccia 16: Multimodal Action Recognition – Video + Audio + Text
**Difficoltà**: Avanzato  
**Modulo**: Vision & Language  
**Quando iniziare**: Dopo la lezione su Vision & Language (07/05/26)

#### Descrizione del problema
Classificare azioni sfruttando contemporaneamente video, audio e didascalie. Più modalità = più robustezza.

#### Dataset
- Sintetico (create vostre video con audio) oppure AudioSet + video
- Sottoinsiemme: 10 classi, 100 video ciascuna

#### Obiettivi minimi
1. **Encoder video**: 3D CNN
2. **Encoder audio**: 1D CNN su spettrogramma (librosa)
3. **Encoder text**: BERT / DistilBERT
4. **Fusion strategy**: concatenazione embedding + FC
5. Loss: standard CE
6. Valutazione: 
   - Accuracy multimodale (tutte e 3)
   - Accuracy singola modalità (per confronto)
   - Contribution analysis: quale modalità conta più?

#### Obiettivi extra
- Missing modality: robustezza quando video/audio/text manca
- Cross-modal attention
- Analisi late vs early fusion

---

<a id='traccia-17'></a>
### Traccia 17: Egocentric Video + Gaze per Procedural Understanding
**Difficoltà**: Intermedio  
**Modulo**: Video Understanding  
**Quando iniziare**: Dopo la lezione su Video Understanding (30/04/26)

#### Descrizione del problema
Combinare video egocentric (dal primo punto di vista) con tracce oculari (dove guarda la persona) per capire cosa sta facendo.

#### Dataset
- **EPIC-Kitchens + gaze** (sottoinsiemme con annotazioni sguardo)
- Oppure **AriaGen2** (smart glasses con IMU, gaze)
- Feature: video frame + gaze heatmap

#### Obiettivi minimi
1. **Video encoder**: 2D CNN su frame (ResNet-18)
2. **Gaze encoder**: CNN 2D su heatmap gaze (gaussian blob su punto sguardo)
3. **Fusion**: concatenazione embedding
4. **Classification**: FC head → azione
5. Evaluation: 
   - Accuracy azione
   - Ablation: video only vs gaze only vs fused
6. Report: gaze aiuta davvero? In quali azioni?

#### Obiettivi extra
- Saliency map: dove il modello "guarda" vs dove guarda la persona?
- Attention bottleneck: gaze è bottleneck per alcune azioni?
- Temporal alignment: sincronizzare video e gaze

---

<a id='traccia-18'></a>
### Traccia 18: State-Space Models (Mamba) per Sequenze Lunghe
**Difficoltà**: Avanzato  
**Modulo**: Advanced Sequential Modeling  
**Quando iniziare**: Dopo la lezione su Advanced Sequential Modeling (19/05/26)

#### Descrizione del problema
Usare State-Space Models (Mamba, Hippo) per modellare sequenze molto lunghe (es. interi video procedurali di 1+ minuto) senza quadratic cost di attention.

#### Dataset
- **EPIC-Kitchens** long sequences (~30 min sessioni, feature ogni 0.5 sec = 3600 step)
- Sottoinsiemme: 100 video, 50 azioni

#### Obiettivi minimi
1. **Baseline LSTM**: modello standard su sequenze lunghe
2. **Mamba/SSM**: implementazione (o usare libreria: mamba-ssm, ssm-lib)
3. Training: same loss (CE per azione), same preprocessing
4. Metriche:
   - Perplexity / Accuracy su azioni
   - Training time (wall-clock)
   - Memory usage
5. Benchmark: Mamba vs LSTM su lunghe sequenze

#### Obiettivi extra
- Hippo variant (iperbolico vs Mamba standard)
- Selective state updates (interpretability)
- Ablation: come lunghezza sequenza influenza Mamba vs LSTM?

---

<a id='traccia-19'></a>
### Traccia 19: Transformer vs RNN per Procedural Video Understanding
**Difficoltà**: Intermedio  
**Modulo**: Advanced Sequential Modeling  
**Quando iniziare**: Dopo la lezione su Advanced Sequential Modeling (19/05/26)

#### Descrizione del problema
Confrontare Transformer e RNN su task di comprensione step procedurale (es. Assembly, cooking): qual è più efficace su procedure?

#### Dataset
- **Assembly101** (sottoinsiemme 5 procedure, ~50 video)
- Annotazioni: step-wise labels, durata ogni step

#### Obiettivi minimi
1. **LSTM baseline**: encoder su feature video
2. **Transformer encoder**: multi-head attention, positional encoding
3. Training: classificazione step dato
 storia precedente
4. Metriche:
   - Frame-level accuracy (predice step corretto per ogni frame)
   - Per-class F1 score
   - Latency: inferenza Transformer vs LSTM

#### Obiettivi extra
- Hybrid: Transformer + recurrence layer
- Attention analysis: attention heads specializzati per temporale vs contentuale?

---

<a id='traccia-20'></a>
### Traccia 20: Diffusion Models per Trajectory/Motion Generation
**Difficoltà**: Intermedio  
**Modulo**: Advanced Sequential Modeling  
**Quando iniziare**: Dopo la lezione su Advanced Sequential Modeling (19/05/26)

#### Descrizione del problema
Generare traiettorie 2D/3D plausibili (persone che camminano) o sequenze di movimenti umani condizionate da contesto iniziale.

#### Dataset
- **Human3.6M** (skeleton motion capture data, sottoinsiemme walking/running)
- Oppure 2D synthetic trajectories (persone in topdown view)

#### Obiettivi minimi
1. **VAE baseline**: comprime traiettoria → latent, può generare
2. **Diffusion model** (DDPM): 
   - Forward: aggiungi rumore progressivo a traiettoria
   - Reverse: network apprende a rimuovere rumore
   - Sampling: iterativo per generare nuove traiettorie
3. Loss: MSE ricostruzione
4. Metriche:
   - Frechet distance (tra traiettorie generate e reali)
   - Diversity: generated variance vs real variance
5. Qualitativo: visualizzare alcune traiettorie generate

#### Obiettivi extra
- Condizionamento: genera traiettoria data velocità iniziale
- Classifier-free guidance
- Analisi steps denoising

---

<a id='traccia-21'></a>
### Traccia 21: Deep Q-Learning per Frame Selection in Video
**Difficoltà**: Base  
**Modulo**: Reinforcement Learning  
**Quando iniziare**: Dopo la lezione su Reinforcement Learning (26/05/26)

#### Descrizione del problema
Un agente apprende a selezionare frame informativi da un video per ridurre costo computazionale mantenendo buona accuratezza classificazione azione. Questo è utile per video compresso.

#### Dataset
- Video classification task (es. HMDB-51 subset, 100 video)
- Assegnate manualmente o automaticamente un score "informatività" a ogni frame (es. massima divergenza azione dall'ultimo frame selezionato)

#### Obiettivi minimi
1. **State**: frame corrente + storia frame selezionati
2. **Action space**: {select_frame, skip_frame}
3. **Reward**: +1 se azione classificata correttamente, -0.01 per frame selezionato (cost)
4. **Q-network**: CNN + FC che predice Q(state, action)
5. **DQN**: training loop con experience replay
6. Evaluation:
   - Accuracy azione vs # frame selezionati
   - Comparison: random selection vs learned selection

#### Obiettivi extra
- Double DQN
- Prioritized experience replay

---

<a id='traccia-22'></a>
### Traccia 22: Policy Gradient per Gesture Control
**Difficoltà**: Intermedio  
**Modulo**: Reinforcement Learning  
**Quando iniziare**: Dopo la lezione su Reinforcement Learning (26/05/26)

#### Descrizione del problema
Agente apprende a controllare ambiente (es. movimento avatar) interpretando gesti umani. L'agente riceve reward se ha interpretato correttamente il gesto.

#### Dataset
- Gesture dataset (es. MediaPipe skeleton di 5 gesti comuni) + Gymnasium environment (CartPole o GridWorld)

#### Obiettivi minimi
1. **Policy network**: gesture encoder (CNN skeleton) → action logits
2. **Algorithm**: REINFORCE (vanilla policy gradient)
   - Sample azioni da policy
   - Calcola return (cumulative reward)
   - Update policy: gradient ascent su log-prob
3. Evaluation:
   - Cumulative reward nel tempo
   - Convergence speed

#### Obiettivi extra
- Actor-Critic (riduce variance)
- Curriculum: inizio con compiti facili, poi difficili

---

<a id='traccia-23'></a>
### Traccia 23: Multi-agent RL per Coordinazione Compiti
**Difficoltà**: Avanzato  
**Modulo**: Reinforcement Learning  
**Quando iniziare**: Dopo la lezione su Reinforcement Learning (26/05/26)

#### Descrizione del problema
Due o tre agenti imparano coordinatamente a svolgere compiti procedurali paralleli (es. montaggio con due bracci robotici, preparazione ricetta con due persone).

#### Dataset
- Sintetico: environment MultiAgentEnv basato su Gymnasium
- Definite compiti semplici (es. 5 step ciascuno, agenti devono coordinare per non conflitto)

#### Obiettivi minimi
1. **Multi-agent environment**: 2–3 agenti, shared reward (coordinazione) o individual reward (compito)
2. **Policy**: policy separata per agente
3. **Communication**: semplice (es. sharing state) o silent (learn implicitly)
4. Training: train tutte le policy contemporaneamente
5. Evaluation:
   - Task completion rate (quante volte entrambi agenti finiscono task?)
   - Coordination efficiency

#### Obiettivi extra
- Emergent behavior: descrivete comportamenti emergenti durante training
- Decentralized learning: agenti non hanno accesso a joint state

---

<a id='traccia-24'></a>
### Traccia 24: Task Graphs Differenziabili (Yao method) – Group A
**Difficoltà**: Avanzato  
**Modulo**: Research Topic (Grafi/Procedural)  
**Quando iniziare**: Metà corso, dopo Domain Adaptation

#### Descrizione del problema
Riprodurre il metodo di Yao et al. (2022) "Differentiable Task Graphs for Procedure Understanding". Idea: ogni procedura è un grafo diretto acyclico (DAG) di step; il modello apprende la feasibilità (quale step può seguire quale) e predice sequenza step.

#### Dataset
- **Assembly101** (sottoinsiemme 5 procedure complete, ~50 video)
- Ogni video ha annotazione step-wise (frame→step)

#### Obiettivi minimi
1. **Graph construction**: estrai grafo task dai ground truth annotations (nodi = step unici, edge = transizioni osservate)
2. **Task graph encoder**: GNN (GCN) che mappa grafo task → embedding feasibilità
3. **Prediction**: dato video + grafo, predici prossimo step
4. **Metrica**: mAP su sequenza step predetta vs ground truth
5. **Baseline**: simple classifier (ignora grafo) per comparazione
6. **Ablation**: rimuovi componenti del grafo (edge, nodi), impact su performance

#### Obiettivi extra
- Analisi errori: quando la feasibilità predetta è sbagliata?
- Visualizzazione grafo predetto vs ground truth

#### Riferimenti
- Paper: Yao et al. "Differentiable Task Graphs for Procedure Understanding" (ICCV 2023)
- GitHub: ufficiale se disponibile

---

<a id='traccia-25'></a>
### Traccia 25: Task Graphs – Softmax vs Sum Feasibility – Group B
**Difficoltà**: Avanzato  
**Modulo**: Research Topic (Grafi/Procedural)  
**Quando iniziare**: Metà corso, dopo Domain Adaptation

#### Descrizione del problema
**Estensione della Traccia 24**: Yao usa somma nella feasibilità aggregation; sperimentate softmax e confrontate.

#### Dataset
- Same as Traccia 24: Assembly101

#### Obiettivi minimi
1. **Baseline Yao**: riprodurre metodo standard (sum aggregation)
2. **Softmax aggregation**: sostituire sum con softmax nella feasibility prediction
3. **Constrained optimization**: aggiungete vincoli (es. ogni step deve avere ≤K predecessori)
4. **Comparison**: mAP, analisi convergenza, interpretabilità
5. **Ablation**: quali modifche aiutano/peggiorano?

#### Obiettivi extra
- Softer constraints (probabilistici vs hard)

---

<a id='traccia-26'></a>
### Traccia 26: Rilevamento Errori Procedurali con Gaze – Group A
**Difficoltà**: Intermedio  
**Modulo**: Research Topic (Egocentric/Multimodal)  
**Quando iniziare**: Dopo Video Understanding (fine aprile)

#### Descrizione del problema
Un operatore segue una procedura (cucina, montaggio) ripreso da egocentric. Il modello rileva se sta facendo un errore combinando video + gaze (dove guarda).

#### Dataset
- **EPIC-Kitchens subset** (~300 video) con manuale annotation "correct/mistake" per step
- Oppure create vostre mini procedure etichettate corretto/errore

#### Obiettivi minimi
1. **Video feature**: TSN o pre-estratte
2. **Gaze feature**: heatmap sguardo (gaussiana su punto tracked)
3. **Fusion**: concatenazione embedding
4. **Binary classifier**: correct vs mistake
5. **Evaluation**: F1-score, confusion matrix, per-class analysis
6. **Ablation**: video only vs gaze only vs fused

#### Obiettivi extra
- Saliency map: dove il modello guarda quando predice errore?
- Temporal localization: non solo predici errore, ma dove nello step?

---

<a id='traccia-27'></a>
### Traccia 27: Rilevamento Errori – Progress-Aware Model – Group B
**Difficoltà**: Intermedio  
**Modulo**: Research Topic (Egocentric/Multimodal)  
**Quando iniziare**: Dopo Video Understanding (fine aprile)

#### Descrizione del problema
**Estensione della Traccia 26**: Aggiungere un task ausiliario di predizione del "progresso" (quale step della procedura siamo?) per migliorare detesione di errori.

#### Dataset
- Same as Traccia 26

#### Obiettivi minimi
1. **Baseline**: mistake detection standard
2. **Progress predictor**: side task che predice qual è lo step corrente (multitask setup)
3. **Joint loss**: L = L_mistake + λ * L_progress
4. **Comparison**: F1-score, ablation su λ
5. **Analysis**: progress help mistake detection?

#### Obiettivi extra
- Adversarial loss: force progress embedding to be independent from mistake prediction

---

<a id='traccia-28'></a>
### Traccia 28: Graph Autoencoder per Rappresentazioni Geometriche
**Difficoltà**: Avanzato  
**Modulo**: Research Topic (Grafi/Representation)  
**Quando iniziare**: Dopo Metric Learning

#### Descrizione del problema
Autoencoder su grafi per apprendere embedding geometrici robusti. Applicabile a scene graph retrieval, graph clustering, ecc.

#### Dataset
- **Visual Genome** (scene graphs)
- Sottoinsiemme: ~500 grafi con 5–15 nodi

#### Obiettivi minimi
1. **Graph encoder**: GCN/GraphSAGE che mappa grafo → embedding
2. **Autoencoder decoder**: MLP che decodifica embedding → predizioni nodi e edge
3. **Loss**: binary cross-entropy per edge reconstruction + node classification
4. **Evaluation**:
   - Link prediction accuracy
   - Clustering nel latent (scene simili clusterizzano?)
   - Grafo "medio" via interpolazione latente
5. **Ablation**: size encoder, decoder depth, etc.

#### Obiettivi extra
- VAE variant (latent distribuzione gaussiana)
- Hyperbolic latent space (prossima traccia)

---

<a id='traccia-29'></a>
### Traccia 29: Hyperbolic Embeddings per Gerarchia Azioni
**Difficoltà**: Intermedio  
**Modulo**: Research Topic (Rappresentazioni avanzate)  
**Quando iniziare**: Dopo Metric Learning

#### Descrizione del problema
Azioni spesso hanno struttura gerarchica (macro-azioni contengono micro-azioni). Spazio iperbolico preserva meglio gerarchie che spazio euclideo.

#### Dataset
- **ActivityNet**: usate hierarchies fornite (event categories)
- Sottoinsiemme: ~20 classi, albero con 3 livelli

#### Obiettivi minimi
1. **Euclidean baseline**: embedding standard su spazio euclideo
2. **Poincare embeddings**: hyperbolic (libreria: geoopt)
3. **Task**: link prediction (data gerarchia parziale, predici link mancanti)
4. **Metric**: 
   - Distortion (quanto bene preserva gerarchia vs euclidean?)
   - Link prediction accuracy
5. **Visualization**: iperboloide proiettato in Poincare disk

#### Obiettivi extra
- Tuning curvature iperbolica (K negativa)
- Mixed curvature spaces

---

<a id='traccia-30'></a>
### Traccia 30: Generative Models per Data Augmentation in Egocentric Domain
**Difficoltà**: Intermedio  
**Modulo**: Research Topic (Egocentric/Generative)  
**Quando iniziare**: Dopo Self-Supervised Learning

#### Descrizione del problema
EPIC-Kitchens è il più grande dataset egocentric, ma a volte specifiche azioni sono rare. Generare frame sintetici realistici per augmentare dataset.

#### Dataset
- **EPIC-Kitchens** (1000 video, 50 azioni)
- Focus su azioni rare (es. "cutting" ha 100 esemplari, "greeting" ha 10)

#### Obiettivi minimi
1. **VAE baseline**: comprime frame → latent, genera nuovi
2. **Diffusion model**: noisy→clean denoising per generare frame
3. **Evaluation**:
   - FID score (qualità frame generati)
   - Downstream action recognition: training su original+synthetic vs original only
4. **Ablation**: quale generative model funziona meglio per egocentric?

#### Obiettivi extra
- Condizionamento: genera frame dato'azione label
- Feature diversity: generated frame aggiunge nuove variazioni visuali?

---

<a id='traccia-31'></a>
### Traccia 31: Online Episodic Memory per Action Anticipation
**Difficoltà**: Avanzato  
**Modulo**: Research Topic (Memory/Anticipation)  
**Quando iniziare**: Dopo Advanced Sequential Modeling

#### Descrizione del problema
Umani anticipano azioni future ricordando situazioni passate simili. Modulo di memoria episodica aggiornato online può migliorare anticipazione.

#### Dataset
- **EPIC-Kitchens** procedural sequences (~500 video)
- Task: dato video chunk, predici azioni future 1–5 secondi

#### Obiettivi minimi
1. **LSTM baseline**: anticipazione senza memoria
2. **Memory bank**: VQ-inspired quantization di query/values
   - Query: feature video corrente
   - Store: feature passate + azioni future osservate
3. **Online update**: aggiorna memory durante rolling window (non static)
4. **Evaluation**:
   - Top-5 accuracy anticipazione a diversi orizzonti
   - Memory efficiency (size, retrieval time)
5. **Ablation**: impatto size memory, decay temporale

#### Obiettivi extra
- Query learning (network che apprende quale memoria recuperare)
- Temporal decay (vecchie memories meno rilevanti)

---


## Gruppi

| Nome Gruppo | Numerosità |
| :--- | :---: |
| LeMeCla | 3 |
| BAT 🦇 (Backpropagation Attention Team) | 3 |
| Deep Team | 3 |
| Justgood AI | 3 |
| FiCo | 3 |
| FlyNow | 3 |
| Overfittony | 3 |
| The Outliers 2.0 | 3 |
| Zero e Uno | 2 |
| DataMinds | 2 |
| TEAM CassiaBranca | 2 |
| Le larunghie | 2 |
| DataLost | 2 |
| EventHorizonTeam | 2 |
| Marte | 2 |
| G16 | 1 |
| G17 | 1 |
| G18 | 1 |
| G19 | 1 |
| G20 | 1 |
| G21 | 1 |
| G22 | 1 |
| G23 | 1 |
| G24 | 1 |
| G25 | 1 |
| G26 | 1 |
| G27 | 1 |
| G28 | 1 |
| G29 | 1 |
| G30 | 1 |
| G31 | 1 |
