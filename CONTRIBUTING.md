# Struttura e Best Practices della Repository

Questo documento serve da guida per mantenere il vostro progetto in ordine. Una repository ben strutturata non solo facilita il lavoro in team, ma è anche il biglietto da visita del vostro progetto in ambito accademico e lavorativo.

---

## 📁 1. Organizzazione delle Directory

Nel fork che avete clonato, trovate già una struttura preimpostata. Ecco come utilizzarla correttamente:

* **`README.md`**: File per la documentazione tecnica del repository. Sostituite i placeholder nel file `README.md` con le informazioni tecniche sul repository.
* **`data/`**: Conservate qui i dataset. Ricordate che **questi file NON devono mai essere caricati su GitHub**. Il file `.gitignore` è già configurato per escludere file contenuti in questa cartella, ma prestate sempre attenzione.
* **`src/`**: Il cuore del vostro codice. Oltre alle cartelle già incluse, potete aggiungere quelle che ritenete necessarie per il vostro progetto.
  * `datasets/`: Script per lo scaricamento, il parsing e i dataloader in PyTorch.
  * `models/`: Architetture delle reti neurali (es. definizioni delle classi `nn.Module`).
  * `training/`: Training loop, custom loss functions, optimizers.
  * `evaluation/`: Codice per test, inferenza e calcolo metriche.
  * `utils/`: Funzioni ausiliarie, logger, visualizzazioni.
* **`notebooks/`**: Notebook Jupyter. Da utilizzare *soltanto* per l'esplorazione iniziale dei dati, visualizzazioni rapide e proof of concept. Il codice di training vero e proprio deve risiedere nei file Python dentro `src/`.
* **`experiments/`**:
  * `configs/`: File `.yaml` o `.json` per gestire i parametri degli esperimenti (hyperparameters, path).
  * `logs/`: Output di TensorBoard, Weights & Biases o semplici log file. *(Non committabili)*.
* **`figures/`**: Grafici, test plot e immagini inserite nel README o nella presentazione.
* **`docs/`**: File per la documentazione dettagliata, slide della presentazione finale, e report finale (`docs/REPORT.md`).

---

## 💻 2. Standard del Codice (Clean Code)

Avere codice leggibile è essenziale, specie nei progetti di Machine/Deep Learning dove implementazioni confusionarie mascherano facilmente bug logici.

* **Nomi Sensati**: Usate nomi chiari e parlanti (es. `compute_cross_entropy` invece di `calc_ce`).
* **Modularità**: Evitate file "mostro" da 2000 righe. Ogni file dovrebbe occuparsi di una cosa sola. Ad esempio, il loop di addestramento non deve contenere anche l'architettura della rete neurale o le trasformazioni sui tensori.
* **Docstrings e Commenti**: 
  * Documentate le classi e le funzioni principali con Docstrings brevi ma esaustivi.
  * Limitate i commenti "in-linea" per lo più ai blocchi controintuitivi / complessi (es. operazioni complicate tra tensori dimensionali che modificano lo shape).
* **Type Hinting**: L'inserimento dei classici `int`, `str`, `List`, ecc. è raccomandato, rende l'intento del codice immediatamente chiaro a chi lo revisiona.

---

## 📦 3. Gestione delle Dipendenze

La riproducibilità è un caposaldo fondamentale della progettazione di sistemi di Deep Learning. Un'altra persona deve poter ricreare il vostro ambiente in un attimo.
* Mantenete il file `environment.yml` **sempre aggiornato**.
* Se scaricate un nuovo pacchetto vitale via pip o conda, abbiate l'accortezza di annotarlo nel file (con i numeri di versione quando opportuno).
* Evitate di includere librerie OS-specific o non strettamente richieste per il progetto nel file environment.

---

## 🌳 4. Flusso di Lavoro ed Uso di Git

Per i gruppi formati da più di una persona, vi sconsigliamo di committare i vostri progressi unicamente sul branch `main`.
* **Uso dei Branch**: Lavorate per funzionalità. Volete testare una ResNet50? Create un branch `feature/resnet50`. Quando l'implementazione è solida, eseguite in Github una "Pull Request" e unitelo (merge) a `main`.
* **Messaggi di Commit**: Evitate commit stile "aggiornamento" o "fix bug". Cercate di dare informazione: `Add custom triplet loss in training loop` o `Fix dataloader out of bounds exception`.

---

Seguendo queste linee guida, svilupperete l'abitudine verso gli standard comunemente impiegati nell'industria e nella ricerca software!
