# Random Search — Best Configuration & Confronto con Autoresearch

> Generato: 2026-05-05 | Progetto: NanoDrone 3Fold

---

## 1. Best Configuration (Random Search)

**Vincitore: config_id = 26 — `PhysicsResidualLSTM`**

```
val_MAE  : 0.3384   (media 3 fold)
test_MAE : 0.0938   (melon metric)
  chirp  : 0.1521
  random : 0.2532
  square : 0.6100
```

### 1.1 Iperparametri

| Parametro        | Valore       |
|-----------------|-------------|
| `model_class`   | `PhysicsResidualLSTM` |
| `n_hidden`      | 128          |
| `num_layers`    | 3            |
| `hidden_sizes`  | [256, 128]   |
| `lr`            | 1.69 × 10⁻³ |
| `dropout`       | 0.25         |
| `weight_decay`  | 1 × 10⁻⁵    |

### 1.2 Architettura — PhysicsResidualLSTM

Il modello è un ibrido **fisica + LSTM residuale** in due blocchi:

```
Input per ogni step t:
  u_t     ∈ ℝ⁴   (velocità dei 4 motori, normalizzate)
  y_{t-1} ∈ ℝ¹²  (stato precedente: pos, vel, euler, omega)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCCO 1 — Physics step (Euler integration)
  p_t     = p_{t-1}     + gain_p ⊙ v_{t-1}     (posizione)
  euler_t = euler_{t-1} + gain_e ⊙ omega_{t-1}  (angoli di Eulero)
  v_t     = v_{t-1}                               (zero-order hold)
  omega_t = omega_{t-1}                           (zero-order hold)
  → y_phys_t ∈ ℝ¹²
  gain_p ∈ ℝ³, gain_e ∈ ℝ³  sono parametri learnable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCCO 2 — LSTM residuale
  Input LSTM: [u_t, y_{t-1}] ∈ ℝ¹⁶
  LSTM: hidden=128, layers=3, batch_first=True
  Output: delta_t ∈ ℝ¹²  (correzione residuale full-state)
  
  Inizializzazione stato nascosto LSTM:
    FF_init: Linear(12→256) → ReLU → Linear(256→128) → ReLU → Linear(128→128)
    → h₀ ∈ ℝ¹²⁸, ripetuto su 3 layer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output finale:
  y_t = y_phys_t + delta_t
```

**Intuizione:** la componente fisica cattura la cinematica di primo ordine (integrazione di Eulero) con guadagni learnable; la LSTM corregge tutto ciò che la fisica non modella (forze aerodinamiche, nonlinearità dei motori, accoppiamenti roll/pitch/yaw).

### 1.3 Contesto del Random Search

Il random search ha esplorato **30 configurazioni** in 2 classi di modelli:

| Classe             | Descrizione                              | Configs testati |
|--------------------|------------------------------------------|----------------|
| `AutoregressiveLSTM` | LSTM pura, black-box                   | 19             |
| `PhysicsResidualLSTM`| Fisica + LSTM residuale (questo modello) | 11             |

Risultato netto: **tutti i migliori 10 risultati per val_MAE appartengono a PhysicsResidualLSTM**. Il vantaggio del prior fisico è sistematico su questo dataset.

Classifica top-5 (val_MAE):

| Rank | Config | val_MAE | Modello | n_hidden | layers |
|------|--------|---------|---------|----------|--------|
| 1 | **26** | **0.3384** | PhysicsResidualLSTM | 128 | 3 |
| 2 | 27 | 0.3511 | PhysicsResidualLSTM | 192 | 3 |
| 3 | 8 | 0.3512 | PhysicsResidualLSTM | 320 | 3 |
| 4 | 10 | 0.3592 | PhysicsResidualLSTM | 128 | 2 |
| 5 | 14 | 0.3559 | PhysicsResidualLSTM | 128 | 2 |

---

## 2. Confronto: Random Search vs Autoresearch

### 2.1 Risultati numerici

| Metrica           | Random Search (best) | Autoresearch (iter28) | Δ      |
|-------------------|---------------------|-----------------------|--------|
| val_MAE (media)   | 0.3384              | **0.2858**            | −15.5% |
| val_MAE chirp     | 0.1521              | **0.125**             | −17.8% |
| val_MAE random    | 0.2532              | **0.162**             | −36.0% |
| val_MAE square    | 0.6100              | **0.571**             | −6.4%  |
| test_MAE (melon)  | 0.0938              | **0.0816**            | −13.0% |

**L'autoresearch batte la random search del ~15% su val_MAE e ~13% su test_MAE.**

Il guadagno è distribuito su tutti i fold ma è massimo su **random** (−36%): il fold con comandi irregolari beneficia enormemente del scheduled sampling.

### 2.2 Architettura a confronto

|                        | Random Search best   | Autoresearch best           |
|------------------------|---------------------|-----------------------------|
| Classe modello         | `PhysicsResidualLSTM` | `KinematicsLSTMModel`      |
| Input LSTM             | [u_t, y_{t-1}] 16D  | [u_t, u_t², y_{t-1}] **20D** |
| Feature u²             | No                  | **Sì** (thrust ∝ ω²)        |
| Output LSTM            | delta_t **12D** (full state) | [Δv, Δω] **6D** only |
| Integrazione cinematica| posizione + angoli  | posizione + angoli + **velocità** |
| n_hidden               | 128                 | **256**                      |
| num_layers LSTM        | 3                   | **5**                        |
| Init hidden            | FF(12→256→128→128)  | FF(12→128→64→256)            |
| Parametri totali (stima)| ~430K              | ~**1.8M**                    |

**Differenze chiave di architettura:**

1. **Output LSTM più ristretto (6D vs 12D):** in KinematicsLSTM la LSTM predice solo le accelerazioni (Δv, Δω); posizione ed euler vengono aggiornate via integrazione cinematica strutturale. In PhysicsResidualLSTM la LSTM corregge direttamente tutto lo stato (12D). Il design di KinematicsLSTM è più strutturato fisicamente.

2. **Feature engineering u²:** KinematicsLSTM aggiunge esplicitamente il quadrato dei comandi motori. La forza di spinta è proporzionale a ω² — questa feature evita alla LSTM di dover imparare la relazione quadratica.

3. **Più capacità:** 256 hidden, 5 layer vs 128 hidden, 3 layer (circa 4× più parametri).

### 2.3 Training a confronto

|                        | Random Search        | Autoresearch                |
|------------------------|---------------------|-----------------------------|
| Teacher forcing        | No (inferenza diretta) | **Sì** (scheduled sampling: 0.3→0 su 500 epoch) |
| Multi-horizon loss     | No                  | **Sì** (loss_50 + 0.5·loss_25 + 0.25·loss_10) |
| Input noise aug.       | No                  | **Sì** (std=0.05 sui comandi) |
| Initial state noise    | No                  | **Sì** (std=0.02 su y₀) |
| Eval granularity       | Non specificata     | **eval_every=5** (scoperta critica) |
| batch_size             | Non specificato     | 128 (ottimizzato) |
| Optimizer              | Non specificato     | AdamW + gradient clip 1.0 |

**Le tecniche di training sono il gap principale.** La random search ha solo ottimizzato architettura e iperparametri classici (hidden size, layers, lr, dropout, weight_decay), senza toccare le tecniche avanzate di training che nell'autoresearch hanno portato il contributo maggiore:

- **Scheduled sampling:** +9% da solo (il contributo singolo più grande in assoluto)
- **eval_every=5:** +2.5% "gratuito" — il random search potrebbe avere best epoch sbagliati
- **Multi-horizon loss + noise aug:** base consolidata dell'autoresearch

### 2.4 Cosa succederebbe applicando il training dell'autoresearch alla best config RS?

La best config RS usa `PhysicsResidualLSTM` con un'architettura simile a `KinematicsLSTMModel`. Se si ri-addestrasse config_id=26 con:
- scheduled sampling (teacher_ratio_start=0.2, decay→0)
- multi-horizon loss
- eval_every=5
- batch=128 + AdamW + grad clip

il guadagno atteso sarebbe almeno +9% (scheduled sampling) portando il val_MAE da 0.338 → ~0.307, avvicinandosi ma probabilmente non raggiungendo il 0.286 dell'autoresearch (che usa anche architettura più profonda e features u²).

### 2.5 Sintesi: cosa ha vinto e perché

| Contributo           | Fonte           | Impatto stimato |
|----------------------|-----------------|-----------------|
| Architettura fisica  | entrambi (RS lo ha trovato casualmente, AR lo ha progettato) | base fondamentale |
| Scheduled sampling   | Autoresearch only | **+9%** — il più importante |
| Architettura più profonda (5L, 256H) | Autoresearch only | +5-6% |
| Feature u²           | Autoresearch only | +2-3% |
| eval_every=5         | Autoresearch only | +2.5% |
| batch_size tuning    | Autoresearch only | +1% |

**La random search ha trovato la classe di architettura giusta** (physics-residual è superiore alla LSTM pura) ma ha mancato tutte le tecniche di training che nell'autoresearch hanno portato il 60% del miglioramento totale.

---

## 3. Come Confrontare i Risultati in Modo Rigoroso

### 3.1 Metriche disponibili

| File                                      | Contenuto                                    |
|-------------------------------------------|----------------------------------------------|
| `random_search/results.tsv`               | val_MAE per 30 config (3 fold separati)      |
| `random_search/best_config.json`          | parametri della config vincente              |
| `final_analysis/best_test_metrics.json`   | test MAE/R² per output autoresearch (stride default) |
| `final_analysis/stride1_test_metrics.json`| test MAE/R² per output autoresearch (stride=1, più finestra) |
| `checkpoints/best_so_far/`                | checkpoint ensemble autoresearch             |

### 3.2 Metriche comparabili direttamente

```
Random Search config_26:
  val_MAE (chirp fold)  : 0.152
  val_MAE (random fold) : 0.253
  val_MAE (square fold) : 0.610
  test_MAE (melon)      : 0.094

Autoresearch iter28:
  val_MAE (chirp fold)  : 0.125
  val_MAE (random fold) : 0.162
  val_MAE (square fold) : 0.571
  test ensemble MAE     : 0.082  (best_test_metrics.json, overall_mae)
  test ensemble MAE*    : 0.083  (stride1_test_metrics.json)
```

*La metrica `test_MAE_melon` nella random search usa la stessa finestra temporale dell'ensemble `best_test_metrics.json` — confronto diretto valido.*

### 3.3 Confronto per singolo output (autoresearch disponibile, RS no)

Il file `final_analysis/best_test_metrics.json` espone MAE e R² per ogni singola delle 12 variabili di stato. La random search salva solo la MAE aggregata. Se si volesse un confronto granulare, bisognerebbe ri-valutare la best config RS sui checkpoint con lo stesso script di `final_analysis.py`.

### 3.4 Script per un confronto diretto

Per valutare la best config RS con le stesse metriche dell'autoresearch:

```python
# Carica la best config RS
import json
with open("random_search/best_config.json") as f:
    rs_config = json.load(f)

# Carica i checkpoint RS (in random_search/checkpoints/)
# Esegui final_analysis.py puntando ai checkpoint RS invece di checkpoints/best_so_far/
```

Il file `random_search/checkpoints/` dovrebbe contenere i modelli addestrati per ciascun fold della config 26 — se presenti, si può eseguire `final_analysis/final_analysis.py` modificando i percorsi dei checkpoint per ottenere la stessa scomposizione per output.

---

## 4. Conclusioni

1. **La random search ha identificato correttamente che `PhysicsResidualLSTM` è superiore all'LSTM pura** — un risultato riproducibile e consistente su tutti i 30 trial.

2. **L'autoresearch supera la random search del ~15%** non tanto per l'architettura (entrambi usano un modello ibrido fisica+LSTM) ma per le **tecniche di training**: scheduled sampling, multi-horizon loss, feature engineering u², e granularità di eval.

3. **Il bottleneck della random search non era lo spazio di ricerca**, ma il fatto di ottimizzare solo gli iperparametri dell'architettura lasciando fisso (e sub-ottimale) il processo di training.

4. **Prossimo esperimento naturale:** addestrare la best config RS (PhysicsResidualLSTM, h=128, L=3) con il training pipeline dell'autoresearch (scheduled sampling + multi-horizon + eval_every=5) per separare il contributo architetturale dal contributo del training.
