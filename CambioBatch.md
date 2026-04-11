  ---
  Batch size 20 → 8

  Cosa cambia nel gradiente:
  La loss è calcolata come media sulla batch, non somma. Quindi la scala della loss non cambia, ma ogni passo di gradiente è stimato su 8 campioni invece di 20 → stima più rumorosa, aggiornamenti più
  "nervosi".

  Effetti pratici:
  - Curve di training più irregolari (val_loss oscillerà di più)
  - Potenzialmente peggiore convergenza per epoch — ma lo stesso LR (5e-5) con batch più piccolo equivale implicitamente a un learning rate effettivo più alto. La regola empirica (linear scaling rule)
  suggerirebbe di ridurre il LR proporzionalmente (~2e-5), ma AdamW è abbastanza robusto da tollerarlo
  - L'early stopping potrebbe scattare prima, o su un epoch diverso rispetto a lb1

  Cosa NON cambia:
  La valutazione PCK carica semplicemente il _best.pt salvato durante il training. La PCK non dipende da batch size o precision — è identica.

  ---
  Precision bf16 → fp16

  Differenza tecnica:
  - bf16: range grande, mantissa corta (7 bit) → meno preciso sui valori piccoli
  - fp16: range più piccolo, mantissa lunga (10 bit) → più preciso, ma overflow a valori alti

  Con fp16 il codice attiva automaticamente il GradScaler (scala la loss prima del backward per evitare underflow nei gradienti). Questo è esattamente il comportamento corretto su Pascal.

  Effetti pratici: trascurabili sulla qualità finale. I checkpoint sono numericamente leggermente diversi ma comparabili.

  ---
  Cosa tenere d'occhio

  1. Lb1 e lb2/lb4 sono stati addestrati sotto condizioni diverse

  ┌────────────┬────────────────┬───────────────────┐
  │            │      lb1       │     lb2, lb4      │
  ├────────────┼────────────────┼───────────────────┤
  │ Precision  │ bf16 (emulato) │ fp16 + GradScaler │
  ├────────────┼────────────────┼───────────────────┤
  │ Batch size │ 20             │ 8                 │
  └────────────┴────────────────┴───────────────────┘

  Per il confronto Stage 2 (quanti blocchi sbloccati conviene usare), questo introduce una piccola confusione: le differenze tra lb1 e lb2 riflettono sia il numero di blocchi che le condizioni di training.
  Per un progetto universitario è accettabile, ma è un limite da menzionare.

  2. GradScaler: skip step

  Se durante l'fp16 training un gradiente va in overflow, il GradScaler salta quel batch e riduce la scala interna. Lo vedrai nel log come nulla (è silenziosa). Se succede spesso, potrebbe rallentare la
  convergenza — ma è raro con questa loss.

  3. Early stopping con batch piccolo

  Con gradiente più rumoroso, la val_loss potrebbe migliorare e peggiorare più erraticamente. È possibile che early stopping scatti prima dell'ottimo reale. Se vedi che lb2/lb4 si fermano molto presto (2-3
  epoch), considera di aumentare FT_PATIENCE da 7 a 10.