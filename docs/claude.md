# Note di lavoro (italiano)

Questo file integra [`info.md`](info.md) con convenzioni e aspettative operative descritte nel README.

## Allineamento con il codice

- **Split**: training su `train`, early stopping su `val`, numeri da benchmark su `test` — come in [`data/dataset.py`](../data/dataset.py).
- **Pesi**: pesi ufficiali o percorsi locali; mirror Hugging Face ammessi **solo** come fallback se le URL ufficiali falliscono (con verifica SHA256 quando disponibile, vedi `info.md` e `scripts/download_pretrained_weights.py`).
- **WSA**: solo a inferenza/valutazione, non nella loss di training.

## Layout dati SPair-71k

- Liste coppie: `Layout/large/{trn,val,test}.txt`.
- Annotazioni: `PairAnnotation/<split>/`.
- Immagini: `JPEGImages/<categoria>/`.

Per verificare percorsi e un campione per split: `python scripts/verify_dataset.py` dalla root del repo (venv attivo).

## Dove approfondire

- Guida pratica: [`README.md`](../README.md).
- Panorama letteratura / riferimenti cartelle modelli: [`stato-arte.md`](stato-arte.md).
