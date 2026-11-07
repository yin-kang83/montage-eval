# Security

This repository is a research artifact.

- It contains no private dialogue data, API keys, or customer cases.
- `scripts/make_synthetic.py` generates synthetic conversations only.
- Do not commit `data/archive/`, `results/`, `.env`, or key files.
- The BM25 baseline is an oracle-query upper bound, not a production retriever.
- The probe metric is literal substring recall, not end-to-end QA accuracy.

Report issues via a private channel if the repository is hosted publicly.
