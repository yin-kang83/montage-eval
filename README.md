# montage-eval

A small, reproducible harness for the **negative result**:

> Three query-agnostic selection heuristics do not outperform tail truncation under a fixed memory budget.

This repository is intentionally **data-free**: it ships a synthetic dialogue generator. It does not contain the private dialogue archive used in the paper.

## What is included

```text
montage_eval/core.py      six strategies + probe hit + budget
scripts/make_synthetic.py synthetic conversations
scripts/run_eval.py       run the harness on a local archive
tests/test_smoke.py       smoke test
```

Strategies:

```text
last        tail truncation
uniform     sequential prefix (historical code name: uniform)
event       fixed 4-message event density
event_cov   event coverage
bm25        query-aware lexical retrieval (oracle query caveat)
oracle      manipulation-check upper bound
```

## Quick start

```bash
python scripts/make_synthetic.py
python scripts/run_eval.py
```

Output:

```text
results/montage_results.json
```

## Important caveats

```text
1. The BM25 baseline uses a query term constructed from the target fact.
   It is an oracle query upper bound, not a realistic RAG gain.
2. The probe metric is literal substring recall; it is a conservative lower bound.
   A 30-item audit is in the paper, not in this repo.
3. The private archive is not released.
4. This repo is a research artifact, not a production service.
```

## License

MIT.
