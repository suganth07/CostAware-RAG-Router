# Experiment Notebooks

The canonical reproducible evaluator is:

```powershell
python scripts\run_grag_evaluation.py --dataset metaqa_1hop --system both
```

Use the notebooks for interactive inspection and plots.

## `01_compare_grag_vs_optimized.ipynb`

The comparison notebook supports MetaQA, 2WikiMultiHopQA, and MultiHop-RAG. It
uses the persistent embedding cache and cached evaluation runner from the library.
Unified MetaQA defaults are 100 first-stage capsules, 12 selected capsules, 500
path candidates, and 20 final paths balanced across depths 1–3.

The notebook reports span-context recall separately from boolean/null questions.
For complete supporting-evidence Recall@K, MRR, EM/F1, boolean accuracy, raw
records, and automatic fingerprint invalidation, use the CLI.

## `02_diagnose_rgcn_projection.ipynb`

This notebook checks artifact inventory, training curves, embedding collapse,
projection margins, and missing/suspicious artifacts. Runtime additionally
requires matching capsule fingerprints before enabling trained structural space.
