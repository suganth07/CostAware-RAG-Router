# GRAG Evaluation Matrix - Complete Guide

This guide provides step-by-step instructions for running comprehensive evaluations of the GRAG (Graph-Augmented Retrieval and Generation) system.

## Overview

The GRAG system includes three main retrieval routes:
1. **Hybrid RAG** - Vector + BM25 sparse retrieval
2. **GraphRAG** - Full graph neighborhood exploration
3. **Optimized GraphRAG** - Capsule-based bounded graph search

A trained reinforcement learning router selects the optimal route for each question.

## Prerequisites

### 1. System Requirements
- Python 3.10+
- 8GB+ RAM recommended
- GPU optional (speeds up embedding/LLM inference)

### 2. Required Services

**Start Ollama Service:**
```bash
# Windows
ollama serve

# Linux/Mac
ollama serve &
```

**Pull Required Models:**
```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
```

**Verify Ollama is Running:**
```bash
curl http://localhost:11434/api/tags
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# For visualization features (optional but recommended)
pip install matplotlib numpy
# Or use the visualization requirements file:
pip install -r requirements_viz.txt
```

**Note:** Without matplotlib, the evaluation will still run but won't generate visualization images.

## Evaluation Options

### Option 1: Quick Evaluation (Recommended for Testing)

Run a fast evaluation on a small sample:

```bash
python run_evaluation_matrix.py --questions 10
```

**Output:**
- Evaluates 10 questions
- Tests all 3 routes + router
- Generates summary metrics
- Takes ~5-10 minutes

### Option 2: Standard Evaluation

Run standard evaluation on 50 questions:

```bash
python run_evaluation_matrix.py --questions 50
```

**Output:**
- Comprehensive metrics
- Statistical significance
- Takes ~20-30 minutes

### Option 3: Full Evaluation

Run complete evaluation on all test samples:

```bash
python run_evaluation_matrix.py --questions 500
```

**Output:**
- Complete benchmark results
- Publication-ready metrics
- Takes ~2-4 hours

### Option 4: Different Hop Levels

Evaluate on different question complexity levels:

```bash
# 1-hop (simple questions)
python run_evaluation_matrix.py --hop 1-hop --questions 50

# 2-hop (medium complexity) - DEFAULT
python run_evaluation_matrix.py --hop 2-hop --questions 50

# 3-hop (complex multi-hop questions)
python run_evaluation_matrix.py --hop 3-hop --questions 50
```

### Option 5: Custom Output Directory

Specify where to save results:

```bash
python run_evaluation_matrix.py --questions 50 --output my_results/evaluation_run_1
```

### Option 6: Quiet Mode

Suppress detailed logging:

```bash
python run_evaluation_matrix.py --questions 50 --quiet
```

## Using the Terminal Interface

The terminal interface provides an interactive way to test individual questions:

### Basic Usage

```bash
# Full interface (loads entire dataset)
python terminal_interface.py

# Fast interface (subset of data, faster startup)
python terminal_interface_fast.py
```

### Commands in Terminal Interface

Once running, you can use these commands:

```
[GRAG] > What films share a director with The Shining?
  # Asks a question, router selects best route

[GRAG] > verbose
  # Toggle detailed routing analysis on/off

[GRAG] > compare
  # Toggle comparison mode (runs all routes for each question)

[GRAG] > history
  # Show decision history

[GRAG] > stats
  # Show session statistics

[GRAG] > help
  # Show help message

[GRAG] > exit
  # Exit the interface
```

### Fast Mode with Seed Query

Pre-filter dataset for faster startup:

```bash
python terminal_interface_fast.py --seed-query "films director actor"
```

### Non-Interactive Mode

Ask questions without entering interactive loop:

```bash
python terminal_interface_fast.py --ask "Who directed The Shining?"
```

## Understanding the Output

### 1. Evaluation Matrix Output Files

After running evaluation, you'll find these files in the output directory:

#### `evaluation_matrix_TIMESTAMP.json`
Complete detailed results including:
- Every question's results for every route
- Router decisions and features
- Full answers and ground truth

#### `summary_metrics_TIMESTAMP.json`
Aggregated metrics per route:
```json
{
  "hybrid_rag": {
    "exact_match_mean": 0.67,
    "token_f1_mean": 0.72,
    "context_recall_mean": 0.85,
    "latency_mean_ms": 234.5,
    "latency_p50_ms": 220.0,
    "latency_p95_ms": 350.0,
    "retrieved_units_mean": 45.2
  },
  ...
}
```

#### `comparison_matrix_TIMESTAMP.json`
Comparative analysis:
```json
{
  "router_accuracy": 0.78,
  "optimized_vs_normal_speedup": 2.34,
  "quality_comparison": {...},
  "latency_comparison": {...},
  "efficiency_comparison": {...}
}
```

#### `detailed_results_TIMESTAMP.csv`
Spreadsheet-friendly format with all results:
- One row per (question, route) pair
- Easy to import into Excel/Google Sheets
- Good for custom analysis

#### Visual Output Files (Generated with matplotlib)

**`performance_heatmap_TIMESTAMP.png`**
- Color-coded heatmap showing all metrics across routes
- Green = better performance, Red = worse performance
- Includes: Exact Match, Token F1, Context Recall, Latency, Efficiency

**`quality_vs_latency_TIMESTAMP.png`**
- Scatter plot showing quality/speed trade-offs
- Bubble size represents retrieved units (cost)
- Identifies ideal region (high quality, low latency)

**`metrics_comparison_TIMESTAMP.png`**
- Bar charts comparing all key metrics
- Side-by-side comparison of EM, F1, Context Recall, Latency
- Easy visual identification of best-performing routes

**`router_analysis_TIMESTAMP.png`** (if router is available)
- Pie chart: Route selection distribution
- Box plot: Router confidence levels per route
- Accuracy gauge: Overall router performance
- Bar chart: Speedup metrics

### 2. Terminal Output

The evaluation prints a summary table:

```
================================================================================
ROUTE PERFORMANCE METRICS
================================================================================

Route                      EM       F1   C-Recall     Latency    Units
--------------------------------------------------------------------------------
hybrid_rag                67.2%    72.4%      85.1%      234 ms       45
graph_rag                 71.8%    76.9%      89.3%      542 ms      187
optimized_graph_rag       70.5%    75.8%      88.7%      231 ms       38

--------------------------------------------------------------------------------
COMPARISON ANALYSIS
--------------------------------------------------------------------------------

Router Accuracy: 78.3%
  (Percentage of times router selected best-performing route)

Optimized vs Normal GraphRAG Speedup: 2.34x
  (135% faster)

Latency Ranking:
  1. optimized_graph_rag       231 ms
  2. hybrid_rag                234 ms
  3. graph_rag                 542 ms

Quality Ranking (by Token F1):
  1. graph_rag                 F1= 76.9%
  2. optimized_graph_rag       F1= 75.8%
  3. hybrid_rag                F1= 72.4%
```

### 3. Key Metrics Explained

**Exact Match (EM)**: Percentage of answers that exactly match ground truth (after normalization)

**Token F1**: Harmonic mean of precision and recall at token level (more lenient than EM)

**Context Recall**: Percentage of questions where the answer appears in retrieved context

**Latency**: Time taken to process the question
- Mean: Average across all questions
- P50: Median latency
- P95: 95th percentile (useful for tail latency)

**Retrieved Units**: Number of evidence pieces retrieved (documents, chunks, paths)

**Router Accuracy**: How often the router picks the best-performing route

**Speedup**: How much faster one approach is compared to another

## Example Workflows

### Workflow 1: Quick Test Before Full Run

```bash
# Test with 10 questions first
python run_evaluation_matrix.py --questions 10

# If successful, run full evaluation
python run_evaluation_matrix.py --questions 500
```

### Workflow 2: Compare Across Hop Levels

```bash
# Evaluate all hop levels
python run_evaluation_matrix.py --hop 1-hop --questions 100 --output results/1hop
python run_evaluation_matrix.py --hop 2-hop --questions 100 --output results/2hop
python run_evaluation_matrix.py --hop 3-hop --questions 100 --output results/3hop

# Then compare the summary files
```

### Workflow 3: Interactive Exploration + Batch Evaluation

```bash
# First, explore interactively
python terminal_interface_fast.py
# Try various questions to understand system behavior

# Then run systematic evaluation
python run_evaluation_matrix.py --questions 100
```

### Workflow 4: Continuous Evaluation During Development

```bash
# Run evaluation, save to timestamped directory
python run_evaluation_matrix.py --questions 50 --output results/$(date +%Y%m%d_%H%M%S)

# Compare results over time by examining different timestamp directories
```

## Troubleshooting

### Error: "Ollama is not reachable"
**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Error: "Policy not found"
**Solution:**
The router policy file should be at `artifacts/router/policies/m4_contextual_bandit.json`. If missing, the evaluation will skip router analysis but still evaluate all routes.

### Error: "CUDA out of memory"
**Solutions:**
1. Reduce batch size in evaluation script
2. Use CPU-only mode (slower but works)
3. Evaluate fewer questions at a time
4. Close other GPU-using applications

### Slow Performance
**Solutions:**
1. Use fast mode: `python terminal_interface_fast.py`
2. Reduce question count: `--questions 10`
3. Use embedding cache (already enabled by default)
4. Consider using GPU for faster inference

### Missing Dependencies
**Solution:**
```bash
pip install -r requirements.txt
```

## Advanced Usage

### Programmatic Access

You can also use the evaluator programmatically:

```python
from run_evaluation_matrix import GRAGEvaluator

# Initialize evaluator
evaluator = GRAGEvaluator(
    dataset_name="metaqa",
    hop="2-hop",
    num_questions=50,
    verbose=True
)

# Run evaluation
evaluator.initialize()
matrix = evaluator.evaluate()

# Access results
print(f"Router accuracy: {matrix.comparison_matrix['router_accuracy']:.1%}")
print(f"Speedup: {matrix.comparison_matrix['optimized_vs_normal_speedup']:.2f}x")

# Save results
paths = evaluator.save_results(matrix, "my_results")
```

### Custom Analysis

Load results for custom analysis:

```python
import json
import pandas as pd

# Load JSON results
with open("results/evaluation_matrix/evaluation_matrix_20260907_123456.json") as f:
    data = json.load(f)

# Load CSV for pandas analysis
df = pd.read_csv("results/evaluation_matrix/detailed_results_20260907_123456.csv")

# Analyze by route
by_route = df.groupby('route').agg({
    'exact_match': 'mean',
    'token_f1': 'mean',
    'latency_ms': ['mean', 'median', 'std']
})
print(by_route)
```

## Interpreting Results for Research/Production

### For Research Papers
- Use full evaluation (500 questions)
- Report all metrics (EM, F1, Context Recall)
- Include latency percentiles (P50, P95)
- Show comparison matrices
- Report router accuracy

### For Production Deployment
- Focus on P95 latency (tail latency matters)
- Monitor router accuracy over time
- Track retrieved units (cost proxy)
- Balance quality vs. speed tradeoff
- Consider question distribution (hop levels)

### For System Optimization
- Identify bottlenecks from detailed CSV
- Analyze router decisions for pattern insights
- Find questions where routes disagree significantly
- Optimize based on your workload characteristics

## Next Steps

1. **Baseline Evaluation**: Run standard 50-question evaluation
2. **Analyze Results**: Review summary metrics and comparison matrix
3. **Interactive Testing**: Use terminal interface to understand behavior
4. **Full Benchmark**: Run 500-question evaluation for complete metrics
5. **Custom Analysis**: Use CSV/JSON outputs for domain-specific insights

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Verify all prerequisites are met
4. Check that Ollama is running and models are loaded
5. Try with fewer questions first (`--questions 10`)

## Summary of Commands

```bash
# Quick test
python run_evaluation_matrix.py --questions 10

# Standard evaluation  
python run_evaluation_matrix.py --questions 50

# Full benchmark
python run_evaluation_matrix.py --questions 500

# Interactive terminal
python terminal_interface.py

# Fast interactive terminal
python terminal_interface_fast.py

# Different hop levels
python run_evaluation_matrix.py --hop 1-hop --questions 50
python run_evaluation_matrix.py --hop 2-hop --questions 50
python run_evaluation_matrix.py --hop 3-hop --questions 50
```
