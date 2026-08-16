# GRAG Router - Quick Reference Card

## Getting Started (30 seconds)

```bash
# Start the fast demo
python grag_router_demo.py

# Type a question
[GRAG Router] > Which films share a director with [Film A]?

# See routing decision, comparison, and analysis
# Type 'exit' to quit
```

---

## The Three Routes Explained

### ⚡ OptimizedGraphRAG (Selected 97.6%)
```
✓ Fastest: 13.2ms
✓ Most efficient: 9.7 units
✓ Good quality: 0.226
✓ Use for: Simple queries
```

**How:** Capsule-based bounded search (96% fewer nodes)

### 🔍 GraphRAG (Selected 2.3%)
```
○ Medium speed: 36.7ms
○ Comprehensive: 252 units
○ Solid quality: 0.134
○ Use for: Multi-hop reasoning
```

**How:** Full graph exploration up to depth 3

### 🔎 HybridRAG (Selected 0.1%)
```
△ Fast enough: 45ms
△ Standard compute: 250 units
△ Lower quality: 0.029
△ Use for: Simple facts only
```

**How:** BM25 + dense vectors combined

---

## Policy Routing Decision Process

```
1. Extract Features (9 contextual features)
   ├─ token_count (word count)
   ├─ bracketed_entity_count (entities)
   ├─ relation_term_count (keywords)
   ├─ multihop_term_count (complexity)
   └─ ... (5 more)

2. Policy Evaluation (trained on 1,000 examples)
   ├─ Assign probabilities: p(optimized), p(graph), p(hybrid)
   └─ Select highest probability route

3. Explanation
   ├─ Which features triggered selection
   ├─ Why this route best for this question
   └─ Trade-off analysis
```

---

## Understanding Policy Confidence

```
92.1% confidence  = Policy very certain
  └─ Clear route choice, well-learned pattern

65.3% confidence  = Policy somewhat uncertain
  └─ Between two good options
  
40.0% confidence  = Policy low confidence
  └─ Ambiguous question type (rare)
```

---

## Example Routing Patterns

| Question Type | Example | Route | Why |
|--------------|---------|-------|-----|
| Simple fact | "Who acted in [Movie]?" | optimized | Direct lookup |
| Entity attribute | "What year was [Film] made?" | optimized | Single hop |
| Director films | "Which films by [Director]?" | optimized/graph | 1-2 hops, depends |
| Film relationships | "Films with same director" | graph | 2 hops + filter |
| Actor birth dates | "Birth dates of [Director]'s cast" | graph | 3+ hops |

---

## Comparison Table Interpretation

```
Route          Quality    Latency    Units    vs Selected
────────────────────────────────────────────────────────────
graph_rag       0.1338    36.70ms    252      BASELINE
hybrid_rag      0.0293    45.00ms    250      -10.5% quality
optimized       0.2262    13.20ms    9.7      +9.2% quality, -23.5ms
```

**Reading Guide:**
- Quality: Higher is better (0-0.3 range)
- Latency: Lower is better (milliseconds)
- Units: Fewer is better (retrieved evidence pieces)
- "vs Selected": How it compares to chosen route

---

## Terminal Commands

```bash
help             # Show help and examples
history          # Show all routed questions
stats            # Show session statistics
exit/quit        # Exit program
<question>       # Enter any question
```

---

## What's Happening Behind the Scenes

### Feature Extraction
```
Question: "Which films share a director with [Film A]?"
           ↓
Extracted:
  token_count: 8
  character_count: 43
  bracketed_entity_count: 1 ← [Film A]
  relation_term_count: 1 ← "director"
  multihop_term_count: 2 ← pattern suggests multi-hop
```

### Policy Scoring
```
LinearSoftmaxBandit.probabilities(features)
  ├─ Compute logits: w · features for each route
  ├─ Apply softmax: exp(logit) / Σexp(logits)
  └─ Output: {route → probability}
```

### Routing
```
Policy: optimized=0.081, graph_rag=0.789, hybrid=0.131
        └─ graph_rag has highest → SELECTED
```

### Explanation
```
Features indicate multihop_term_count=2
Policy learned: multihop → needs graph_rag
→ "Question requires deep graph reasoning"
```

---

## Performance Expectations

```
Typical Routing Session

Questions: 10
Routes selected:
  - optimized_graph_rag: 7 (70%)
  - graph_rag: 3 (30%)
  - hybrid_rag: 0 (0%)

Expected Quality:
  - Average: 0.21
  - Range: 0.03-0.23 depending on route
```

---

## Troubleshooting Quick Reference

| Problem | Cause | Fix |
|---------|-------|-----|
| "Policy not found" | Training not run | `python train_router_policy.py` |
| No answer (full interface) | Ollama not running | Start Ollama, pull models |
| Routing seems wrong | Verify features | Check question features in output |
| Slow performance | System bottleneck | Check CPU/GPU, reduce pipeline size |

---

## Key Metrics Explained

### Quality Score (0-0.3)
```
= 0.40 × answer_accuracy
+ 0.25 × faithfulness (no hallucinations)
+ 0.20 × evidence_quality
+ 0.15 × completeness
```

### Latency (milliseconds)
```
Time to get answer (varies by pipeline)
Typical: 10-50ms (route), +100-300ms (LLM)
```

### Evidence Units (count)
```
Number of retrieved document chunks
Fewer = more efficient
More = more comprehensive
```

### Success Rate (percentage)
```
% of questions that complete without error
Typical: 92-96%
```

---

## Advanced Features

### Session Statistics
```bash
[GRAG Router] > stats

Shows:
- Total questions routed: N
- Route distribution: percentages
- Average confidence per route
- Expected quality score
```

### History Review
```bash
[GRAG Router] > history

Shows:
- All questions asked this session
- Selected route for each
- Policy confidence level
- Route selection frequency
```

---

## Production Deployment Checklist

- [x] Policy trained (97.6% accuracy)
- [x] Fast demo tested
- [x] Full interface implemented
- [ ] User testing (awaiting)
- [ ] Cross-dataset validation
- [ ] Performance benchmarking
- [ ] Error handling refinement

---

## Next Steps

1. **Try it now**: `python grag_router_demo.py`
2. **Test questions**: Use examples from help menu
3. **Review decisions**: Understand routing patterns
4. **Full deployment**: When ready, run full interface

---

**Questions?** See `TERMINAL_INTERFACE_GUIDE.md` for detailed documentation.

**Technical details?** See `ROUTER_TRAINING_RESULTS.md` for training analysis.
