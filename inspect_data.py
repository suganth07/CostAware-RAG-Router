#!/usr/bin/env python
import json
import sys

# Read first outcome
with open("artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl") as f:
    line = f.readline()
    outcome = json.loads(line)

print("=== OUTCOME STRUCTURE ===")
print(f"Top-level keys: {list(outcome.keys())}")
print()

print("=== SAMPLE VALUES ===")
print(f"question: {outcome.get('question', 'N/A')[:100]}...")
print(f"route: {outcome.get('route', 'N/A')}")
print(f"features: {outcome.get('features', {})}")
print()

print("=== METRICS KEYS ===")
if 'metrics' in outcome:
    print(f"metrics: {list(outcome['metrics'].keys())}")
    print(f"Sample metrics: {json.dumps({k: v for k, v in list(outcome['metrics'].items())[:3]}, indent=2)}")
print()

print("=== LATENCY & COMPUTE ===")
print(f"latency_seconds: {outcome.get('latency_seconds')}")
print(f"retrieved_units: {outcome.get('retrieved_units')}")
print()

# Count records by route
print("=== COUNTING RECORDS ===")
routes = {"hybrid_rag": 0, "graph_rag": 0, "optimized_grag": 0}
questions_seen = set()
with open("artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl") as f:
    for i, line in enumerate(f):
        outcome = json.loads(line)
        routes[outcome.get("route")] = routes.get(outcome.get("route"), 0) + 1
        questions_seen.add(outcome.get("sample_id"))
        if i >= 5:  # Just check first few to verify pattern
            break

print(f"Total questions (estimate): {len(questions_seen)} unique")
print(f"Routes distribution (first 6 records): {routes}")
print()

# Get full count
import subprocess
result = subprocess.run(['wc', '-l', 'artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl'], 
                       capture_output=True, text=True)
total_lines = int(result.stdout.split()[0])
print(f"Total records in file: {total_lines}")
print(f"Estimated unique questions: {total_lines // 3}")
