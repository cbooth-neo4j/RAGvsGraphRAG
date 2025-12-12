# Global GraphRAG Retriever Optimization (Before/After) + Client Recommendations

## Executive summary

We optimized the **global** GraphRAG retrieval path to reduce both **cold-start overhead** and **LLM round-trips**.

- **Before**: full-graph eager loads + map/reduce over many community batches (multiple LLM calls).
- **After**: global-only retriever that fetches **top‑k communities per query** and answers in **one LLM call**.

Measured on the same query set in the same environment:
- **Median latency** improved from **~28.5s → ~7.1s** (**~304% speed-up**)
- **P95 latency** improved from **~30.0s → ~11.6s** (**~160% speed-up**)
- **LLM calls** reduced from **3 → 1** per global query

Reference comparison output: [`benchmark/results/global_before_after_report.md`](../benchmark/results/global_before_after_report.md)

## What changed (technical)

### 1) Global-only cold-start-friendly retriever

- Added `LightweightAdvancedGlobalOnlyRetriever` which avoids:
  - Loading entities/relationships/text units/communities into memory
  - Initializing `Neo4jVectorStore` (and its index checks/creation)

### 2) Query-conditioned community selection (top‑k)

- Global context is built from **top‑k community reports** fetched from Neo4j per query.
- Current selection strategy is a safe fallback for OpenAI-only dev:
  - **Keyword match** on `__Community__.summary/title` if query terms exist
  - Otherwise **rank-based** top‑k fallback (`community_rank`)

### 3) Single-pass global answering (1 LLM call)

- Added `GlobalSearchSinglePass` which:
  - Builds community context once
  - Calls the LLM once (`ainvoke` if available; thread fallback otherwise)

### 4) Provider-agnostic async invocation

- Added `_ainvoke_compatible()`:
  - Uses `model.ainvoke(...)` when supported (OpenAI/VertexAI LangChain chat models)
  - Falls back to running sync `model.invoke(...)` in a worker thread

### 5) Performance harness for before/after benchmarking

- Added:
  - `benchmark/perf_global_search.py`
  - `benchmark/queries/global.json`
  - `benchmark/report_global_perf.py` (generates markdown comparison)

## Before/after performance (measured)

Comparison file: [`benchmark/results/global_before_after_report.md`](../benchmark/results/global_before_after_report.md)

Key deltas observed:
- **Total time**: ~28.5s median → ~7.1s median
- **LLM calls**: 3 → 1
- **New timing breakdown** in optimized path:
  - `neo4j_ms`: ~2.0s (dominant fixed cost)
  - `llm_ms`: typically ~5–9s but can spike (dominant variable cost)
  - `context_ms`: ~sub-10ms (effectively solved)

## What drives `llm_ms` (and what the client can tweak)

`llm_ms` is *not just “model compute”*; it includes network + provider queueing + generation time + any retries.

### High-impact knobs (recommendations)

1) **Cap output tokens for global answers**
   - Set `max_tokens` (OpenAI) / `max_output_tokens` (VertexAI).
   - This is the most reliable way to prevent 15–20s outliers.

2) **Force a compact response format**
   - E.g., “max 8 bullets, 1–2 sentences each, then a 1-paragraph conclusion”.
   - This preserves quality for “global overview” while bounding generation work.

3) **Brief-first UX**
   - Default: concise answer.
   - If user asks “go deeper”, run a second call to expand.
   - Net: faster median with no loss of capability.

4) **Tune `top_k_communities`**
   - More communities increases prompt size and synthesis complexity (slower, sometimes higher quality).
   - Start with `top_k_communities=6–10`, measure; only increase if needed.

5) **Model selection and reliability**
   - You’re already using a fast model (`gpt-4.1-mini` in dev).
   - For prod VertexAI: pick a model tier that matches required synthesis quality; cap output to stabilize latency.

### Operational knobs (often overlooked)

- **Retries/backoff**: large retry budgets can inflate p95 during transient provider issues.
- **Streaming**: doesn’t reduce total `llm_ms`, but improves time-to-first-token UX.
- **Concurrency limits**: if multiple users hit global simultaneously, enforce limits to avoid provider throttling cascades.

## Remaining bottlenecks / next steps

### 1) Reduce Neo4j time (`neo4j_ms` ~2s)

Current query-conditioned selection uses keyword `CONTAINS`, which can be slow without indexing.

Recommended upgrade:
- Add a **community summary vector embedding** (`__Community__.summary_embedding`)
- Create a **Neo4j vector index** (e.g. `community_summary_embedding`)
- Retrieve top‑k via `db.index.vector.queryNodes(...)`

This should reduce DB time and improve relevance at the same time.

### 2) Add full timing breakdown to map/reduce (optional)

If the client still wants map/reduce fallback, add:
- per-map `llm_ms`
- reduce `llm_ms`
- total `neo4j_ms/context_ms`

## How to reproduce benchmarks

Optimized single-pass:

```bash
python -m benchmark.perf_global_search --impl optimized --runs 1 --cold-start --strategy single_pass
```

Legacy baseline (full-graph + map-reduce):

```bash
python -m benchmark.perf_global_search --impl baseline --runs 1 --cold-start --strategy map_reduce
```

Generate comparison markdown:

```bash
python -m benchmark.report_global_perf --before <baseline_json> --after <optimized_json> --out benchmark/results/global_before_after_report.md
```


