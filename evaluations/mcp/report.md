# Search mode comparison: semantic-search-demo

- Run: 2026-10-23 against `https://semantic-search-demo.stanford.edu/mcp` (`catalog_search_tool`, 20 rows, no filters)
- Queries: 31, scored at cutoffs 1, 5, 10
- Relevance: graded 0–3. nDCG uses gain 2^g−1; MRR, recall and success count any grade ≥ 1 as relevant. Unjudged results score 0.

## Overall

| Mode | nDCG@5 | nDCG@10 | MRR@10 | Recall@10 | Success@1 | Judged@10 |
|---|---|---|---|---|---|---|
| Keyword | 0.27 | 0.28 | 0.41 | 0.27 | 0.35 | 0.12 |
| Vector | **0.50** | **0.51** | **0.73** | **0.48** | **0.61** | 0.28 |
| Hybrid | 0.46 | 0.46 | 0.69 | 0.44 | 0.58 | 0.24 |

## Best mode by test type (nDCG@10)

| test_type (n) | Keyword | Vector | Hybrid | Best |
|---|---|---|---|---|
| conceptual_thematic (19) | 0.16 | **0.47** | 0.39 | Vector |
| fact_lookup (3) | 0.40 | **0.48** | 0.43 | Vector |
| table_extraction (3) | 0.67 | **0.82** | 0.74 | Vector |
| chunk_boundary (2) † | 0.50 | 0.63 | **0.82** | Hybrid |
| synonym_match (2) † | 0.13 | 0.21 | **0.35** | Hybrid |
| keyword_search (1) † | **0.61** | **0.61** | **0.61** | tie |
| known_item (1) † | **0.61** | **0.62** | **0.62** | tie |
| **All** (31) | 0.28 | **0.51** | 0.46 | Vector |

Bold marks the leader (within 0.02); “tie” means no mode leads by ≥ 0.02. † n ≤ 2, anecdotal.

## Head-to-head (nDCG@10, per query)

| Pair | Wins | Ties | Losses |
|---|---|---|---|
| Vector vs Keyword | 21 | 5 | 5 |
| Hybrid vs Keyword | 22 | 7 | 2 |
| Hybrid vs Vector | 8 | 7 | 16 |

## Largest disagreements

nDCG@10 per mode, with the rank of the first top-graded document in parentheses (– = not in top 20).

| Test | Type | Query | Keyword | Vector | Hybrid |
|---|---|---|---|---|---|
| `ua1-stanford-news-service-5` | conceptual_thematic | What were the most controversial events or activities at Stanford i… | 0.00 (–) | 0.90 (1) | 0.53 (1) |
| `ua1-stanford-news-service-3` | conceptual_thematic | What major changes occurred amongst the academic program at Stanfor… | 0.00 (–) | 0.79 (1) | 0.45 (2) |
| `campus-history-1` | conceptual_thematic | What was Stanford's role in the development of Silicon Valley? | 0.15 (3) | 0.89 (1) | 0.53 (1) |
| `ua1-stanford-news-service-4` | conceptual_thematic | What were the most impactful reports issued by the Stanford Faculty… | 0.00 (18) | 0.68 (1) | 0.55 (1) |
| `ua-tables-chunk-boundaries-5` | chunk_boundary | According to Stanford's Courses and degrees for 1957-1958, who taug… | 0.00 (14) | 0.63 (2) | 0.63 (2) |

## Caveats

- Small samples: 4 of 7 test types have ≤ 2 queries; treat their rows as anecdotes, not trends.
- The MCP tool returns at most 20 objects, so recall is bounded for queries with many relevant documents.
