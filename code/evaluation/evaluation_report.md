# Evaluation Report

## Deterministic sample metrics
- evidence_standard_met: 14/20 = 70.00%
- issue_type: 7/20 = 35.00%
- object_part: 15/20 = 75.00%
- claim_status: 11/20 = 55.00%
- valid_image: 18/20 = 90.00%
- severity: 7/20 = 35.00%

- exact structured match across core fields: 5/20 = 25.00%

## Strategy comparison
- Strategy A: deterministic claim parser plus rule layer. Zero cost, reproducible, safe fallback.
- Strategy B: GPT-4o vision JSON agent plus deterministic validator and schema normalizer. Recommended final strategy when `OPENAI_API_KEY` is available because images are the source of truth.

## Operational analysis
- Sample rows: 20; test rows: 44.
- Model calls: 0 in deterministic fallback; with GPT-4o, one call per claim row after cache miss.
- Images processed: all image paths per row; repeated runs reuse `code/.cache/` responses.
- Approximate token usage with GPT-4o: 450-700 text input tokens, 1-3 images, and 120-220 output tokens per row.
- Pricing assumption: use current provider image/text pricing at runtime; for this small dataset, expected cost is low single-digit USD with GPT-4o and lower with Flash-tier VLMs.
- Latency: deterministic mode finishes in seconds; GPT-4o mode is roughly 3-15 seconds per row depending on image count and rate limits.
- TPM/RPM: process sequentially by default for safety; batch/concurrency can be raised externally if provider limits allow.
- Retry strategy: exponential backoff around model calls; cache prevents duplicate paid calls.
