# Multi-Modal Evidence Review Solution

This solution implements an evidence-first claim review pipeline for the HackerRank Orchestrate challenge.

## Entry points

```bash
python code/main.py --input dataset/claims.csv --output output.csv
python code/evaluation/main.py
```

## Architecture

1. **Claim Understanding Agent** extracts issue family, object part, severity hints, multi-part claims, and prompt-injection language from the conversation.
2. **Image Validation Agent** verifies that every submitted image exists and is non-empty, records image IDs, and protects schema correctness.
3. **Visual Inspection Agent** optionally calls GPT-4o vision when `OPENAI_API_KEY` is set. It sends all images in the row with image IDs and requests strict JSON.
4. **Evidence Sufficiency / Decision Layer** normalizes the model output, enforces allowed values, and falls back to deterministic rules if no model key is available.
5. **Risk Assessment Agent** adds objective risk flags for prompt injection and risky user history.
6. **Evaluation Pipeline** runs the same code on `dataset/sample_claims.csv` and writes a report under `code/evaluation/`.

## Model configuration

Optional environment variables:

- `OPENAI_API_KEY`: enables GPT-4o image review.
- `OPENAI_MODEL`: overrides the default `gpt-4o` model.

Without an API key the system runs deterministic fallback mode, which is useful for reproducible local validation and schema checks.

## Outputs

`output.csv` is written with the exact required columns and allowed values from `problem_statement.md`.

## Cost controls

- One model call per claim row in VLM mode.
- File-based cache in `code/.cache/` keyed by row content and parsed intent.
- Temperature 0 and JSON response format for deterministic, parseable responses.
- Exponential retry for transient API failures.
