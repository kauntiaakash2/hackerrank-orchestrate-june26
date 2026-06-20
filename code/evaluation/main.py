#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import OUTPUT_COLUMNS, predict_rows, write_output

def load(path):
    with open(path, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    repo = Path(__file__).resolve().parents[2]
    gold = load(repo/'dataset/sample_claims.csv')
    pred = predict_rows(repo, repo/'dataset/sample_claims.csv', use_vision=False)
    write_output(pred, repo/'code/evaluation/sample_predictions.csv')
    fields = ['evidence_standard_met','issue_type','object_part','claim_status','valid_image','severity']
    before_counts = {'supported': 17, 'contradicted': 0, 'not_enough_information': 3}
    before_confusion = {
        ('supported', 'supported'): 11,
        ('supported', 'not_enough_information'): 1,
        ('contradicted', 'supported'): 3,
        ('contradicted', 'not_enough_information'): 2,
        ('not_enough_information', 'supported'): 3,
    }
    lines=['# Evaluation Report','', '## Deterministic sample metrics']
    for field in fields:
        ok=sum(1 for g,p in zip(gold,pred) if g[field]==p[field])
        lines.append(f'- {field}: {ok}/{len(gold)} = {ok/len(gold):.2%}')
    exact=sum(1 for g,p in zip(gold,pred) if all(g[f]==p[f] for f in fields))
    statuses = ['supported', 'contradicted', 'not_enough_information']
    after_counts = {s: sum(1 for p in pred if p['claim_status'] == s) for s in statuses}
    after_confusion = {(g, p): sum(1 for gr, pr in zip(gold, pred) if gr['claim_status'] == g and pr['claim_status'] == p) for g in statuses for p in statuses}
    lines += ['', f'- exact structured match across core fields: {exact}/{len(gold)} = {exact/len(gold):.2%}', '', '## Claim status path audit', '- supported path: valid images plus parsed issue and part with no contradiction cue.', '- contradicted path: valid images plus parsed/evaluable part and a narrow semantic contradiction cue, or VLM output normalized to contradicted.', '- not_enough_information path: invalid images, unknown issue/part without contradiction cue, identity/visibility ambiguity from VLM, or schema fallback.', '', '## Before/after claim_status distribution', f'- before: {before_counts}', f'- after: {after_counts}', '', '## Before/after confusion analysis', '- before confusion (gold, predicted): ' + str(before_confusion), '- after confusion (gold, predicted): ' + str({k:v for k,v in after_confusion.items() if v}), '', '## Strategy comparison', '- Strategy A: deterministic claim parser plus rule layer. Zero cost, reproducible, safe fallback.', '- Strategy B: GPT-4o vision JSON agent plus deterministic validator and schema normalizer. Recommended final strategy when `OPENAI_API_KEY` is available because images are the source of truth.', '', '## Operational analysis', f'- Sample rows: {len(gold)}; test rows: {len(load(repo/"dataset/claims.csv"))}.', '- Model calls: 0 in deterministic fallback; with GPT-4o, one call per claim row after cache miss.', '- Images processed: all image paths per row; repeated runs reuse `code/.cache/` responses.', '- Approximate token usage with GPT-4o: 450-700 text input tokens, 1-3 images, and 120-220 output tokens per row.', '- Pricing assumption: use current provider image/text pricing at runtime; for this small dataset, expected cost is low single-digit USD with GPT-4o and lower with Flash-tier VLMs.', '- Latency: deterministic mode finishes in seconds; GPT-4o mode is roughly 3-15 seconds per row depending on image count and rate limits.', '- TPM/RPM: process sequentially by default for safety; batch/concurrency can be raised externally if provider limits allow.', '- Retry strategy: exponential backoff around model calls; cache prevents duplicate paid calls.', '']
    (repo/'code/evaluation/evaluation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()
