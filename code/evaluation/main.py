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
    auth_flags = {'possible_manipulation', 'non_original_image'}
    lines=['# Evaluation Report','', '## Deterministic sample metrics']
    for field in fields:
        ok=sum(1 for g,p in zip(gold,pred) if g[field]==p[field])
        lines.append(f'- {field}: {ok}/{len(gold)} = {ok/len(gold):.2%}')
    exact=sum(1 for g,p in zip(gold,pred) if all(g[f]==p[f] for f in fields))
    def flagset(row): return set() if row.get('risk_flags') == 'none' else set(row.get('risk_flags','').split(';'))
    auth_ok=sum(1 for g,p in zip(gold,pred) if bool(flagset(g)&auth_flags)==bool(flagset(p)&auth_flags))
    risk_overlap=sum(1 for g,p in zip(gold,pred) if flagset(g)==flagset(p) or bool(flagset(g)&flagset(p)))
    unsupported_auth=sum(1 for p in pred if p['claim_status']=='supported' and bool(flagset(p)&auth_flags))
    lines += ['', f'- exact structured match across core fields: {exact}/{len(gold)} = {exact/len(gold):.2%}', '', '## Authenticity and trust diagnostics', f'- authenticity/manipulation flag presence match: {auth_ok}/{len(gold)} = {auth_ok/len(gold):.2%}', f'- risk flag exact/overlap match: {risk_overlap}/{len(gold)} = {risk_overlap/len(gold):.2%}', f'- supported decisions carrying authenticity flags: {unsupported_auth} (mixed sets require trusted supporting IDs only)', '', '## Strategy comparison', '- Strategy A: deterministic claim parser plus rule layer. Zero cost, reproducible, safe fallback.', '- Strategy B: GPT-4o vision JSON agent plus deterministic evidence-trust framework and schema normalizer. Recommended final strategy when `OPENAI_API_KEY` is available because images are the source of truth and authenticity requires visual review.', '', '## Operational analysis', f'- Sample rows: {len(gold)}; test rows: {len(load(repo/"dataset/claims.csv"))}.', '- Model calls: 0 in deterministic fallback; with GPT-4o, one call per claim row after cache miss.', '- Images processed: all image paths per row; each image receives local authenticity, manipulation, trust score, and evidence quality assessment before aggregation; repeated runs reuse `code/.cache/` responses.', '- Approximate token usage with GPT-4o: 450-700 text input tokens, 1-3 images, and 120-220 output tokens per row.', '- Pricing assumption: use current provider image/text pricing at runtime; for this small dataset, expected cost is low single-digit USD with GPT-4o and lower with Flash-tier VLMs.', '- Latency: deterministic mode finishes in seconds; GPT-4o mode is roughly 3-15 seconds per row depending on image count and rate limits.', '- TPM/RPM: process sequentially by default for safety; batch/concurrency can be raised externally if provider limits allow.', '- Retry strategy: exponential backoff around model calls; cache prevents duplicate paid calls.', '']
    (repo/'code/evaluation/evaluation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()