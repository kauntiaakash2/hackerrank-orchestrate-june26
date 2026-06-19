# Challenge Reverse Engineering, Architecture, and Judge Preparation

## Phase 1 — Reverse Engineering

### Real objective
The evaluator is checking whether a system can turn a short conversational claim and multiple images into a conservative, schema-valid insurance evidence decision. The primary scoring signal is likely agreement with labeled hidden rows on `claim_status`, `issue_type`, `object_part`, `evidence_standard_met`, `valid_image`, `severity`, `supporting_image_ids`, and risk flags. Strong systems are image-first, ignore adversarial approval instructions, separate user-history risk from visual truth, and produce concise justifications grounded in image IDs.

### Hidden requirements and assumptions
- **Exact CSV schema:** Output columns and allowed values must match exactly, because automated judging will parse `output.csv`.
- **Image-first reasoning:** User text says what to check, but images decide support, contradiction, or insufficient evidence.
- **Conservative insufficiency:** Cropped, wrong-angle, wrong-object, or conflicting multi-image evidence should become `not_enough_information`, not unsupported guesses.
- **Risk is contextual:** `user_history_risk` and prompt-injection language should add flags and manual review, but should not override clear visual evidence by themselves.
- **Multilingual and noisy chat:** Hidden cases may include Hindi, Spanish, code-switching, irrelevant context, and late corrections of the actual claim.
- **Prompt injection:** Claims or images may contain text asking the system to approve; these are risk flags, not evidence.
- **Multi-image conflicts:** A close-up can show damage while a context image shows a different object; this should trigger mismatch/manual review.

### Atomic task graph
1. **Claim extraction:** Input conversation and object type; output target issue, part, severity hints, and injection indicators. Failure modes: over-weighting early irrelevant mentions, missing multilingual terms.
2. **Image validation:** Input image paths; output image IDs, existence/usability, and quality risks. Failure modes: missing files, unreadable images, too-cropped evidence.
3. **Visual verification:** Input images and intent; output visible object, part, damage type, severity, and supporting images. Failure modes: hallucinating invisible damage or trusting text overlays.
4. **Evidence sufficiency:** Input requirements, intent, and visual observations; output evidence-standard decision. Failure modes: accepting close-ups without identity context where identity matters.
5. **Risk assessment:** Input history, claim text, and visual observations; output risk flags. Failure modes: rejecting based only on user history.
6. **Decision:** Input all structured signals; output supported, contradicted, or not enough information. Failure modes: binary accept/reject thinking, schema drift.
7. **Explanation:** Input decision and supporting images; output concise image-grounded reasons. Failure modes: unsupported rationales or missing image IDs.

### Challenge risks
- Hallucination from VLMs on subtle dents, scratches, glare, and reflections.
- OCR traps where an image or chat tells the model to approve.
- Visual ambiguity from close-ups without context.
- Multi-image conflict between object identity, color, or part.
- Misuse of user history as a substitute for visual evidence.
- Schema compliance errors: booleans as `True`, comma-separated flags, invalid parts.

### Winning strategy tiers
- **Baseline:** text parser plus simple rules; schema valid but weak visual accuracy.
- **Good:** VLM per row with strict prompt and allowed values; reasonable hidden performance.
- **Top 10%:** VLM observations separated from deterministic decision/risk layers, caching, evaluation report, prompt-injection handling, and conservative insufficiency.
- **Winning:** ensemble or adjudicated VLM observations, part/identity consistency checks, calibrated severity, manual-review triage, repeatable evaluation, and strong interview defensibility.

## Phase 2 — System Design

### High-level architecture and data flow
CSV rows are loaded, claim intent is parsed, image paths are validated, optional GPT-4o vision inspects all images with explicit image IDs, deterministic validators normalize the JSON, and the decision layer writes the required CSV. The fallback mode uses deterministic extraction and risk rules so the system remains runnable without secrets.

### Component responsibilities
- **Claim Understanding Agent:** extracts part, issue family, severity hints, multi-part intent, prompt injection. It fails closed to `unknown` when ambiguous.
- **Image Validation Agent:** verifies file existence and basic usability and maps filenames to supporting image IDs.
- **Visual Inspection Agent:** calls GPT-4o with all row images, temperature 0, JSON output, and explicit instructions to ignore approval text.
- **Evidence Sufficiency Agent:** checks whether the relevant object part is visible and enough images support identity/part review.
- **Risk Assessment Agent:** adds text-instruction, history, mismatch, quality, and manual-review flags.
- **Decision Agent:** maps observations to `supported`, `contradicted`, or `not_enough_information` using deterministic rules rather than raw LLM text alone.
- **Explanation Agent:** creates short justifications citing image IDs and never inventing unseen evidence.

### Model selection
- **GPT-4o:** best default for this small multimodal insurance task because of strong VLM reasoning, JSON support, and broad multilingual robustness.
- **Gemini 2.5 Pro:** strong alternative for visual reasoning and long context, but may be slower/costlier for a small CSV.
- **Gemini Flash:** cost-effective fallback for large batches, lower reasoning depth.
- **Claude Sonnet / Opus:** strong language reasoning and explanations; visual capability depends on available platform and JSON strictness.
- **Qwen VL / InternVL:** viable local/open VLMs if GPU is available, but operationally risky in a short hackathon.
- **Florence, GroundingDINO, YOLO:** useful for object localization/detection, but they do not solve subtle damage semantics alone without training or labels.

Selected stack: GPT-4o vision plus deterministic parser, validator, risk layer, cache, and schema normalizer. Deterministic fallback is included for reproducibility.

### Decision framework
- **Supported:** the relevant object part is visible, issue type and severity are materially consistent with the claim, and no fatal mismatch exists.
- **Contradicted:** the part is visible but the claimed issue/type/severity is absent or materially different.
- **Not enough information:** relevant part is not visible, images are unusable, identity conflicts exist, or multi-image evidence cannot establish the claim.

## Phase 3 — Implementation Plan
The repository keeps `code/main.py` as the runnable entry point, `code/evaluation/main.py` as the evaluator, `code/README.md` as operational documentation, and `code/evaluation/evaluation_report.md` as generated metrics. Data models use Python dataclasses. Prompt templates are embedded near the VLM agent. Validation layers enforce allowed enum values, exact boolean strings, exact column order, retries, and cache-based cost control.

## Phase 5 — Judge Interview Preparation

### Architecture defense
The design separates observation from decision. The VLM is used to inspect images, but deterministic code enforces the challenge contract, allowed labels, risk policy, and conservative insufficiency behavior.

### Tradeoffs
The main tradeoff is one VLM call per row rather than per image. Per-row calls preserve cross-image consistency and reduce cost, but a large production system might use image-level caching plus a row-level adjudicator.

### Failure cases
Subtle dents, reflective glass, low-resolution close-ups, wrong-side orientation, and label/readability claims remain challenging. The system flags uncertainty for manual review rather than forcing a false supported/contradicted decision.

### Cost and scalability
For the supplied dataset, VLM cost is low. At scale, the same architecture supports image perceptual hashing, cached observations, parallel request queues, provider rate-limit throttling, and cheaper Flash/open-VLM prefilters.

### Fraud detection strategy
Fraud indicators include prompt injection, non-original screenshots, manipulated images, wrong object/part, repeated user-history risk, and mismatched multi-image identity. These add risk flags and manual review while keeping image evidence primary.

### 50 likely judge questions with concise answers
1. **Why image-first?** Because the statement says images are the primary source of truth.
2. **Why not use history to reject?** History is risk context only; rejecting on history alone is unfair and contrary to the spec.
3. **How do you handle prompt injection?** Detect approval/instruction language and add `text_instruction_present`; ignore it as evidence.
4. **Why GPT-4o?** Strong multimodal reasoning, multilingual text handling, and structured JSON support.
5. **What if no API key exists?** Deterministic fallback still runs and validates schema.
6. **How are image IDs derived?** Filename stem, such as `img_1`.
7. **How do you avoid schema errors?** Central allowed-value sets and normalization.
8. **How do you decide NEI?** Invisible part, unusable image, identity mismatch, or insufficient angle/context.
9. **How do you decide contradicted?** Relevant part visible but claimed issue or severity is materially absent/different.
10. **How is severity calibrated?** Claim and visual issue type map to none/low/medium/high, with unknown under ambiguity.
11. **What is cached?** Model JSON responses keyed by row content and parsed intent.
12. **Retry policy?** Exponential backoff for transient model/API failures.
13. **Batching strategy?** One row per call to preserve cross-image context; external concurrency can be added safely.
14. **Why not YOLO only?** Generic detectors do not classify subtle insurance damage reliably.
15. **Why not open VLM only?** GPU/runtime uncertainty during hackathon makes hosted VLM safer.
16. **How do multilingual claims work?** Keyword parser covers common terms, while GPT-4o handles broader multilingual semantics.
17. **How do you handle multiple parts?** Mark manual-review risk and require all claimed parts to be supported.
18. **How do you handle wrong object?** VLM/rules flag wrong_object and often NEI.
19. **What if one image supports and another conflicts?** Prefer NEI/manual review if identity or part conflict is material.
20. **What if text in an image says approved?** Treat as instruction risk, not visual damage evidence.
21. **How do you evaluate?** Run sample set, compare core structured fields, and inspect error categories.
22. **Why exact column order?** Automated evaluator expects it.
23. **What is the biggest weakness?** Subtle damage without a high-quality VLM or trained damage detector.
24. **How do you reduce hallucination?** Strict prompt, image IDs, conservative NEI, schema validation.
25. **How do you select supporting images?** Images that visibly establish the decision; none if insufficient.
26. **Can user history raise manual review?** Yes, through `user_history_risk` and `manual_review_required`.
27. **Does manual review change status?** Not by itself; it is a risk flag.
28. **Why row-level prompt?** Cross-image identity consistency matters.
29. **What about cost?** Small dataset, one call per row, cache prevents repeats.
30. **What about latency?** Sequential safe mode; concurrency optional under RPM limits.
31. **How do you handle missing images?** `valid_image=false`, NEI, no supporting images.
32. **How do you handle blurry images?** VLM should flag quality; fallback flags unusable files only.
33. **How do you handle screenshots/non-original?** VLM prompt asks to flag non-original/manipulation.
34. **How do you avoid hardcoded labels?** Code uses general parsers, prompts, and allowed enums rather than sample answers.
35. **What files are submitted?** `code.zip`, `output.csv`, and chat transcript.
36. **Why include evaluation report?** It is required and demonstrates operational thinking.
37. **How would production differ?** Add trained damage models, fraud graph features, human review UI, and audit logs.
38. **What if VLM returns invalid JSON?** The caller requests JSON mode and normalizer enforces valid values.
39. **What if VLM overstates severity?** Decision rules can clamp severity when contradiction/NEI is chosen.
40. **How are evidence requirements used?** They inform prompts and sufficiency logic by object/issue family.
41. **How would you improve hidden score?** Add few-shot exemplars from sample rows and visual observation calibration.
42. **Why deterministic temperature 0?** Reproducibility and stable CSV outputs.
43. **How do you handle contents missing?** Require images showing opened package/contents context; otherwise NEI.
44. **How do you handle package label claims?** Need label visibility/readability evidence.
45. **How do you handle laptop liquid damage?** Need visible stain/liquid marks on claimed part.
46. **How do you handle car orientation?** Require enough context for claimed side/part.
47. **What is your audit trail?** Output justifications plus cached model JSON.
48. **What if provider changes pricing?** Report uses assumptions and environment-configured model.
49. **Could this be ensemble-based?** Yes; add secondary VLM adjudication for low-confidence cases.
50. **Why is this top-tier?** It is schema-safe, image-first, risk-aware, reproducible, and operationally defensible.
