# Evaluation Report

This project includes an eight-question StarVault benchmark designed to cover factual lookup, list aggregation, numeric calculation, implicit inference, contradiction detection, and unanswerable boundary behavior.

## Benchmark Design

Test document:

- `test_data/starvault_report.md`
- Internal Q2 project report with project descriptions, team table, budget table, and timeline table

Question groups:

| ID | Type | Main Capability |
| --- | --- | --- |
| Q1 | Single fact | Retrieve exact budget and spending |
| Q2 | Single fact | Retrieve technical stack and reason |
| Q3 | List aggregation | List all projects and departments |
| Q4 | Numeric aggregation | Calculate total budget, total spending, and max remaining budget |
| Q5 | Implicit inference | Infer likely support candidates from skills and project state |
| Q6 | Strategic inference | Connect revenue sources and strategic constraints |
| Q7 | Contradiction detection | Identify inconsistent project timeline/status |
| Q8 | Boundary | Refuse to invent missing financing / valuation information |

## Current Routing Behavior

| ID | Adaptive Route | Expected |
| --- | --- | --- |
| Q1 | medium / single_step | Correct |
| Q2 | medium / single_step | Correct |
| Q3 | medium / single_step | Correct, list top_k=5 |
| Q4 | medium / single_step | Correct, deterministic table calculation available |
| Q5 | complex / multi_step | Correct |
| Q6 | complex / multi_step | Correct |
| Q7 | complex / multi_step | Correct |
| Q8 | complex / multi_step | Correct |

## Why Q5 and Q8 Can Score Poorly in RAGAS

### Q5: Implicit Inference

Q5 asks who is likely to be reassigned if the TianShu project urgently needs people before release. The answer is not explicitly written in the document. A good answer must combine:

- team table
- current project ownership
- technical skills
- project timeline and risk

Default RAGAS faithfulness prefers claims that are directly stated in context. A reasonable inference such as "Ma Xiaojun is a possible support candidate because he is full-stack and has React experience" may be judged unsupported because the document never explicitly states that he will be reassigned.

For this reason, Q5 should be evaluated with an inference rubric, not only default RAGAS.

Recommended checks:

- Team table chunk is retrieved.
- Project status/timeline evidence is retrieved.
- Answer provides 2-3 candidates.
- Answer labels the conclusion as inference rather than direct document fact.
- Candidate reasons cite skills, current project, and likely support scenario.

### Q8: Unanswerable Boundary

Q8 asks about financing plan and valuation. The correct behavior is to say the document does not mention this information.

Default RAGAS context precision and answer relevancy can under-score this because the correct answer is based on absence of evidence. The system should be rewarded for not inventing financing facts.

Recommended checks:

- Answer explicitly says the document does not mention financing or valuation.
- Answer does not invent B-round/A-round details, valuation, investors, or financing amount.
- Retrieved contexts are allowed to be generally relevant company context rather than exact answer-bearing snippets.

## Recommended Evaluation Strategy

Use RAGAS as one signal, but combine it with task-specific validators:

- Numeric validator for Q4-style aggregate questions
- Abstention validator for Q8-style unanswerable questions
- Evidence coverage validator for Q5-style implicit inference
- Manual rubric for small benchmark reports

This is more reliable than optimizing only for RAGAS scores.

