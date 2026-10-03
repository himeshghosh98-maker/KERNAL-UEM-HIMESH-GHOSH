# GapDetect Evaluation Results

Config: min_severity=0.4, unanswered_window=12 msgs / 180 min, clar_sim=0.45, topic_shared_tokens=1

A prediction counts as a true positive if its message ids overlap a labeled gap of the same type.

## college_project_team (53 messages)

| Gap Type | Gold | Pred | TP | FP | FN | Precision | Recall | F1 |
|----------|-----:|-----:|---:|---:|---:|----------:|-------:|---:|
| clarification | 2 | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ignored | 1 | 3 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| unanswered | 4 | 5 | 2 | 3 | 2 | 0.40 | 0.50 | 0.44 |
| unresolved | 1 | 2 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |

## event_planning (44 messages)

| Gap Type | Gold | Pred | TP | FP | FN | Precision | Recall | F1 |
|----------|-----:|-----:|---:|---:|---:|----------:|-------:|---:|
| clarification | 2 | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ignored | 0 | 1 | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| unanswered | 3 | 7 | 3 | 4 | 0 | 0.43 | 1.00 | 0.60 |
| unresolved | 1 | 3 | 2 | 1 | 0 | 0.67 | 1.00 | 0.80 |

## study_group (53 messages)

| Gap Type | Gold | Pred | TP | FP | FN | Precision | Recall | F1 |
|----------|-----:|-----:|---:|---:|---:|----------:|-------:|---:|
| unanswered | 4 | 9 | 3 | 6 | 1 | 0.33 | 0.75 | 0.46 |
| unresolved | 2 | 5 | 3 | 2 | 0 | 0.60 | 1.00 | 0.75 |

## Aggregate by gap type (all chats)

| Gap Type | Gold | Pred | TP | FP | FN | Precision | Recall | F1 |
|----------|-----:|-----:|---:|---:|---:|----------:|-------:|---:|
| clarification | 4 | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ignored | 1 | 4 | 1 | 3 | 0 | 0.25 | 1.00 | 0.40 |
| unanswered | 11 | 21 | 8 | 13 | 3 | 0.38 | 0.73 | 0.50 |
| unresolved | 4 | 10 | 6 | 4 | 0 | 0.60 | 1.00 | 0.75 |

**Micro overall:** precision=0.49 recall=0.86 F1=0.62

_Thresholds tuned once after this run — see README Results section for before/after._