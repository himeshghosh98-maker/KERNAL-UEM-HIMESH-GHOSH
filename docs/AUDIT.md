# Phase 0 — Audit (read-only)

Date: 2026-10-03. Workspace: `C:\Users\Himesh\Desktop\Cerebro`.

## Workspace contents

| Path | Notes |
|------|-------|
| `.freebuff/project-id` | Hackathon project id only. |
| `gapdetect/` | Prior partial scaffold (different architecture than the required spec). Rebuilt to spec. |

**Important:** The three reference repos named in the brief (`groupchat-decoder`, `parshift`, `Agreement-and-Disagreement-Recognition`) are **NOT present in the workspace**. They were audited via the GitHub API instead. Per the rules, the project **must work without any repo** — the entire pipeline is implemented natively in Python.

## Repo-by-repo audit

### 1. groupchat-decoder
- **Found:** https://github.com/tanishamalik0208/groupchat-decoder (repo search match).
- **Language:** JavaScript.
- **License:** None declared (`license: null`).
- **Idea:** AI-powered group chat intelligence — summaries, action items, unanswered questions, collaboration insights.
- **Install/run:** Not applicable — JS app (Vercel homepage), not present in workspace, no license to copy from.
- **Input/output:** Unknown (JS front-end; no stable Python API).
- **Dependency conflicts:** N/A for our Python stack.
- **Decision: SKIP import/subprocess.** Borrow the *logic concept* only: unanswered questions / unassigned requests, reimplemented natively in `core/detectors/unanswered.py`.

### 2. parshift
- **Found:** https://github.com/bdfsaraiva/parshift.
- **Language:** Python package.
- **License:** MIT (spdx `MIT`).
- **Idea:** Gibson (2003) participation-shift framework for turn-taking in group conversation (Speaker S / Addressee A coding, AB-BA reciprocity, turn usurpation).
- **Install/run:** `pip install parshift` (homepage: https://bdfsaraiva.github.io/parshift/). Not installed in the project venv to avoid extra deps; optional import attempted at runtime.
- **Input/output:** Turn sequences; participation-shift counts.
- **Dependency conflicts:** Low risk; guarded optional import.
- **Decision: Borrow logic + optional adapter.** Native lightweight S/A coding lives in `core/adapters.py` and feeds the `ignored` detector; if `parshift` is importable it can be wrapped, failures fall back safely.

### 3. Agreement-and-Disagreement-Recognition (ADR)
- **Found:** No public GitHub repository matched this exact name (search returned 0 results).
- **License:** Unknown / not available.
- **Idea (from literature):** classify agreement vs disagreement / pushback stances; flag unresolved disagreements.
- **Decision: SKIP (cannot integrate within 20 min).** Borrow the *stance-cue concept* only — disagreement / hesitation cues inform `core/detectors/clarification.py` and `core/detectors/unresolved.py`.

## Final integration decisions

| Repo | Decision | Where it shows up |
|------|----------|-------------------|
| groupchat-decoder | SKIP import; borrow logic | `core/detectors/unanswered.py` |
| parshift | Borrow logic; optional import adapter | `core/adapters.py`, `core/detectors/ignored.py` |
| Agreement-and-Disagreement-Recognition | SKIP; borrow stance cues | cue lists in clarification/unresolved detectors |

**Guarantee:** `gapdetect` runs end-to-end with zero reference repos installed. Adapters are try/except-guarded; LLM (Gemini) and embeddings are optional flags and default OFF.
