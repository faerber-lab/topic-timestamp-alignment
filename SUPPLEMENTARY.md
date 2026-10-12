# Supplementary Material

Supplementary material for *Topic-to-Timestamp Alignment by Constrained Evidence Selection*. It contains the dataset construction details, model and prompt configuration, offline retrieval evaluation, error analysis, and additional implementation details referenced in the paper.

## Contents

1. [Dataset Construction](#1-dataset-construction)
2. [Model and Prompt Configuration](#2-model-and-prompt-configuration)
3. [Offline Retrieval Evaluation](#3-offline-retrieval-evaluation)
4. [Error Analysis](#4-error-analysis)
5. [Additional Implementation Details](#5-additional-implementation-details)

---

## 1. Dataset Construction

### Transcript construction

We collected 200 public municipal meeting recordings. Audio was converted to 16 kHz mono using FFmpeg. We used Whisper-medium for ASR and `pyannote/speaker-diarization` for speaker segmentation. Word-level ASR outputs were assigned to speaker segments by temporal overlap. The resulting transcripts contain timestamped speaker turns, e.g.:

```
[00:15:32] SPEAKER_01: We now discuss the budget amendment...
```

### Annotation

The candidate set contains 600 topic–timestamp pairs. Each instance consists of a source transcript, a topic heading, and a gold timestamp marking the start of the corresponding discussion. When public video descriptions provided timestamps, these were used as initial boundaries. Missing boundaries were identified manually from the video and transcript. The dataset was curated by a single annotator; we therefore do not report inter-annotator agreement.

### Procedural filtering

The full candidate set contains repetitive procedural headings, including call-to-order, roll call, pledge, invocation, opening remarks, and adjournment. We remove these categories before evaluation, yielding the 420-query benchmark used in the paper. The filtering is based only on heading categories.

**Table S1.** Full candidate set and filtered benchmark using the same Mistral-7B-Instruct pipeline.

| Set | Queries | Recall@5 | Exact@30s | MAE ↓ | Parsed |
|---|---:|---:|---:|---:|---:|
| Full candidate set | 600 | **56.3%** | **37.3%** | **632.0s** | 600/600 |
| Filtered benchmark | 420 | 50.0% | 30.0% | 761.0s | 419/420 |

---

## 2. Model and Prompt Configuration

### Models

The open-source model is `mistral:7b-instruct`, served through Ollama. The GPT model identifiers are:

| Name in paper | API model identifier |
|---|---|
| GPT-5.1 | `gpt-5.1-2025-11-13` |
| GPT-5.2 | `gpt-5.2-2025-12-11` |
| GPT-5.4 | `gpt-5.4-2026-03-05` |

All GPT models were accessed in May 2026. Decoding settings and parsing rules are included in the evaluation configuration.

### Timestamp-generation prompt (baseline)

```
Task: Given the topic and transcript excerpts, identify when the topic is discussed.

Topic: {topic}

Transcript excerpts: {retrieved_context}

Return only one timestamp in HH:MM:SS format.
```

### Candidate-ID prompt (proposed)

```
Task: Select the candidate utterance that best marks the start of the topic discussion.

Topic: {topic}

Candidates (ID | timestamp | excerpt):
C01 | {utterance_timestamp} | {utterance_text}
C02 | {utterance_timestamp} | {utterance_text}
...

Rules: Output ONLY the Candidate-ID, e.g., C03. Do NOT output a timestamp.
```

### Parsing

- **Timestamp generation:** we parse `HH:MM:SS`, `MM:SS`, and unambiguous natural-language times.
- **Candidate-ID selection:** we parse the first valid Candidate-ID.
- Empty or unparseable outputs count as unparsed.

---

## 3. Offline Retrieval Evaluation

For completeness, we report Precision@5 and F1@5 for the retrieval stage, evaluated independently of the language model. A retrieved window counts as relevant if it contains the gold timestamp; Precision@5 is the mean fraction of the top five windows that are relevant.

**Table S2.** Offline retrieval performance.

| Retrieval setting | Precision@5 | Recall@5 | F1@5 |
|---|---:|---:|---:|
| Dense + 30s/15s windows | 9.1% | 31.9% | 14.2% |
| Dense + 90s/45s windows | 11.3% | 41.9% | 17.8% |
| Hybrid RRF + 30s/15s windows | 11.7% | 39.0% | 18.0% |
| Hybrid RRF + 90s/45s windows | **15.3%** | **50.0%** | **23.5%** |

---

## 4. Error Analysis

### Failure categories

We group errors into three categories:

1. the relevant segment is not retrieved;
2. the selected utterance is topically related but temporally offset;
3. the output is invalid or unparseable.

The proposed system reduces invalid outputs and large outlier errors, but does not remove boundary-offset errors.

**Table S3.** Absolute-error (AE) statistics, end-to-end comparison (Standard RAG vs. proposed pipeline).

| System | Queries | Median AE | P95 AE |
|---|---:|---:|---:|
| Mistral baseline (own parsed subset) | 373 | 150.0s | 3962.2s |
| Mistral proposed (own parsed subset) | 419 | 204.0s | 3313.9s |
| GPT-5.4 baseline (own parsed subset) | 406 | 284.0s | 4966.0s |
| GPT-5.4 proposed (own parsed subset) | 420 | 191.0s | 3173.4s |
| Mistral baseline / proposed (parsed by both) | 372 | 151.0s / 177.5s | 3965.6s / 3598.7s |
| GPT-5.4 baseline / proposed (parsed by both) | 406 | 284.0s / 180.5s | 4966.0s / 3263.0s |

### Qualitative error patterns

We manually examined representative grounding failures in which the gold timestamp was present among the retrieved candidates, but the model selected a candidate offset from the annotated topic start. These were typically boundary errors within or near the correct agenda item rather than mismatches to unrelated topics. For procedural headings such as *Minutes Approval and Agenda Adoption*, the model selected later utterances announcing or confirming agenda adoption, whereas for *State Street Master Plan Update* it selected an early transition cue preceding the annotated start. These examples help explain the gap between Recall@5 and Exact@30s: although retrieval often identifies the correct discussion region, localizing its temporal boundary accurately remains challenging.

**Table S4.** Illustrative grounding errors where the gold timestamp was present in the retrieved candidates, but the selected candidate was offset from the annotated topic start.

| Source | Topic heading | Gold | Pred. | Error pattern |
|---|---|---|---|---|
| Vermillion City Council | Minutes Approval and Agenda Adoption | 00:02:08 | 00:03:32 | Selected a later utterance: "Adoption of the agenda." |
| 2022-11-21 City Council | Adoption of Agenda and Minutes Approval | 00:01:28 | 00:02:32 | Selected the adoption decision rather than the item start. |
| Access Advisory Committee | State Street Master Plan Update | 00:24:25 | 00:23:33 | Selected an early transition cue before the topic start. |

---

## 5. Additional Implementation Details

In the reported hybrid setting, candidates are retrieved with dense retrieval (`BAAI/bge-large-en-v1.5`) and BM25 using multiple query variants, fused with Reciprocal Rank Fusion (k = 60), and lightly re-ranked using heading-term overlap and an early-section temporal prior when applicable. These steps use only the query text and the information retrieved from the candidates, never the gold topic timestamps or the model outputs.

For the reported Candidate-ID experiments, deterministic fallback extraction is disabled: the model selects a Candidate-ID, which is then resolved to the timestamp of the selected utterance.