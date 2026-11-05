# `</think>` row gain x1.3 stacked on the hedge-damping leader

Author: vdimarco. Agent-assisted (Claude), human-reviewed. Arm 2 of the `</think>` gain line;
arm 1 is `research/vdimarco-think-gain/` (PR #16).

## What arm 1 showed

Gain x1.5 on the parent cut truncation from 0.746 to 0.440 and output tokens by 42%, with the
private test flat (17.6 vs 18.6, SE 1.7). The mechanism works; converting capped generations
into answers did not add correct answers on the parent, because the trace that filled the cap
was rumination, not unfinished reasoning.

## Why stack

The board leader `ab592312` (source-free, private test 28.2, truncation 0.526) attacks the
rumination directly. Diffing its checkpoint against the parent (`diff_vs_parent.py`,
`leader_diff.json`):

- 11 embedding rows shrunk to roughly 0.6 of their norm (cosine 0.95-0.96 to the original):
  `Wait`, ` Wait`, ` wait`, `Hmm`, ` Hmm`, `Maybe`, ` Maybe`, `Actually`, ` Actually`,
  `Alternatively`, ` Alternatively`.
- Final RMSNorm gain `(1+w)` scaled by 1.40 at every coordinate (std of the ratio 0.002),
  which multiplies every logit by 1.4.

Hedge damping and the closing gain act on different moments of the trace: one lowers the odds
of starting another "Wait" loop, the other raises the odds of closing once the model has the
answer. Hypothesis: on a model that ruminates less, the generations that still hit the cap are
more often unfinished reasoning with an answer in reach, so closing them earlier now converts
to correct answers.

## Edit

Multiply `E[</think>]` (row 248069 of `model.language_model.embed_tokens.weight`) by 1.3 on
the leader checkpoint. The norm gain already scales that logit by 1.4, so 1.3 here gives a
closing push of about `(1.3-1) * 1.4 * 16 = 6.7` nats at a typical pre-close logit, close to
the `0.5 * 16 = 8` nats arm 1 applied on the parent. No other logit changes. 1,024 bf16 entries.

## Pre-registered predictions (private test, `public-ladder500-8k-v2`)

Leader: validation 41.02, private 28.2 (SE about 1.7), truncation 0.526.

1. Truncation falls below 0.40.
2. Private test at or above 28.2.

Falsifier: private test below 24.8 (two standard errors under the leader). If 1 holds and 2
fails within noise, the closing push is not where the remaining headroom is and this line stops.

## Lineage and reproduction

- Parent version `ab592312-ec77-4f98-b17f-6ea17723b74e` (its own parent: original
  `0ea62f40`). Evaluation `public-ladder500-8k-v2`.
- Patch `sparse-diff`, indices 254022656..254023679, `weights.safetensors` sha256
  `f75033ee1342296baeef5eb4c684ee0366f174a0c3cbb1dd05efb0626b742260`, 10,392 bytes.
- Edited checkpoint sha256 `330e10f4b7afb00cbc7046afb89efa5214b9c83c64f220c6ea45f6486c0dce62`.

```
delta competition init qwen35-08b-v1 --parent ab592312-ec77-4f98-b17f-6ea17723b74e --output exp-leader
cd exp-leader
python edit_think_gain.py model 1.3 edited.safetensors
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
```

No local runs for this arm; the edit is the same deterministic row scale verified in arm 1,
and the leader checkpoint was only diffed, not executed.

## Access during research

Public checkpoints of the parent and the leader via the CLI, leaderboard API, PRs #7, #12,
#16. No private test data. The leader's edit was inferred from its weights; its author did not
publish a record.

## Submission

Pending.
