# Evaluation protocol

This document records the protocol used for the Histoscope expert study. It is provided to support replication of the study design; the original feature-level ratings and raw rater responses are not part of this release.

## Feature scoring

For each SAE feature and each tissue class, we compute one-vs-rest AUPRC separately on the training and test splits. The class score is the lower of the two split-specific AUPRC values. The highest class score defines the dominant class, and the gap is the difference between the highest and second-highest class scores.

A feature must also exceed its feature-wise 95th-percentile training-activation threshold with sufficient class recall on both splits. For a class with prevalence `p_c`, the recall floor is:

```text
max(0.01, min(0.045, p_c))
```

The prespecified labels were:

| Label | Criteria |
|---|---|
| Monosemantic | AUPRC >= 0.90, top-vs-second gap >= 0.20, and recall floor passed on both splits |
| Nearly monosemantic | 0.65 <= AUPRC < 0.90, with the same gap and recall criteria |
| Polysemantic | All remaining features |

For study sampling, monosemantic and nearly monosemantic features were combined. The 50 highest-ranked features from this combined group and the 50 lowest-ranked polysemantic features formed a confidence-stratified sample. This was not a representative sample of the full SAE dictionary.

## Expert review

Two pathologists independently reviewed each feature as a panel of its 12 highest-activating test patches. Raters were blinded to the AUPRC-derived label and score. They selected one or more morphology terms from the controlled vocabulary and rated:

1. consistency of the recurring concept, from 1 to 5;
2. diagnostic relevance, from 1 to 5; and
3. optional free-text notes.

The primary expert endpoint classified a feature as monosemantic when both raters assigned consistency >= 4. A stricter secondary endpoint additionally required descriptor Jaccard similarity >= 0.5.

See [`RATER_INSTRUCTIONS.md`](RATER_INSTRUCTIONS.md) for the rater-facing wording and [`controlled_vocabulary.json`](controlled_vocabulary.json) for the released vocabulary.

## Preprocessing note

The post-hoc export used for the original expert panels normalized test embeddings with test-split statistics. This did not affect SAE training or feature identity. A corrected audit using training-split statistics found mean embedding cosine similarity of 0.9984 and 97.6% mean top-12 panel overlap; no panel changed dominant tissue class and no analyzed feature changed AUPRC group. The corrected train-statistics preprocessing is the released implementation.
