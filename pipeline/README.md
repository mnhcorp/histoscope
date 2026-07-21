# Training and artifact pipeline

Histoscope is the visualization layer. It reads analysis artifacts generated from an SAE checkpoint; it does not train or analyze the SAE itself.

The release pipeline has three stages:

```text
cached UNI embeddings
        |
        v
simple_sae_spider.py          train the TopK SAE
        |
        v
sae_feature_analysis_v2.py    score features and write interactive caches
        |
        v
histoscope.py                 inspect the generated artifacts
```

`run_pipeline.py` joins the first two stages and prints the command for the third.

## From scratch

The embedding cache must contain `train/`, `test/`, and `label_map.json`. Each split contains matching `emb_*.pt`, `labels_*.pt`, and `paths_*.json` shards. Training statistics are stored in the training split and reused for test normalization.

```bash
python pipeline/run_pipeline.py \
  --cache-root /path/to/cache-spider-colorectal-uni \
  --models-dir /path/to/sae-models \
  --cluster
```

## From the released checkpoint

Skip training and generate the dashboard artifacts from an existing checkpoint bundle:

```bash
python pipeline/run_pipeline.py \
  --cache-root /path/to/cache-spider-colorectal-uni \
  --models-dir /path/to/sae-models \
  --checkpoint /path/to/model-exp49152-l2-zscore-tied-prebias-acttopk250-bs32-lr0.0001 \
  --cluster
```

The source scripts are retained here to preserve the exact analysis path used for the paper. Generated feature matrices, patch activations, image paths, and checkpoints are deliberately ignored by Git.
