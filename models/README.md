# Model files

The paper checkpoint is hosted in the [Histoscope UNI/SPIDER TopK SAE model repository](https://huggingface.co/mnhcorp/histoscope-uni-spider-sae). Generated analysis caches are not committed to Git or bundled with the checkpoint.

## Upstream access

Complete reproduction requires separate access to:

- [SPIDER-colorectal](https://huggingface.co/datasets/histai/SPIDER-colorectal), which supplies the image patches; and
- [UNI](https://huggingface.co/MahmoodLab/UNI), which supplies the frozen image encoder.

Both repositories are gated and remain subject to their own non-commercial terms. Histoscope does not redistribute either resource.

## Download the paper checkpoint

From the Histoscope repository root:

```bash
hf download mnhcorp/histoscope-uni-spider-sae \
  --local-dir sae-models/uni/spider/model-exp49152-l2-zscore-tied-prebias-acttopk250-bs32-lr0.0001

cd sae-models/uni/spider/model-exp49152-l2-zscore-tied-prebias-acttopk250-bs32-lr0.0001
sha256sum -c SHA256SUMS
```

Expected checkpoint SHA-256:

```text
fc6e46464c569423162d049792649b059fec545ed9eed8219e81b1ad68790ce6  model.pt
```

The checkpoint is a PyTorch state dictionary containing `encoder.weight` with shape `49152 x 1024` and `b_pre` with shape `1024`. The tied decoder uses `encoder.weight.T`.

## Generate the dashboard cache

The checkpoint alone is sufficient to reproduce SAE inference, but Histoscope also needs patch paths and analysis caches. Build them from a local cache of precomputed UNI embeddings:

```bash
python pipeline/run_pipeline.py \
  --cache-root /path/to/cache-spider-colorectal-uni \
  --models-dir "$PWD/sae-models" \
  --checkpoint "$PWD/sae-models/uni/spider/model-exp49152-l2-zscore-tied-prebias-acttopk250-bs32-lr0.0001" \
  --cluster
```

The embedding-cache contract is documented in [`../pipeline/README.md`](../pipeline/README.md). Paths written to `image_paths.json` must resolve to the separately downloaded SPIDER patches on the machine running Histoscope.

## Expected layout

```text
sae-models/
└── uni/
    └── spider/
        └── model-exp49152-l2-zscore-tied-prebias-acttopk250-bs32-lr0.0001/
            ├── model.pt
            ├── metadata.json
            └── analysis/
                ├── interactive-cache/
                │   ├── train/
                │   │   ├── cache.npz
                │   │   └── image_paths.json
                │   └── test/
                │       ├── cache.npz
                │       └── image_paths.json
                └── patch-activations/
```

The exact configuration is recorded in [`../configs/uni_spider_topk250.json`](../configs/uni_spider_topk250.json).
