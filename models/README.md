# Model files

The trained SAE checkpoint and its interactive-cache bundle are large release artifacts and are not committed to Git.

Expected layout after downloading and extracting the model bundle:

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

The public download location will be added here when the release artifact has been uploaded. Its configuration is recorded in [`../configs/uni_spider_topk250.json`](../configs/uni_spider_topk250.json).

Patch paths in `image_paths.json` must resolve on the machine running Histoscope. The SPIDER images and UNI weights are governed by their respective upstream terms and are not redistributed here.
