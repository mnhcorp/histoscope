from torch.utils.data import Dataset
import os
import torch
import json

class Kather100kEmbeddingDataset(Dataset):
    '''
    Dataset class that returns the pre-computed embeddings for the NCT-CRC-HE-100K dataset
    (also known as Kather100k dataset)
    '''
    def __init__(self,
                 emb_dir='cache-kather100k/train',
                 include_paths=False,
                 zscore: bool = True,                   # apply z-score standardization using emb_stats.pt
                 l2_normalize: bool = True,             # apply L2 normalization to embeddings
                 eps: float = 1e-6,
                 dtype: torch.dtype = torch.float32,
                 stats_path=None):
        self.emb_list = []
        self.lab_list = []
        self.paths_list = []
        self.include_paths = include_paths
        self.eps = eps
        self.dtype = dtype
        self.stats_path = stats_path or os.path.join(emb_dir, "emb_stats.pt")

        # Embeddings are named emb_0000.pt, emb_0001.pt, ...
        emb_files = sorted([os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.startswith('emb_') and f.endswith('.pt')])
        lab_files = sorted([os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.startswith('labels_') and f.endswith('.pt')])

        if include_paths:
            paths_files = sorted([os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.startswith('paths_') and f.endswith('.json')])

        for i, (emb_path, lab_path) in enumerate(zip(emb_files, lab_files)):
            emb = torch.load(emb_path) # [N, D]
            lab = torch.load(lab_path) # [N]
            self.emb_list.append(emb)
            self.lab_list.append(lab)

            if include_paths and i < len(paths_files):
                with open(paths_files[i], 'r') as f:
                    paths_data = json.load(f)
                # Extract just the patch_path from each dict
                for item in paths_data:
                    self.paths_list.append(item["patch_path"])

        self.emb_all = torch.cat(self.emb_list, dim=0) # [total_N, D]
        self.lab_all = torch.cat(self.lab_list, dim=0) # [total_N]
        # Keep a raw (pre-normalization) copy for analyses (e.g., MS without z-score)
        self.emb_all_raw = self.emb_all.clone()

        # ---- Z-score standardization ----
        if zscore:
            if os.path.exists(self.stats_path):
                # Load pre-computed stats
                stats = torch.load(self.stats_path, map_location="cpu")
                mean = stats["mean"].to(self.dtype)
                std = stats["std"].to(self.dtype)
                assert mean.shape[-1] == self.emb_all.shape[-1], "Stats dim mismatch"
            else:
                # Compute stats on-the-fly and save them
                print(f"emb_stats.pt not found at {self.stats_path}, computing stats on-the-fly...")
                mean = self.emb_all.mean(dim=0)
                std = self.emb_all.std(dim=0, unbiased=False)
                # Save computed stats for future use
                torch.save({"mean": mean.cpu(), "std": std.cpu()}, self.stats_path)
                print(f"Saved computed stats to {self.stats_path}")

            # Apply z-score normalization
            # ensure that each feature has zero mean and unit variance
            # so that each feature contributes equally and features with larger magnitudes don't dominate
            # We add eps to the denominator to avoid divide-by-zero errors
            self.emb_all = (self.emb_all - mean) / (std + eps)

        # ---- Optional row-wise L2 ----
        # Make sure each vector is unit length - get rid of the effect of vector magnitude
        if l2_normalize:
            norms = self.emb_all.norm(dim=1, keepdim=True).clamp_min(eps)
            self.emb_all = self.emb_all / norms

    def get_raw_embeddings(self):
        return self.emb_all_raw

    def __getitem__(self, index):
        if self.include_paths and self.paths_list and index < len(self.paths_list):
            return self.emb_all[index], self.lab_all[index], self.paths_list[index]
        else:
            return self.emb_all[index], self.lab_all[index]

    def __len__(self):
        return self.emb_all.shape[0]

    def get_patch_path(self, index):
        """Get the full patch path for a specific index"""
        if self.paths_list and index < len(self.paths_list):
            return self.paths_list[index]
        return None

    @staticmethod
    def get_class_names():
        """Return the class names for the NCT-CRC-HE-100K dataset"""
        return {
            0: "ADI - Adipose",
            1: "BACK - Background",
            2: "DEB - Debris",
            3: "LYM - Lymphocytes",
            4: "MUC - Mucus",
            5: "MUS - Smooth muscle",
            6: "NORM - Normal colon mucosa",
            7: "STR - Cancer-associated stroma",
            8: "TUM - Colorectal adenocarcinoma epithelium"
        }

    @staticmethod
    def get_num_classes():
        """Return the number of classes in the dataset"""
        return 9
