import os, sys
import json
from datetime import datetime
import argparse

from analysis_utils import *

def compute_purity_at_k(feature_matrix, labels, feature_index, k=100):
    """Compute purity@k for a single feature: fraction of majority class among top-k activations.
    Falls back to available samples if fewer than k. Safe for list/np/tensor labels.
    """
    if not isinstance(feature_matrix, torch.Tensor):
        raise ValueError("feature_matrix must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        labels_tensor = labels.to(dtype=torch.long)

    feats = feature_matrix[:, feature_index]
    if feats.numel() == 0:
        return 0.0
    k_eff = min(k, feats.shape[0])
    if k_eff == 0:
        return 0.0

    # Use absolute values to find strongest activations regardless of sign
    _, top_idx = torch.topk(torch.abs(feats), k=k_eff, largest=True, sorted=True)

    top_labels = labels_tensor[top_idx]
    if top_labels.numel() == 0:
        return 0.0
    bincount = torch.bincount(top_labels)
    if bincount.numel() == 0:
        return 0.0
    majority = int(bincount.max().item())
    return float(majority) / float(k_eff)

def generate_baseline_cache(args):
    """Generate baseline interactive cache using original UNI or Hibou embeddings (1024 or hibou features)"""
    from analysis_utils import (
        create_datasets_from_metadata, extract_baseline_features,
        generate_baseline_interactive_cache, calculate_baseline_monosemanticity_scores
    )

    # Determine embedding directory based on model choice and dataset type
    dataset_type = getattr(args, 'dataset', 'spider')

    # Determine cache root based on dataset type
    if dataset_type == 'ignite':
        cache_root = "cache-ignite"
    elif dataset_type == 'kather100k':
        cache_root = "cache-nctcrche100k"
    elif dataset_type == 'spider-thorax':
        cache_root = "cache-spider-thorax"
    elif dataset_type == 'spider-skin':
        cache_root = "cache-spider-skin"
    else:
        cache_root = "cache"

    if getattr(args, 'use_hibou', False):
        if dataset_type == 'ignite':
            emb_dir_train = "cache-ignite-hibou/train"
            emb_dir_test = "cache-ignite-hibou/test"
        else:
            if dataset_type == 'spider-thorax':
                emb_dir_train = "cache-spider-thorax/hibou/train"
                emb_dir_test = "cache-spider-thorax/hibou/test"
            else:
                emb_dir_train = "cache-hibou/train"
                emb_dir_test = "cache-hibou/test"
        model_type = "hibou"
        print(f"[baseline] Using Hibou embeddings for baseline analysis ({dataset_type} dataset)")
    else:
        if dataset_type == 'ignite':
            emb_dir_train = "cache-ignite/train"
            emb_dir_test = "cache-ignite/test"
        elif dataset_type == 'kather100k':
            # Kather embeddings stored under cache-nctcrche100k/embeddings_uni_{split}
            emb_dir_train = os.path.join("cache-nctcrche100k", "embeddings_uni_train")
            emb_dir_test = os.path.join("cache-nctcrche100k", "embeddings_uni_validation")
        elif dataset_type == 'spider-thorax':
            emb_dir_train = "cache-spider-thorax/train"
            emb_dir_test = "cache-spider-thorax/test"
        else:
            emb_dir_train = "cache/train"
            emb_dir_test = "cache/test"
        model_type = "uni"
        print(f"[baseline] Using UNI embeddings for baseline analysis ({dataset_type} dataset)")

    # Set up baseline directory with dataset-specific path
    baseline_dir = os.path.join("sae-models", dataset_type, f"baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    analysis_base_dir = os.path.join(baseline_dir, "analysis")
    os.makedirs(analysis_base_dir, exist_ok=True)

    # Determine which artifacts already exist (for incremental generation)
    force_regen = getattr(args, 'force_regenerate', False)
    interactive_cache_root = os.path.join(analysis_base_dir, "interactive-cache")
    comparison_file_path = os.path.join(analysis_base_dir, "baseline_feature_comparison.json")
    topk_root = os.path.join(analysis_base_dir, "top-k-features")
    topk_meta_path = os.path.join(topk_root, "baseline_feature_analysis.json")

    has_interactive_cache = (
        os.path.isdir(os.path.join(interactive_cache_root, "train")) and
        os.path.isdir(os.path.join(interactive_cache_root, "test"))
    )
    has_comparison = os.path.isfile(comparison_file_path)
    has_topk = os.path.isfile(topk_meta_path)

    if force_regen:
        print("[baseline] --force-regenerate specified: all baseline artifacts will be rebuilt.")
    else:
        print("[baseline] Incremental generation summary (delete a subfolder to rebuild just that part):")
        print(f"  - interactive-cache: {'present' if has_interactive_cache else 'MISSING -> will build'}")
        print(f"  - baseline_feature_comparison.json: {'present' if has_comparison else 'MISSING -> will build'}")
        print(f"  - top-k-features (baseline_feature_analysis.json): {'present' if has_topk else 'MISSING -> will build'}")
        # Note: prior behavior always recomputed monosemanticity scores even when only top-k-features was missing;
        # we now attempt to reuse scores from existing comparison JSON to avoid the expensive pass.

    # Use the pre-computed embeddings in cache/train and cache/test (or hibou variants)
    metadata = {
        "dataset_type": dataset_type,
        "dataset_args": {
            "emb_dir": emb_dir_train,
            "zscore": True,
            "l2_normalize": True
        }
    }

    print("Creating datasets from metadata...")
    train_dataset, test_dataset = create_datasets_from_metadata(metadata)
    # Update test dataset to use correct directory
    test_dataset.emb_list = []
    test_dataset.lab_list = []
    test_dataset.paths_list = []

    # Reload test dataset with correct directory
    emb_files = sorted([os.path.join(emb_dir_test, f) for f in os.listdir(emb_dir_test) if f.startswith('emb_') and f.endswith('.pt')])
    lab_files = sorted([os.path.join(emb_dir_test, f) for f in os.listdir(emb_dir_test) if f.startswith('labels_') and f.endswith('.pt')])
    paths_files = sorted([os.path.join(emb_dir_test, f) for f in os.listdir(emb_dir_test) if f.startswith('paths_') and f.endswith('.json')])

    for emb_path, lab_path in zip(emb_files, lab_files):
        emb = torch.load(emb_path)
        lab = torch.load(lab_path)
        test_dataset.emb_list.append(emb)
        test_dataset.lab_list.append(lab)

    # Load paths for test dataset
    import json
    for paths_path in paths_files:
        with open(paths_path, 'r') as f:
            paths_data = json.load(f)
        # Extract just the patch_path from each dict
        for item in paths_data:
            test_dataset.paths_list.append(item["patch_path"])

    test_dataset.emb_all = torch.cat(test_dataset.emb_list, dim=0)
    test_dataset.lab_all = torch.cat(test_dataset.lab_list, dim=0)
    test_dataset.emb_all_raw = test_dataset.emb_all.clone()

    # Apply same normalization as train
    if test_dataset.emb_all.shape[0] > 0:
        stats_path = os.path.join(emb_dir_train, "emb_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
            mean = stats["mean"].to(test_dataset.dtype)
            std = stats["std"].to(test_dataset.dtype)
            test_dataset.emb_all = (test_dataset.emb_all - mean) / (std + test_dataset.eps)
        norms = test_dataset.emb_all.norm(dim=1, keepdim=True).clamp_min(test_dataset.eps)
        test_dataset.emb_all = test_dataset.emb_all / norms

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=32)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    print(f"Extracting baseline features (original {model_type.upper()} embeddings)...")
    # Extract original embeddings (1024-dim for UNI, varies for Hibou) instead of SAE features
    feature_matrix_train, image_paths_train, labels_train = extract_baseline_features(train_dataloader)
    feature_matrix_test, image_paths_test, labels_test = extract_baseline_features(test_dataloader)

    print(f"Train baseline feature matrix shape: {feature_matrix_train.shape}")
    print(f"Test baseline feature matrix shape: {feature_matrix_test.shape}")

    # All 1024 features (no selection needed)
    all_feature_indices = list(range(feature_matrix_train.shape[1]))  # [0, 1, 2, ..., 1023]

    # Decide whether to recompute monosemanticity scores or reuse existing
    reuse_mono = (not force_regen) and has_comparison
    mono_scores = {}
    if reuse_mono:
        try:
            with open(comparison_file_path, 'r') as f:
                existing_comp = json.load(f)
            print("Reusing monosemanticity scores from existing baseline_feature_comparison.json...")
            for feat in existing_comp.get("feature_comparisons", []):
                fid = feat["feature_index"]
                mono_scores[fid] = {
                    'M': feat.get('monosemanticity_score', 0.0),
                    'm_train': feat.get('train_margin', 0.0),
                    'm_test': feat.get('test_margin', 0.0),
                    'c_star_train_name': feat.get('dominant_class_train', None),
                    'c_star_test_name': feat.get('dominant_class_test', None),
                    'flipped': feat.get('dominant_class_flipped', False)
                }
            missing = [fid for fid in all_feature_indices if fid not in mono_scores]
            if missing:
                print(f"Found existing comparison but missing {len(missing)} feature scores; recomputing all.")
                reuse_mono = False
        except Exception as e:
            print(f"Failed to reuse existing monosemanticity scores ({e}); recomputing...")
            reuse_mono = False

    if not reuse_mono:
        print("Calculating baseline monosemanticity scores...")
        mono_scores = calculate_baseline_monosemanticity_scores(
            feature_matrix_train, labels_train,
            feature_matrix_test, labels_test,
            all_feature_indices,
            cache_root=cache_root
        )

    # --- NEW: Compute MS_score (baseline monosemanticity via pairwise similarity weighting) ---
    # Use train split for MS_score (can extend to test later if desired)
    algorithm = "slow O(N²)" if getattr(args, 'ms_slow', False) else "fast O(N*D)"
    ms_variant = "ratio-form" if getattr(args, 'ms_ratio', False) else "density-form"
    try:
        print(f"Computing MS_score for baseline features on TRAIN split using {algorithm} algorithm ({ms_variant})...")

        # Determine which embeddings to use for MS_score similarity computation
        if getattr(args, 'use_hibou', False):
            # Load Hibou embeddings for MS_score computation only
            print("Loading Hibou embeddings for MS_score similarity computation...")
            if dataset_type == 'ignite':
                hibou_cache_dir = "cache-ignite-hibou/train"
                from ignite_embedding_dataset import IgniteEmbeddingDataset
                hibou_dataset = IgniteEmbeddingDataset(
                    emb_dir=hibou_cache_dir,
                    zscore=True,
                    l2_normalize=True
                )
            else:
                hibou_cache_dir = "cache-hibou/train"
                from spider_embedding_dataset import SpiderEmbeddingDataset
                hibou_dataset = SpiderEmbeddingDataset(
                    emb_dir=hibou_cache_dir,
                    zscore=True,
                    l2_normalize=True
                )
            if getattr(args, 'ms_use_raw', False):
                ms_embeddings = hibou_dataset.get_raw_embeddings()
                print("Using raw Hibou embeddings for MS_score")
            else:
                ms_embeddings = hibou_dataset.emb_all
                print("Using normalized Hibou embeddings for MS_score")
        else:
            # Use the same baseline embeddings (UNI) for MS_score
            if getattr(args, 'ms_use_raw', False):
                # Use raw (pre-normalization) UNI embeddings for MS computation
                raw_embeddings = train_dataset.get_raw_embeddings()
                ms_embeddings = raw_embeddings
                print("Using raw UNI embeddings for MS_score computation")
            else:
                # Use the processed feature matrix (original UNI embeddings) as both feature and embedding matrix
                ms_embeddings = feature_matrix_train
                print("Using normalized UNI embeddings for MS_score computation")

        # Compute MS_score using selected embeddings
        if getattr(args, 'ms_ratio', False):
            from analysis_utils import compute_ms_scores_ratio
            ms_scores = compute_ms_scores_ratio(feature_matrix_train, ms_embeddings, all_feature_indices)
        elif getattr(args, 'ms_slow', False):
            from analysis_utils import compute_ms_scores_slow
            ms_scores = compute_ms_scores_slow(feature_matrix_train, ms_embeddings, all_feature_indices)
        else:
            ms_scores = compute_ms_scores(feature_matrix_train, ms_embeddings, all_feature_indices)

        ms_available = True
        ms_mean = float(np.mean(list(ms_scores.values()))) if ms_scores else 0.0
        embedding_source = "hibou" if getattr(args, 'use_hibou', False) else "uni"
        embedding_type = "raw" if getattr(args, 'ms_use_raw', False) else "normalized"
        print(f"MS_score computed for {len(ms_scores)} features using {embedding_source} {embedding_type} embeddings. Mean MS={ms_mean:.4f}")
    except Exception as e:
        print(f"Warning: Failed to compute MS_score ({e}). Proceeding without MS_score.")
        ms_scores = {}
        ms_available = False

    debug_purity = getattr(args, 'debug_purity', False)

    # -- Optional: interactive cache generation --
    if force_regen or not has_interactive_cache:
        print("Generating baseline interactive cache (train & test)...")
        # Generate for train split
        generate_baseline_interactive_cache(
            feature_matrix_train, labels_train, image_paths_train, "train",
            feature_indices=all_feature_indices,
            mono_scores=mono_scores,
            out_root=interactive_cache_root,
            cache_root=cache_root
        )
        # Generate for test split
        generate_baseline_interactive_cache(
            feature_matrix_test, labels_test, image_paths_test, "test",
            feature_indices=all_feature_indices,
            mono_scores=mono_scores,
            out_root=interactive_cache_root,
            cache_root=cache_root
        )
    else:
        print("[skip] interactive-cache already present.")

    cluster_artifacts = {}
    if getattr(args, "cluster", False):
        print("[baseline][cluster] Clustering baseline embeddings by class-wise means...")
        cluster_dir = os.path.join(analysis_base_dir, "cluster-cache")
        os.makedirs(cluster_dir, exist_ok=True)

        splits_map = {
            "train": (feature_matrix_train, labels_train, image_paths_train),
            "test": (feature_matrix_test, labels_test, image_paths_test),
        }
        if args.cluster_split == "both":
            splits_to_cluster = ["train", "test"]
        else:
            splits_to_cluster = [args.cluster_split]

        for split in splits_to_cluster:
            fm, labs, paths = splits_map.get(split, (None, None, None))
            if fm is None or labs is None or paths is None:
                print(f"[baseline][cluster] Skipping {split} split: missing data.")
                continue
            if len(labs) == 0 or fm.numel() == 0:
                print(f"[baseline][cluster] Skipping {split} split: empty activations.")
                continue
            try:
                n_clusters = None if args.cluster_algorithm == "dbscan" else args.cluster_count
                result = cluster_neurons_by_class_means(
                    fm,
                    labs,
                    paths,
                    all_feature_indices,
                    out_dir=cluster_dir,
                    split_name=split,
                    algorithm=args.cluster_algorithm,
                    n_clusters=n_clusters,
                    random_state=args.cluster_random_state,
                    top_k_patches=args.cluster_top_patches,
                    cache_root=cache_root,
                    use_absolute=args.cluster_use_abs,
                    dbscan_eps=args.cluster_eps,
                    dbscan_min_samples=args.cluster_min_samples,
                )
                cluster_artifacts[split] = result
                print(f"[baseline][cluster] Saved {split} cluster artifacts to {result['npz_path']}")
            except Exception as exc:
                print(f"[baseline][cluster] Failed to cluster {split} split: {exc}")

    # Precompute per-class recall / precision / AUPRC for baseline features
    label2desc = load_label_map(cache_root)
    train_labels_tensor = torch.tensor(labels_train, dtype=torch.long)
    test_labels_tensor = torch.tensor(labels_test, dtype=torch.long)
    max_label_train = int(train_labels_tensor.max().item()) if train_labels_tensor.numel() > 0 else -1
    max_label_test = int(test_labels_tensor.max().item()) if test_labels_tensor.numel() > 0 else -1
    num_classes = max(max_label_train, max_label_test, len(label2desc) - 1) + 1

    train_scores = torch.abs(feature_matrix_train)
    test_scores = torch.abs(feature_matrix_test)
    recall_percentile = getattr(args, 'recall_percentile', 95.0)
    thresholds = compute_activation_thresholds(train_scores, percentile=recall_percentile)
    train_recall_matrix = compute_recall_by_class(train_scores, train_labels_tensor, thresholds, num_classes)
    test_recall_matrix = compute_recall_by_class(test_scores, test_labels_tensor, thresholds, num_classes)
    train_precision_matrix = compute_precision_at_threshold(train_scores, train_labels_tensor, thresholds, num_classes)
    test_precision_matrix = compute_precision_at_threshold(test_scores, test_labels_tensor, thresholds, num_classes)
    train_auprc_matrix = compute_auprc_by_class(train_scores, train_labels_tensor, num_classes)
    test_auprc_matrix = compute_auprc_by_class(test_scores, test_labels_tensor, num_classes)

    thresholds_np = thresholds.detach().cpu()
    train_recall_np = train_recall_matrix.detach().cpu()
    test_recall_np = test_recall_matrix.detach().cpu()
    train_precision_np = train_precision_matrix.detach().cpu()
    test_precision_np = test_precision_matrix.detach().cpu()
    train_auprc_np = train_auprc_matrix.detach().cpu()
    test_auprc_np = test_auprc_matrix.detach().cpu()

    # -- Optional: baseline comparison JSON --
    if force_regen or not has_comparison:
        print("Generating baseline comparison file...")
        baseline_comparison = {
            "baseline_feature_count": len(all_feature_indices),
            "feature_comparisons": [],
            "summary_stats": {
                "monosemanticity": {
                    "total_features": len(all_feature_indices),
                    "mean_monosemanticity": float(np.mean([mono_scores[fid]['M'] for fid in all_feature_indices])),
                    "std_monosemanticity": float(np.std([mono_scores[fid]['M'] for fid in all_feature_indices])),
                    "median_monosemanticity": float(np.median([mono_scores[fid]['M'] for fid in all_feature_indices])),
                    "min_monosemanticity": float(np.min([mono_scores[fid]['M'] for fid in all_feature_indices])),
                    "max_monosemanticity": float(np.max([mono_scores[fid]['M'] for fid in all_feature_indices])),
                    "percentile_25": float(np.percentile([mono_scores[fid]['M'] for fid in all_feature_indices], 25)),
                    "percentile_75": float(np.percentile([mono_scores[fid]['M'] for fid in all_feature_indices], 75))
                }
            }
        }

        purity_train_values = []
        purity_test_values = []
        dead_count = 0
        classification_counts_auprc = {"monosemantic": 0, "nearly-mono": 0, "polysemantic": 0}
        # For binary classification, use lower coverage floor since high selectivity (low recall) is desirable
        # For multi-class, need broader coverage to distinguish from many alternatives
        coverage_floor = 0.02 if num_classes == 2 else 0.10

        for fid in all_feature_indices:
            p_train = compute_purity_at_k(feature_matrix_train, labels_train, fid, k=100)
            p_test = compute_purity_at_k(feature_matrix_test, labels_test, fid, k=100)
            purity_train_values.append(p_train)
            purity_test_values.append(p_test)

            train_cov = float((feature_matrix_train[:, fid] > 1e-6).float().mean().item())
            test_cov  = float((feature_matrix_test[:, fid] > 1e-6).float().mean().item())
            combined_cov = train_cov + test_cov

            margin = mono_scores[fid]['M']
            min_purity = min(p_train, p_test)
            if (margin >= 0.30) or (margin >= 0.18 and min_purity >= 0.95):
                cls_label = "monosemantic"
            elif margin >= 0.10 and min_purity >= 0.60:
                cls_label = "nearly-mono"
            elif combined_cov == 0.0:
                cls_label = "dead"
                dead_count += 1
            else:
                cls_label = "polysemantic"

            recall_threshold = float(thresholds_np[fid].item())
            train_recall_list = train_recall_np[:, fid].tolist()
            test_recall_list = test_recall_np[:, fid].tolist()
            train_precision_list = train_precision_np[:, fid].tolist()
            test_precision_list = test_precision_np[:, fid].tolist()
            train_auprc_list = train_auprc_np[:, fid].tolist()
            test_auprc_list = test_auprc_np[:, fid].tolist()

            combined_auprc = [min(tr, te) for tr, te in zip(train_auprc_list, test_auprc_list)]
            if combined_auprc:
                best_idx = int(np.argmax(combined_auprc))
                best_score = float(combined_auprc[best_idx])
                sorted_scores = sorted(combined_auprc, reverse=True)
                second_best = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            else:
                best_idx = 0
                best_score = 0.0
                second_best = 0.0
            gap_score = best_score - second_best
            train_class_cov = float(train_recall_np[best_idx, fid]) if best_idx < train_recall_np.shape[0] else 0.0
            test_class_cov = float(test_recall_np[best_idx, fid]) if best_idx < test_recall_np.shape[0] else 0.0
            if (train_class_cov < coverage_floor) or (test_class_cov < coverage_floor):
                auprc_label = "polysemantic"
            elif gap_score < 0.2:
                auprc_label = "polysemantic"
            elif best_score >= 0.9:
                auprc_label = "monosemantic"
            elif best_score >= 0.7:
                auprc_label = "nearly-mono"
            else:
                auprc_label = "polysemantic"
            classification_counts_auprc[auprc_label] += 1

            feature_info = {
                "feature_index": fid,
                "monosemanticity_score": margin,
                "MS_score": ms_scores.get(fid, None),
                "train_margin": mono_scores[fid]['m_train'],
                "test_margin": mono_scores[fid]['m_test'],
                "dominant_class_train": mono_scores[fid]['c_star_train_name'],
                "dominant_class_test": mono_scores[fid]['c_star_test_name'],
                "dominant_class_flipped": mono_scores[fid]['flipped'],
                "purity_train_at_100": p_train,
                "purity_test_at_100": p_test,
                "min_train_test_purity": min_purity,
                "train_coverage": train_cov,
                "test_coverage": test_cov,
                "classification_label": cls_label,
                "recall_threshold": recall_threshold,
                "recall_percentile": recall_percentile,
                "train_recall_by_class": train_recall_list,
                "test_recall_by_class": test_recall_list,
                "train_precision_at_threshold_by_class": train_precision_list,
                "test_precision_at_threshold_by_class": test_precision_list,
                "train_auprc_by_class": train_auprc_list,
                "test_auprc_by_class": test_auprc_list,
                "auprc_selectivity_score": best_score,
                "auprc_selectivity_gap": gap_score,
                "auprc_selectivity_class_index": best_idx,
                "auprc_selectivity_class_name": label2desc.get(best_idx, f"Label_{best_idx}"),
                "auprc_selectivity_threshold": 0.9,
                "auprc_selectivity_gap_threshold": 0.2,
                "auprc_selectivity_class_train_coverage": train_class_cov,
                "auprc_selectivity_class_test_coverage": test_class_cov,
                "classification_label_auprc": auprc_label
            }
            baseline_comparison["feature_comparisons"].append(feature_info)

        classifications = [f["classification_label"] for f in baseline_comparison["feature_comparisons"]]
        baseline_comparison["summary_stats"]["classification_counts"] = {
            "monosemantic": classifications.count("monosemantic"),
            "nearly-mono": classifications.count("nearly-mono"),
            "polysemantic": classifications.count("polysemantic"),
            "dead": classifications.count("dead")
        }
        baseline_comparison["summary_stats"]["classification_counts_auprc"] = classification_counts_auprc
        baseline_comparison["summary_stats"]["classification_percentages_auprc"] = {
            k: (v / len(all_feature_indices) * 100.0) if all_feature_indices else 0.0
            for k, v in classification_counts_auprc.items()
        }
        baseline_comparison["summary_stats"]["recall_metadata"] = {
            "percentile": recall_percentile,
            "num_classes": num_classes
        }

        if purity_train_values and purity_test_values:
            baseline_comparison["summary_stats"]["purity_at_100"] = {
                "train_mean": float(np.mean(purity_train_values)),
                "train_std": float(np.std(purity_train_values)),
                "train_median": float(np.median(purity_train_values)),
                "test_mean": float(np.mean(purity_test_values)),
                "test_std": float(np.std(purity_test_values)),
                "test_median": float(np.median(purity_test_values)),
                "train_min": float(np.min(purity_train_values)),
                "train_max": float(np.max(purity_train_values)),
                "test_min": float(np.min(purity_test_values)),
                "test_max": float(np.max(purity_test_values)),
                "train_percentile_25": float(np.percentile(purity_train_values, 25)),
                "train_percentile_75": float(np.percentile(purity_train_values, 75)),
                "test_percentile_25": float(np.percentile(purity_test_values, 25)),
                "test_percentile_75": float(np.percentile(purity_test_values, 75))
            }

        if ms_available and ms_scores:
            ms_vals = list(ms_scores.values())
            baseline_comparison["summary_stats"]["MS_score"] = {
                "mean": float(np.mean(ms_vals)),
                "std": float(np.std(ms_vals)),
                "median": float(np.median(ms_vals)),
                "min": float(np.min(ms_vals)),
                "max": float(np.max(ms_vals)),
                "percentile_25": float(np.percentile(ms_vals, 25)),
                "percentile_75": float(np.percentile(ms_vals, 75))
            }

        with open(comparison_file_path, 'w') as f:
            json.dump(baseline_comparison, f, indent=2)
        with open(os.path.join(analysis_base_dir, "baseline_feature_indices.json"), "w") as f:
            json.dump(all_feature_indices, f)
        print(f"Baseline comparison written: {comparison_file_path}")
    else:
        # Incremental MS_score injection if missing
        try:
            with open(comparison_file_path, 'r') as f:
                existing = json.load(f)
            need_ms = False
            # Determine if MS_score already present
            if existing.get("feature_comparisons"):
                sample_feat = existing["feature_comparisons"][0]
                if "MS_score" not in sample_feat:
                    need_ms = True
            else:
                need_ms = True
            if need_ms and ms_scores:
                print("[incremental] Adding MS_score fields to existing baseline_feature_comparison.json ...")
                for feat_entry in existing.get("feature_comparisons", []):
                    fid = feat_entry.get("feature_index")
                    if fid is not None:
                        feat_entry["MS_score"] = ms_scores.get(fid, None)
                # Summary stats patch
                ms_vals = list(ms_scores.values())
                existing.setdefault("summary_stats", {})["MS_score"] = {
                    "mean": float(np.mean(ms_vals)),
                    "std": float(np.std(ms_vals)),
                    "median": float(np.median(ms_vals)),
                    "min": float(np.min(ms_vals)),
                    "max": float(np.max(ms_vals)),
                    "percentile_25": float(np.percentile(ms_vals, 25)),
                    "percentile_75": float(np.percentile(ms_vals, 75))
                }
                with open(comparison_file_path, 'w') as f:
                    json.dump(existing, f, indent=2)
                print("[incremental] MS_score successfully appended.")
            else:
                print("[skip] Existing baseline comparison already has MS_score.")
        except Exception as e:
            print(f"Warning: Failed incremental MS_score patch: {e}")


    print(f"Baseline core analysis done. Base dir: {analysis_base_dir}/")
    if not (force_regen or not has_interactive_cache):
        print("  (interactive-cache skipped)")
    if not (force_regen or not has_comparison):
        print("  (baseline_feature_comparison skipped)")

    # --- NEW: Generate baseline feature analysis (top 50 by monosemanticity score) ---
    try:
        if force_regen or not has_topk:
            TOP_N = 50
            print(f"Generating baseline feature analysis for top {TOP_N} features (by monosemanticity score)...")
            sorted_by_mono = sorted(all_feature_indices, key=lambda fid: mono_scores[fid]['M'], reverse=True)
            top_feature_indices = sorted_by_mono[:TOP_N]
            if top_feature_indices:
                mono_vals = [mono_scores[fid]['M'] for fid in top_feature_indices]
                print(f"Top mono scores: max={max(mono_vals):.4f} min={min(mono_vals):.4f} median={np.median(mono_vals):.4f}")

            from analysis_utils import analyze_locked_features, save_feature_visualizations_fast
            baseline_train_analysis = analyze_locked_features(feature_matrix_train, labels_train, top_feature_indices, top_k=25, debug=debug_purity, cache_root=cache_root)

            viz_parent_dir = os.path.join(analysis_base_dir, "top-k-features")
            os.makedirs(viz_parent_dir, exist_ok=True)
            save_feature_visualizations_fast(
                feature_matrix_train,
                image_paths_train,
                labels_train,
                top_feature_indices,
                split_name="train",
                feature_analysis=baseline_train_analysis,
                base_dir=viz_parent_dir,
                top_k=25,
                grid_images=100,
                style='mpl',
                png=True,
                cache_root=cache_root
            )

            baseline_test_analysis = analyze_locked_features(feature_matrix_test, labels_test, top_feature_indices, top_k=25, debug=debug_purity, cache_root=cache_root)
            save_feature_visualizations_fast(
                feature_matrix_test,
                image_paths_test,
                labels_test,
                top_feature_indices,
                split_name="test",
                feature_analysis=baseline_test_analysis,
                base_dir=viz_parent_dir,
                top_k=25,
                grid_images=100,
                style='mpl',
                png=True,
                cache_root=cache_root
            )

            out_meta = {
                "top_feature_indices": top_feature_indices,
                "train": baseline_train_analysis,
                "test": baseline_test_analysis,
                "selection": "top_monosemanticity",
                "k_purity": 100,
                "generated_at": datetime.now().isoformat()
            }
            with open(topk_meta_path, 'w') as f:
                json.dump(out_meta, f)
            print(f"Baseline feature analysis visuals + metadata saved under: {topk_root}")
            if debug_purity:
                # Consistency report
                inconsistencies = []
                for fid in top_feature_indices:
                    tr = baseline_train_analysis.get(fid, {})
                    if 'viz_majority_label' in tr and tr.get('majority_label') != tr.get('viz_majority_label'):
                        inconsistencies.append({
                            'feature': fid,
                            'purity_majority': tr.get('majority_label_name'),
                            'viz_majority': tr.get('viz_majority_label_name'),
                            'purity': tr.get('purity'),
                            'purity_k': tr.get('purity_k'),
                            'viz_fraction_topk': tr.get('viz_majority_label_fraction_topk')
                        })
                if inconsistencies:
                    print(f"[debug-purity] Found {len(inconsistencies)} inconsistencies (purity@100 majority != top_k majority). Showing first 10:")
                    for item in inconsistencies[:10]:
                        print("  "+json.dumps(item))
                else:
                    print("[debug-purity] No inconsistencies detected between purity@100 and top_k majority labels.")
        else:
            print("[skip] top-k-features already present.")
    except Exception as e:
        print(f"Warning: baseline feature analysis generation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAE Feature Analysis with Train/Test Comparison')

    parser.add_argument('model_dir', type=str, nargs='?', default=None,
                       help='Path to model directory (e.g., sae-models/model-8192-lambda0.4-l2-stats-bs32-lr0.0001)')

    # Feature selection method (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument('--selection-k', type=int, default=None,
                                help='Top k activations to consider when selecting features')
    selection_group.add_argument('--percentile', type=float, default=None,
                                help='Percentile for feature selection (e.g., 95.0 for 95th percentile)')

    parser.add_argument('--topk-samples', type=int, default=25,
                       help='Top k samples per feature for analysis (default: 25)')
    parser.add_argument('--num-features', type=int, default=50,
                       help='Number of top features to analyze (default: 50)')
    parser.add_argument('--recall-percentile', type=float, default=95.0,
                       help='Percentile (0-100) used to set activation thresholds for recall (default: 95.0)')
    parser.add_argument('--auprc-nearly-mono-threshold', type=float, default=0.65,
                       help='Minimum combined train/test AUPRC to treat a feature as nearly-mono when selectivity fails (default: 0.65)')
    parser.add_argument('--grid-images', type=int, default=100,
                       help='Number of images to show in grid (default: 100)')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration of feature matrices even if cached versions exist')
    parser.add_argument('--all-viz', action='store_true',
                        help='Generate all visualizations including feature grids and flipper comparisons')
    parser.add_argument('--baseline', action='store_true',
                       help='Generate baseline interactive cache using original UNI or Hibou embeddings (use with --use-hibou for Hibou)')
    parser.add_argument('--debug-purity', action='store_true',
                       help='Enable detailed purity vs top_k debugging information for baseline feature analysis')
    parser.add_argument('--ms-score', action='store_true', default=False,
                       help='Compute MS_score (similarity-based monosemanticity) for SAE locked features (default: False)')
    parser.add_argument('--ms-use-raw', action='store_true', default=True,
                       help='Use raw (pre z-score & pre L2) input embeddings as similarity space for MS computation (default: True)')
    parser.add_argument('--ms-slow', action='store_true',
                       help='Use slow O(N²) MS_score implementation for performance comparison (default: use fast O(N*D) version)')
    parser.add_argument('--ms-ratio', action='store_true', default=True,
                       help='Use ratio-form MS_score: (||Σ w_i e_i||² - Σ w_i²) / Σ_{i≠j} w_i w_j (default: True)')
    parser.add_argument('--no-ms-ratio', dest='ms_ratio', action='store_false',
                       help='Disable ratio-form MS_score and use density-form instead')
    parser.add_argument('--use-hibou', action='store_true',
                       help='Use Hibou embeddings for MS_score similarity computation only (SAE still uses UNI embeddings)')
    parser.add_argument('--dataset', type=str, choices=["spider", "spider-thorax", "ignite", "kather100k"], default=None,
                       help='Dataset type (auto-detected from model metadata if not specified)')
    parser.add_argument('--cluster', action='store_true',
                       help='Cluster neurons by class-wise activation means and export artifacts for HistoSCOPE')
    parser.add_argument('--cluster-algorithm', type=str, default='kmeans',
                       choices=['kmeans', 'agglomerative', 'dbscan', 'kl', 'wasserstein'],
                       help='Clustering algorithm for --cluster (default: kmeans). Use kl/wasserstein for agglomerative clustering with distribution distances.')
    parser.add_argument('--cluster-count', type=int, default=None,
                       help='Number of clusters for kmeans/agglomerative algorithms')
    parser.add_argument('--cluster-top-patches', type=int, default=3,
                       help='Representative patches to record per neuron for cluster hover (default: 3)')
    parser.add_argument('--cluster-eps', type=float, default=0.6,
                       help='DBSCAN epsilon value (only used when --cluster-algorithm=dbscan)')
    parser.add_argument('--cluster-min-samples', type=int, default=5,
                       help='DBSCAN min_samples value (only used when --cluster-algorithm=dbscan)')
    parser.add_argument('--cluster-use-abs', action='store_true',
                       help='Use absolute activations when computing class means for clustering')
    parser.add_argument('--cluster-random-state', type=int, default=0,
                       help='Random seed for stochastic clustering steps (default: 0)')
    parser.add_argument('--cluster-split', type=str, default='train',
                       choices=['train', 'test', 'both'],
                       help='Which split(s) to base clustering on (default: train)')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming extraction for large datasets (auto-enabled for sr386-raw)')

    args = parser.parse_args()

    # Handle baseline mode
    if args.baseline:
        # Enable raw embeddings by default for baseline analysis
        if not args.ms_use_raw:
            args.ms_use_raw = True
            print("Baseline mode: automatically enabling --ms-use-raw for raw UNI embeddings")
        print("Running baseline analysis with original UNI embeddings...")
        generate_baseline_cache(args)
        sys.exit(0)

    if args.model_dir is None:
        print("Error: model_dir argument is required unless --baseline is specified.")
        parser.print_help()
        sys.exit(1)

    # Set defaults for feature selection if neither specified
    if args.selection_k is None and args.percentile is None:
        args.percentile = 95.0  # Default to percentile method

    # Load model and metadata
    model, metadata, model_name = load_model_from_dir(args.model_dir)

    # Detect dataset type from metadata or command line
    dataset_type = args.dataset
    if dataset_type is None:
        # Try to detect from metadata
        if 'dataset_type' in metadata:
            dataset_type = metadata['dataset_type']
        elif 'dataset_args' in metadata and 'emb_dir' in metadata['dataset_args']:
            emb_dir = metadata['dataset_args']['emb_dir']
            if 'cache-ignite' in emb_dir:
                dataset_type = 'ignite'
            else:
                dataset_type = 'spider'
        else:
            dataset_type = 'spider'  # Default fallback
            print(f"Warning: Could not detect dataset type from metadata, defaulting to '{dataset_type}'")

    print(f"Using dataset type: {dataset_type}")
    args.dataset = dataset_type  # Set for baseline generation

    # Auto-enable streaming for sr386-raw (1.3M patches)
    if dataset_type == 'sr386-raw' and not args.streaming:
        args.streaming = True
        print(f"Auto-enabling --streaming for large dataset: {dataset_type}")

    # Extract cache root from metadata's emb_dir (already contains encoder info)
    emb_dir = metadata.get('dataset_args', {}).get('emb_dir', 'cache/train')
    # Strip the /train or /test suffix to get cache root
    if '/' in emb_dir:
        cache_root = emb_dir.rsplit('/', 1)[0]
    else:
        cache_root = 'cache'

    print(f"Using cache_root: {cache_root} (from metadata emb_dir: {emb_dir})")

    # Set up analysis directory within the model directory
    analysis_base_dir = os.path.join(args.model_dir, "analysis")
    os.makedirs(analysis_base_dir, exist_ok=True)
    report_path = os.path.join(analysis_base_dir, "report.txt")
    report_file = open(report_path, "w")

    # set analysis_util.report_file (this is a global)
    import analysis_utils
    analysis_utils.report_file = report_file

    print("Starting analysis...")  # minimal console echo

    rprint(f"=== LOADING MODEL FROM {args.model_dir} ===")
    rprint(f"=== CONFIGURATION ===")
    rprint(f"Model: {model_name}")
    rprint(f"Dataset: {dataset_type}")
    rprint(f"Analysis output: {analysis_base_dir}")
    rprint(f"SAE embeddings: UNI (always)")
    if getattr(args, 'use_hibou', False):
        rprint(f"MS_score similarity: Hibou (cache-hibou/train)")
    else:
        rprint(f"MS_score similarity: UNI ({dataset_type} cache)")
    if args.selection_k is not None:
        rprint(f"Selection method: Top-k with k={args.selection_k}")
    else:
        rprint(f"Selection method: Percentile-based with percentile={args.percentile}")
    rprint(f"Top k samples: {args.topk_samples}")
    rprint(f"Number of features: {args.num_features}")
    rprint(f"Grid images: {args.grid_images}")
    rprint(f"Force regenerate: {args.force_regenerate}")

    train_dataset, test_dataset = create_datasets_from_metadata(metadata)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=32)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    print("Extracting train features...")  # minimal console echo
    rprint("\n=== STEP 1: Extract features from TRAIN split ===")

    # Use streaming extraction for large datasets or when explicitly requested
    use_streaming = args.streaming

    if use_streaming:
        rprint("Using STREAMING extraction (memory-efficient for large datasets)")
        # Streaming mode: two-pass extraction with feature selection integrated
        if args.ms_score:
            feature_matrix_train, image_paths_train, labels_train, locked_feature_indices, train_coverage_stats, train_input_embeddings = extract_features_streaming(
                model, train_dataloader,
                percentile=args.percentile if args.percentile is not None else 95.0,
                num_features=args.num_features,
                return_inputs=True
            )
        else:
            feature_matrix_train, image_paths_train, labels_train, locked_feature_indices, train_coverage_stats = extract_features_streaming(
                model, train_dataloader,
                percentile=args.percentile if args.percentile is not None else 95.0,
                num_features=args.num_features
            )
            train_input_embeddings = None

        # In streaming mode, feature_matrix_train is already locked (shape: [N, num_features])
        rprint(f"Train locked feature matrix shape: {feature_matrix_train.shape}")
        rprint(f"Locked {len(locked_feature_indices)} features: {locked_feature_indices[:10]}... (showing first 10)")

        # Note: In streaming mode, we don't save full feature matrices (too large)
        # Coverage stats already computed during Pass 1
        train_activation_stats = compute_activation_stats(feature_matrix_train, eps=1e-6)

    else:
        rprint("Using STANDARD extraction (loads full feature matrix into memory)")
        # Standard mode: extract full matrix, then select features
        if not args.force_regenerate:
            feature_matrix_train, image_paths_train, labels_train = load_feature_matrices("train", args.model_dir)
            train_input_embeddings = None
        else:
            feature_matrix_train, image_paths_train, labels_train = None, None, None
            train_input_embeddings = None

        if feature_matrix_train is None:
            rprint("Generating train feature matrix...")
            if args.ms_score:
                feature_matrix_train, image_paths_train, labels_train, train_input_embeddings = extract_features(model, train_dataloader, return_inputs=True)
            else:
                feature_matrix_train, image_paths_train, labels_train = extract_features(model, train_dataloader)
            save_feature_matrices(feature_matrix_train, image_paths_train, labels_train, "train", args.model_dir)

        rprint(f"Train feature matrix shape: {feature_matrix_train.shape}")

        rprint("\n=== STEP 1.5: Compute activation sparsity statistics ===")
        train_activation_stats = compute_activation_stats(feature_matrix_train, eps=1e-6)
        train_coverage_stats = compute_coverage_stats(feature_matrix_train, eps=1e-6)

    # Save patch metadata (image paths and labels) for the patch-browser
    save_patch_metadata(image_paths_train, labels_train, "train", analysis_base_dir, cache_root)
    rprint("Train split activation statistics:")
    rprint(f"  Total samples: {train_activation_stats['num_samples']:,}")
    rprint(f"  Hidden dimensions: {train_activation_stats['hidden_dim']:,}")
    rprint(f"  Mean active units per sample: {train_activation_stats['mean_active_units_per_sample']:.2f}")
    rprint(f"  Median active units per sample: {train_activation_stats['median_active_units_per_sample']:.0f}")
    rprint(f"  Features ever active: {train_activation_stats['features_ever_active']:,} / {train_activation_stats['hidden_dim']:,} ({100*train_activation_stats['features_ever_active']/train_activation_stats['hidden_dim']:.1f}%)")
    rprint(f"  Features active in >0.1% samples: {train_activation_stats['features_active_>0.1%_samples']:,}")
    rprint(f"  Features active in >1% samples: {train_activation_stats['features_active_>1%_samples']:,}")
    rprint(f"  Features active in >5% samples: {train_activation_stats['features_active_>5%_samples']:,}")
    rprint(f"  Features active in >10% samples: {train_activation_stats['features_active_>10%_samples']:,}")
    rprint(f"  Mean activation rate per feature: {train_activation_stats['mean_activation_rate_per_feature']:.4f}")
    rprint(f"  Median activation rate per feature: {train_activation_stats['median_activation_rate_per_feature']:.4f}")

    # Save activation stats to JSON
    activation_stats_file = os.path.join(analysis_base_dir, "activation_stats.json")
    with open(activation_stats_file, 'w') as f:
        json.dump({"train": train_activation_stats, "train_coverage": train_coverage_stats}, f, indent=2)
    rprint(f"Activation statistics saved to: {activation_stats_file}")

    if not use_streaming:
        # Standard mode: select features from full matrix
        if args.selection_k is not None:
            locked_feature_indices, train_scores = select_top_features_from_train(
                feature_matrix_train, selection_k=args.selection_k, num_features=args.num_features
            )
        else:
            locked_feature_indices, train_scores = select_top_features_percentile(
                feature_matrix_train, percentile=args.percentile, num_features=args.num_features
            )

        rprint(f"Locked {len(locked_feature_indices)} features based on train data")
        rprint(f"Locked feature indices: {sorted(locked_feature_indices)[:10]}... (showing first 10, sorted)")
        save_locked_patch_activations(feature_matrix_train, locked_feature_indices, "train", analysis_base_dir)
    else:
        # Streaming mode: features already selected and matrix already locked
        # feature_matrix_train is [N, num_features], need to save activations directly
        # Create a temporary full-width matrix view for save_locked_patch_activations compatibility
        rprint(f"Skipping save_locked_patch_activations in streaming mode (features already locked)")

    print("Analyzing train features...")  # minimal console echo
    rprint("\n=== STEP 2: Analyze locked features on TRAIN split ===")
    train_analysis = analyze_locked_features(feature_matrix_train, labels_train, locked_feature_indices,
                                            top_k=args.topk_samples, cache_root=cache_root,
                                            already_locked=use_streaming)

    if args.all_viz:
        save_feature_visualizations_fast(feature_matrix_train, image_paths_train, labels_train,
                                   locked_feature_indices, "train", train_analysis,
                                   base_dir=os.path.join(analysis_base_dir, "top-k-features"),
                                   top_k=args.topk_samples, grid_images=args.grid_images, cache_root=cache_root)
    else:
        rprint("Skipping train feature visualizations (use --all-viz to generate)")

    train_sorted = sorted(train_analysis.items(), key=lambda x: x[1]['purity'], reverse=True)
    rprint(f"Top 5 purest locked features in train:")
    for i, (feat_idx, info) in enumerate(train_sorted[:5]):
        rprint(f"  {i+1}. Feature {feat_idx}: {info['majority_label_name']} "
              f"(purity@{info['purity_k']}: {info['purity']:.3f})")

    high_purity_features = [info for info in train_analysis.values() if info['purity'] > 0.9]
    rprint(f"Number of features with purity > 0.9: {len(high_purity_features)}")

    print("Extracting test features...")  # minimal console echo
    rprint("\n=== STEP 3: Evaluate same locked features on TEST split ===")

    if use_streaming:
        # Streaming mode: extract only locked features using Pass 2 logic
        rprint("Extracting TEST split with locked features only (streaming)...")

        feature_matrix_test = []
        image_paths_test = []
        labels_test = []
        test_input_embeddings = [] if args.ms_score else None

        locked_indices_tensor = torch.tensor(locked_feature_indices, dtype=torch.long)

        model.eval()
        with torch.no_grad():
            for emb, lab, path in tqdm(test_dataloader, desc="Extracting test (locked features)"):
                input_batch = emb.to("cuda" if torch.cuda.is_available() else "cpu")
                _, h = model(input_batch)
                h_locked = h.cpu()[:, locked_indices_tensor]
                feature_matrix_test.append(h_locked)
                image_paths_test.extend(path)
                labels_test.extend(lab)
                if args.ms_score:
                    test_input_embeddings.append(emb.cpu())

        feature_matrix_test = torch.cat(feature_matrix_test, dim=0)
        if args.ms_score:
            test_input_embeddings = torch.cat(test_input_embeddings, dim=0)
            norms = test_input_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-6)
            test_input_embeddings = test_input_embeddings / norms
        else:
            test_input_embeddings = None

        rprint(f"Test locked feature matrix shape: {feature_matrix_test.shape}")

        # Compute stats on locked features
        test_activation_stats = compute_activation_stats(feature_matrix_test, eps=1e-6)
        # Compute coverage on locked features (safe since matrix is already locked)
        test_coverage_stats = compute_coverage_stats(feature_matrix_test, eps=1e-6)
    else:
        # Standard mode
        if not args.force_regenerate:
            feature_matrix_test, image_paths_test, labels_test = load_feature_matrices("test", args.model_dir)
            test_input_embeddings = None
        else:
            feature_matrix_test, image_paths_test, labels_test = None, None, None
            test_input_embeddings = None

        if feature_matrix_test is None:
            rprint("Generating test feature matrix...")
            if args.ms_score:
                feature_matrix_test, image_paths_test, labels_test, test_input_embeddings = extract_features(model, test_dataloader, return_inputs=True)
            else:
                feature_matrix_test, image_paths_test, labels_test = extract_features(model, test_dataloader)
            save_feature_matrices(feature_matrix_test, image_paths_test, labels_test, "test", args.model_dir)

        rprint(f"Test feature matrix shape: {feature_matrix_test.shape}")
        save_locked_patch_activations(feature_matrix_test, locked_feature_indices, "test", analysis_base_dir)

        test_activation_stats = compute_activation_stats(feature_matrix_test, eps=1e-6)
        test_coverage_stats = compute_coverage_stats(feature_matrix_test, eps=1e-6)

    save_patch_metadata(image_paths_test, labels_test, "test", analysis_base_dir, cache_root)
    rprint("Test split activation statistics:")
    rprint(f"  Total samples: {test_activation_stats['num_samples']:,}")
    rprint(f"  Features ever active: {test_activation_stats['features_ever_active']:,} / {test_activation_stats['hidden_dim']:,} ({100*test_activation_stats['features_ever_active']/test_activation_stats['hidden_dim']:.1f}%)")
    rprint(f"  Mean active units per sample: {test_activation_stats['mean_active_units_per_sample']:.2f}")
    rprint(f"  Median active units per sample: {test_activation_stats['median_active_units_per_sample']:.0f}")

    # Update activation stats file with test data
    with open(activation_stats_file, 'r') as f:
        existing_stats = json.load(f)
    existing_stats["test"] = test_activation_stats
    existing_stats["test_coverage"] = test_coverage_stats
    with open(activation_stats_file, 'w') as f:
        json.dump(existing_stats, f, indent=2)

    # Analyze dead and near-dead features
    dead_analysis = analyze_dead_and_near_dead_features(
        train_coverage_stats, test_coverage_stats,
        tau=1e-5, locked_feature_indices=locked_feature_indices
    )

    # Save dead analysis
    dead_analysis_file = os.path.join(analysis_base_dir, "dead_feature_analysis.json")
    with open(dead_analysis_file, 'w') as f:
        json.dump(dead_analysis, f, indent=2)

    # Print coverage analysis to report
    print_coverage_analysis(train_coverage_stats, test_coverage_stats, dead_analysis, file=report_file)

    test_analysis = analyze_locked_features(feature_matrix_test, labels_test, locked_feature_indices,
                                           top_k=args.topk_samples, cache_root=cache_root,
                                           already_locked=use_streaming)

    if args.all_viz:
        save_feature_visualizations_fast(feature_matrix_test, image_paths_test, labels_test,
                                   locked_feature_indices, "test", test_analysis,
                                   base_dir=os.path.join(analysis_base_dir, "top-k-features"),
                                   top_k=args.topk_samples, grid_images=args.grid_images, cache_root=cache_root)
    else:
        rprint("Skipping test feature visualizations (use --all-viz to generate)")

    print("Comparing train/test features...")  # minimal console echo
    rprint("\n=== STEP 4: Compare train vs test for locked features ===")
    comparison_file = os.path.join(analysis_base_dir, "locked_feature_comparison.json")
    comparison = compare_train_test_locked(train_analysis, test_analysis, locked_feature_indices,
                                        feature_matrix_train, labels_train, feature_matrix_test, labels_test,
                                        comparison_file,
                                        recall_percentile=args.recall_percentile,
                                        nearly_mono_threshold=args.auprc_nearly_mono_threshold,
                                        already_locked=use_streaming)

    # --- Optional: compute MS_score for SAE locked features (raw embedding variant) ---
    if args.ms_score:
        algorithm = "slow O(N²)" if getattr(args, 'ms_slow', False) else "fast O(N*D)"
        ms_variant = "ratio-form" if getattr(args, 'ms_ratio', False) else "density-form"
        print(f"Computing MS_score using {algorithm} algorithm ({ms_variant})...")
        try:
            # Determine which embeddings to use for similarity computation
            if getattr(args, 'use_hibou', False):
                # Load Hibou embeddings for similarity computation
                print("Loading Hibou embeddings for MS_score similarity computation...")
                if dataset_type == 'ignite':
                    hibou_cache_dir = "cache-ignite-hibou/train"
                    from ignite_embedding_dataset import IgniteEmbeddingDataset
                    hibou_dataset = IgniteEmbeddingDataset(
                        emb_dir=hibou_cache_dir,
                        zscore=True,
                        l2_normalize=True
                    )
                else:
                    hibou_cache_dir = "cache-hibou/train"
                    from spider_embedding_dataset import SpiderEmbeddingDataset
                    hibou_dataset = SpiderEmbeddingDataset(
                        emb_dir=hibou_cache_dir,
                        zscore=True,
                        l2_normalize=True
                    )
                if args.ms_use_raw:
                    ms_embeddings = hibou_dataset.get_raw_embeddings().clone()
                    print("Using raw Hibou embeddings for MS_score")
                else:
                    ms_embeddings = hibou_dataset.emb_all.clone()
                    print("Using normalized Hibou embeddings for MS_score")
            else:
                # Use UNI embeddings for similarity computation
                if args.ms_use_raw:
                    try:
                        # Recreate train_dataset to access raw UNI embeddings
                        if 'train_dataset' in locals():
                            ms_embeddings = train_dataset.get_raw_embeddings().clone()
                            print("Using raw UNI embeddings for MS_score")
                        else:
                            ms_embeddings = None
                    except Exception:
                        print("Warning: Could not retrieve raw UNI embeddings from dataset.")
                        ms_embeddings = None
                else:
                    # Use normalized UNI embeddings (the ones fed to SAE)
                    raw_list = []
                    for emb_batch, _, _ in train_dataloader:
                        raw_list.append(emb_batch)
                    ms_embeddings = torch.cat(raw_list, dim=0)
                    print("Using normalized UNI embeddings for MS_score")

            # Fallback if embeddings couldn't be loaded
            if ms_embeddings is None:
                print("Warning: Could not load embeddings for MS_score; using SAE activations as fallback.")
                ms_embeddings = feature_matrix_train.clone()

            # Compute MS_score using the selected embeddings
            if getattr(args, 'ms_ratio', False):
                from analysis_utils import compute_ms_scores_ratio
                ms_scores = compute_ms_scores_ratio(feature_matrix_train, ms_embeddings, locked_feature_indices)
            elif getattr(args, 'ms_slow', False):
                from analysis_utils import compute_ms_scores_slow
                ms_scores = compute_ms_scores_slow(feature_matrix_train, ms_embeddings, locked_feature_indices)
            else:
                ms_scores = compute_ms_scores(feature_matrix_train, ms_embeddings, locked_feature_indices)

            ms_vals = list(ms_scores.values())

            # Inject into in-memory comparison structure
            for fc in comparison['feature_comparisons']:
                fid = fc.get('feature_index')
                if fid in ms_scores:
                    fc['MS_score'] = ms_scores[fid]

            # Summary stats
            embedding_source = "hibou" if getattr(args, 'use_hibou', False) else "uni"
            embedding_type = "raw" if args.ms_use_raw else "normalized"
            comparison.setdefault('summary_stats', {})['MS_score'] = {
                'mean': float(np.mean(ms_vals)),
                'std': float(np.std(ms_vals)),
                'min': float(np.min(ms_vals)),
                'max': float(np.max(ms_vals)),
                'num_features': len(ms_vals),
                'embedding_source': embedding_source,
                'embedding_type': embedding_type
            }

            # Overwrite comparison file with MS augmentation
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            rprint(f"Added MS_score ({embedding_source} {embedding_type} embeddings) to comparison: mean={np.mean(ms_vals):.4f} max={np.max(ms_vals):.4f}")
        except Exception as e:
            rprint(f"Failed to compute MS_score: {e}")
            import traceback
            traceback.print_exc()

    # Print summary to report.txt instead of console
    def print_locked_comparison_summary_to_file(comparison, file):
        stats = comparison['summary_stats']
        print(f"\n=== LOCKED FEATURE CONSISTENCY ANALYSIS ===", file=file)
        print(f"Total locked features: {comparison['locked_feature_count']}", file=file)
        print(f"Features with same majority label: {stats['same_majority_label_count']}/{comparison['locked_feature_count']} ({stats['same_majority_label_percentage']:.1f}%)", file=file)
        print(f"Flipper features (different majority): {stats['flipper_count']}", file=file)
        print(f"Average purity difference: {stats['average_purity_difference']:.3f}", file=file)
        print(f"Max purity difference: {stats['max_purity_difference']:.3f}", file=file)
        print(f"Min purity difference: {stats['min_purity_difference']:.3f}", file=file)
        consistent_features = [f for f in comparison['feature_comparisons']
                              if f['same_majority_label']]
        consistent_features.sort(key=lambda x: x['purity_difference'])
        print(f"\nTop 5 most consistent features (same majority, lowest purity diff):", file=file)
        for i, feat in enumerate(consistent_features[:5]):
            print(f"  {i+1}. Feature {feat['feature_index']}: {feat['train_majority']} "
                  f"(train purity: {feat['train_purity']:.3f}, test purity: {feat['test_purity']:.3f}, "
                  f"diff: {feat['purity_difference']:.3f})", file=file)
        if comparison['flippers']:
            print(f"\nFlipper features (different majority labels):", file=file)
            for i, feat in enumerate(comparison['flippers'][:5]):
                print(f"  {i+1}. Feature {feat['feature_index']}: {feat['train_majority']} → {feat['test_majority']} "
                      f"(purity diff: {feat['purity_difference']:.3f})", file=file)
    print_locked_comparison_summary_to_file(comparison, report_file)

    if args.all_viz:
        print("Generating flipper visualizations...")  # minimal console echo
        rprint("\n=== STEP 5: Generate flipper visualizations ===")
        if comparison['flippers']:
            rprint(f"Generating side-by-side visualizations for {len(comparison['flippers'])} flippers...")
            save_flipper_visualizations_fast(feature_matrix_train, image_paths_train, labels_train,
                                       feature_matrix_test, image_paths_test, labels_test,
                                       comparison['flippers'],
                                       base_dir=os.path.join(analysis_base_dir, "top-k-features"),
                                       top_k=args.topk_samples, grid_images=args.grid_images, cache_root=cache_root)
            rprint(f"Flipper visualizations saved to {analysis_base_dir}/top-k-features/flippers/")
        else:
            rprint("No flipper features found!")
    else:
        rprint("\n=== STEP 5: Skipping flipper visualizations (use --all-viz to generate) ===")

    # print("Generating class feature analysis...")  # minimal console echo
    # rprint("\n=== STEP 6: Generate class feature analysis ===")
    # for split, feature_matrix, labels in [("train", feature_matrix_train, labels_train), ("test", feature_matrix_test, labels_test)]:
    #     heatmap_dir = os.path.join(analysis_base_dir, "class-heatmaps", split)
    #     rprint(f"Generating class feature heatmap for {split} split...")
    #     generate_class_feature_heatmap(
    #         feature_matrix, labels,
    #         feature_indices=locked_feature_indices,
    #         top_features=len(locked_feature_indices),
    #         save_dir=heatmap_dir,
    #         sort_features=True
    #     )

    print("Generating interactive cache...")  # minimal console echo
    rprint("\n=== STEP 6: Generate interactive cache ===")
    for split, feature_matrix, labels, image_paths in [
        ("train", feature_matrix_train, labels_train, image_paths_train),
        ("test", feature_matrix_test, labels_test, image_paths_test)
    ]:
        rprint(f"Generating interactive cache for {split} split...")
        generate_interactive_cache(
            feature_matrix, labels, image_paths, split,
            feature_indices=locked_feature_indices,
            top_features=len(locked_feature_indices),
            topk_per_cell=50,
            out_root=os.path.join(analysis_base_dir, "interactive-cache"),
            cache_root=cache_root
        )

    cluster_artifacts = {}
    if args.cluster:
        rprint("\n=== STEP 7: Cluster neurons ===")
        cluster_dir = os.path.join(analysis_base_dir, "cluster-cache")
        splits_map = {
            "train": (feature_matrix_train, labels_train, image_paths_train),
            "test": (feature_matrix_test, labels_test, image_paths_test),
        }
        if args.cluster_split == "both":
            splits_to_cluster = ["train", "test"]
        else:
            splits_to_cluster = [args.cluster_split]

        for split in splits_to_cluster:
            fm, labs, paths = splits_map.get(split, (None, None, None))
            if fm is None or labs is None or paths is None:
                rprint(f"[cluster] Skipping {split} split: missing data.")
                if report_file:
                    print(f"[cluster] Skipping {split} split: missing data.", file=report_file)
                continue

            rprint(f"[cluster] Clustering {len(locked_feature_indices)} neurons on {split} split using {args.cluster_algorithm}...")
            try:
                n_clusters = args.cluster_count if args.cluster_algorithm != "dbscan" else None
                cluster_result = cluster_neurons_by_class_means(
                    fm,
                    labs,
                    paths,
                    locked_feature_indices,
                    out_dir=cluster_dir,
                    split_name=split,
                    algorithm=args.cluster_algorithm,
                    n_clusters=n_clusters,
                    random_state=args.cluster_random_state,
                    top_k_patches=args.cluster_top_patches,
                    cache_root=cache_root,
                    use_absolute=args.cluster_use_abs,
                    dbscan_eps=args.cluster_eps,
                    dbscan_min_samples=args.cluster_min_samples,
                )
                cluster_artifacts[split] = cluster_result
                rprint(f"[cluster] Saved {split} cluster artifacts to {cluster_result['npz_path']}")
                if report_file:
                    print(f"[cluster] Saved {split} cluster artifacts to {cluster_result['npz_path']}", file=report_file)
            except Exception as exc:
                message = f"[cluster] Failed to cluster {split} split: {exc}"
                rprint(message)
                if report_file:
                    print(message, file=report_file)

    print("Analysis complete.")  # minimal console echo
    rprint(f"\n=== ANALYSIS COMPLETE ===")
    rprint(f"Model: {model_name}")
    rprint(f"All results saved to: {analysis_base_dir}/")
    rprint(f"Feature visualizations: {analysis_base_dir}/top-k-features/")
    rprint(f"Class heatmaps: {analysis_base_dir}/class-heatmaps/")
    rprint(f"Interactive cache: {analysis_base_dir}/interactive-cache/")
    if cluster_artifacts:
        rprint(f"Cluster cache: {analysis_base_dir}/cluster-cache/")
    rprint(f"Detailed comparison: {analysis_base_dir}/locked_feature_comparison.json")

    with open(os.path.join(analysis_base_dir, "locked_feature_indices.json"), "w") as f:
        json.dump(locked_feature_indices, f)
    rprint(f"Locked feature indices saved to {analysis_base_dir}/locked_feature_indices.json")
    report_file.close()
