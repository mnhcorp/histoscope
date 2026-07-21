import torch
import torch.nn as nn               # for nn.Module
import torch.nn.functional as F     # for ReLU
from torch.utils.data import DataLoader, TensorDataset
from spider_embedding_dataset import SpiderEmbeddingDataset
from ignite_embedding_dataset import IgniteEmbeddingDataset
from kather100k_embedding_dataset import Kather100kEmbeddingDataset
import json
import os
import argparse
import random
import numpy as np
import time
from tqdm import tqdm
import psutil

# We inherit from nn.Module to instantiate our network
class SparseAutoEncoder(nn.Module):
    # The constructor is where we set up the layers
    # The parameters allow us to create SAEs of different sizes without changing class code.
    def __init__(self, input_dim, hidden_dim, tie_weights=False, use_pre_bias=False,
                 activation="relu", topk_k=None):
        # Calls the constructor of nn.Module? What does this do (TBD #1)
        # Ok, the init method of nn.Module sets up dictionaries to track parameters and suck. Deep-dive later.
        super().__init__()

        # --- The Linear Layer ---

        # A Linear layer is a simple linear transformation
        # It takes the input_dim and output_dim as parameters
        # Here the output_dim is just the hidden layer

        # Encoder: This is the data coming into the network and mapped to a lower dimension, the latent layer.
        # For TopK and BatchTopK activations we avoid an encoder bias (matching the paper); otherwise keep default bias.
        if activation in {"topk", "batchtopk"}:
            self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # Store whether to tie weights
        self.tie_weights = tie_weights

        # Store whether to use pre-bias
        self.use_pre_bias = use_pre_bias

        # Store activation configuration
        if activation not in {"relu", "topk", "batchtopk"}:
            raise ValueError(f"Unsupported activation '{activation}'. Choose 'relu', 'topk', or 'batchtopk'.")
        if activation in {"topk", "batchtopk"}:
            if topk_k is None:
                raise ValueError("topk_k must be provided when activation='topk' or 'batchtopk'")
            if topk_k <= 0:
                raise ValueError("topk_k must be a positive integer")
        self.activation = activation
        self.topk_k = topk_k

        # Pre-bias: subtracted from input before encoding, added back after decoding
        if use_pre_bias:
            self.b_pre = nn.Parameter(torch.zeros(input_dim))
        else:
            self.b_pre = None

        if tie_weights:
            # When tying weights, decoder weights are the transpose of encoder weights
            # We don't create a separate decoder layer, instead we use encoder.weight.T
            self.decoder = None
        else:
            # Decoder: This is the data going out of the network. We are reconstructing the input from the latent layer.
            # Why is the bias set to false? What does the bias even do?? (TBD #2)
            # Bias is literally what it says, it's a bias. It is a value added to the result to shift the value up or down.
            # But why do we set it to False?
            # Prevent the model from cheating. Since the bias is also a learned parameter
            # the encoder may output negative values and the ReLU will set it to zero, which means
            # the sparsity penalty is a happy camper, all zero values. And the decoder will learn a bias
            # to make sure that the reconstructed output is an average of the input - which makes the
            # reconstruction loss happy. The result is that the model is stuck in a local minima,
            # which topographically I cannot visualise but imagine it's stuck in a rock pool or something.
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    # The forward method is what runs the forward pass on input x
    def forward(self, x):
        # Apply pre-bias if enabled: subtract b_pre from input
        if self.use_pre_bias:
            x_centered = x - self.b_pre
        else:
            x_centered = x

        # We first compute the encoder pre-activations
        pre_activations = self.encoder(x_centered)

        # Apply activation
        if self.activation == "relu":
            h = F.relu(pre_activations)
        elif self.activation == "topk":
            # TopK: Keep only the top-k activations per sample (by value), zero out the rest
            k = min(self.topk_k, pre_activations.shape[1])
            if k <= 0:
                h = torch.zeros_like(pre_activations)
            elif k == pre_activations.shape[1]:
                h = pre_activations
            else:
                topk_vals, topk_indices = torch.topk(pre_activations, k=k, dim=1)
                h = torch.zeros_like(pre_activations)
                h.scatter_(1, topk_indices, topk_vals)
        else:  # batchtopk
            # BatchTopK: Keep only the top-k activations across the entire batch (by value), zero out the rest
            batch_size, num_features = pre_activations.shape
            total_activations = batch_size * num_features
            k = min(self.topk_k, total_activations)

            if k <= 0:
                h = torch.zeros_like(pre_activations)
            elif k >= total_activations:
                h = pre_activations
            else:
                # Flatten to select top-k across entire batch
                flat_activations = pre_activations.view(-1)
                topk_vals, topk_flat_indices = torch.topk(flat_activations, k=k)

                # Create output tensor and scatter the top-k values back
                h = torch.zeros_like(pre_activations).view(-1)
                h[topk_flat_indices] = topk_vals
                h = h.view(batch_size, num_features)

        # We then reconstruct the input
        if self.tie_weights:
            # Use transpose of encoder weights for decoder
            x_reconstructed = F.linear(h, self.encoder.weight.t())
        else:
            x_reconstructed = self.decoder(h)

        # Add back pre-bias if enabled: x_hat = W_dec * z + b_pre
        if self.use_pre_bias:
            x_reconstructed = x_reconstructed + self.b_pre

        # We return both because the loss = reconstruction loss + sparsity of h
        return x_reconstructed, h

def set_seeds(seed=42):
    """Set all random seeds for reproducible training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt

def plot_loss_vs_sparsity(lambda_list, loss_list, avg_sparse_list, lambda_values, hidden_dim, dataset_config, batch_size, learning_rate, config_name,
                          tie_weights=False, use_pre_bias=False, activation="relu", topk_k=None, dataset_type="spider", out_dir="sae-models"):
    for i, lambda_l1 in enumerate(lambda_values):
        model_name = create_model_name(hidden_dim, lambda_l1, dataset_config, batch_size, learning_rate, tie_weights, use_pre_bias, activation, topk_k, dataset_type)
        model_dir = create_model_dir(model_name, dataset_type, out_dir)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        color1 = 'tab:blue'
        ln1 = ax1.plot(lambda_list, loss_list, marker='o', color=color1, label='Loss')
        ax1.axvline(x=lambda_l1, color='red', linestyle='--', alpha=0.7, label=f'Current λ={lambda_l1}')
        ax1.set_xlabel('Lambda (Sparsity Coefficient)')
        ax1.set_ylabel('Loss', color=color1)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ln2 = ax2.plot(lambda_list, avg_sparse_list, marker='s', color=color2, label='Avg Active Features')
        ax2.set_ylabel('Avg Active Features', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc='best')

        ax1.grid(True, which='both', axis='both', alpha=0.3)
        ax1.set_title(f'Loss vs Sparsity Curve (Current λ={lambda_l1})')
        fig.tight_layout()
        fig.savefig(f"{model_dir}/loss_vs_sparsity.png", dpi=150)
        plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color1 = 'tab:blue'
    ln1 = ax1.plot(lambda_list, loss_list, marker='o', color=color1, label='Loss')
    ax1.set_xlabel('Lambda (Sparsity Coefficient)')
    ax1.set_ylabel('Loss', color=color1)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ln2 = ax2.plot(lambda_list, avg_sparse_list, marker='s', color=color2, label='Avg Active Features')
    ax2.set_ylabel('Avg Active Features', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='best')

    ax1.grid(True, which='both', axis='both', alpha=0.3)
    ax1.set_title(f'Loss vs Sparsity Curve - {config_name}')
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{dataset_type}/lambda_vs_loss_vs_sparse_{config_name}.png", dpi=150)

    all_model_names = [create_model_name(hidden_dim, lam, dataset_config, batch_size, learning_rate, tie_weights, use_pre_bias, activation, topk_k, dataset_type) for lam in lambda_values]
    all_model_dirs = [create_model_dir(name, dataset_type, out_dir) for name in all_model_names]
    print(f"\nModels saved in directories: {all_model_dirs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE and optionally run analysis.")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input dimension (default: 1024)")
    parser.add_argument("--hidden-dim", type=int, default=8192, help="Hidden dimension (default: 8192)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default: 2)")
    parser.add_argument("--lambda-values", type=float, nargs="+", default=[0.4], help="List of lambda values (default: [0.4])")
    parser.add_argument("--l2", action="store_true", help="Enable L2 normalization for dataset")
    parser.add_argument("--zscore", action="store_true", help="Enable z-score standardization for dataset")
    parser.add_argument("--run-analysis", action="store_true", help="Run sae_feature_analysis_v2.py after training")
    parser.add_argument("--viz", action="store_true", help="Generate all visualizations when running analysis (adds --all-viz flag)")
    parser.add_argument("--streaming-topk", action="store_true", help="Use streaming two-pass top-k extraction in evaluation to avoid materializing full feature matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--log-interval", type=int, default=50, help="Batches between progress logs (default: 50)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor (default: 2)")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU training via DataParallel when multiple GPUs available")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP) training on CUDA")
    parser.add_argument("--dataset", type=str, choices=["spider", "spider-breast", "spider-thorax", "spider-skin", "ignite", "kather100k", "surgen", "surgen-sr386", "sr386-raw"], default="spider", help="Dataset to use (default: spider)")
    parser.add_argument("--cache-root", type=str, default=None, help="Optional base cache directory to read/write dataset caches (overrides defaults)")
    parser.add_argument("--split", type=str, choices=["train", "test", "train_nonorm", "validation"], default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--tie-weights", action="store_true", help="Tie encoder and decoder weights (decoder = encoder.T)")
    parser.add_argument("--use-pre-bias", action="store_true", help="Enable pre-bias: z = ReLU(W_enc*(x - b_pre) + b_enc), x_hat = W_dec*z + b_pre")
    parser.add_argument("--activation", type=str, choices=["relu", "topk", "batchtopk"], default="relu",
                        help="Activation function for latent codes (default: relu)")
    parser.add_argument("--topk-k", type=int, default=None,
                        help="Number of activations to keep when using top-k or batch-top-k activation")
    parser.add_argument("--num-features", type=int, default=None,
                        help="Number of top features to analyze (passed to sae_feature_analysis_v2.py)")
    parser.add_argument("--eval-classification", action="store_true",
                        help="Evaluate classification performance after training (runs evaluate_classification_performance.py)")
    parser.add_argument("--topk-dense", type=int, default=None,
                        help="Extract only top-k features globally for classification eval (passed to evaluate_classification_performance.py)")
    parser.add_argument("--topk-method", type=str, default="frequency", choices=["frequency", "magnitude", "variance"],
                        help="Method for selecting top-k features: frequency (default), magnitude, or variance")
    parser.add_argument("--pca", action="store_true",
                        help="Compute PCA baseline for classification eval (passed to evaluate_classification_performance.py)")
    parser.add_argument("--rp", action="store_true",
                        help="Compute Random Projection baseline for classification eval (passed to evaluate_classification_performance.py)")
    parser.add_argument("--rp-hierarchical-bootstrap", action="store_true",
                        help="When used with --rp, enable multi-seed hierarchical bootstrap in evaluation script")
    parser.add_argument("--rp-hierarchical-seeds", type=int, default=None,
                        help="Number of RP seeds for hierarchical bootstrap (forwarded to evaluation script)")
    parser.add_argument('--classifier', type=str, default='lr', choices=['lr','mlp','transformer'],
                        help='Classifier to use for evaluation script (default: lr)')
    parser.add_argument('--classifier-params', type=str, default=None,
                        help='JSON string of classifier parameters to forward to evaluation script')
    parser.add_argument('--out-dir', type=str, default='sae-models',
                        help='Output directory for trained models (default: sae-models)')
    args = parser.parse_args()
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    NUM_EPOCHS = args.epochs
    lambda_values = args.lambda_values
    l2_normalize = args.l2
    zscore_normalize = args.zscore
    dataset_type = args.dataset
    split = args.split
    tie_weights = args.tie_weights
    use_pre_bias = args.use_pre_bias
    activation = args.activation
    topk_k = args.topk_k
    out_dir = args.out_dir

    if activation in {"topk", "batchtopk"}:
        if topk_k is None:
            raise ValueError(f"--topk-k must be provided when --activation {activation}")
        if topk_k <= 0:
            raise ValueError("--topk-k must be a positive integer")

    # Configure dataset based on type
    if dataset_type == "spider":
        emb_dir = os.path.join(args.cache_root, split) if getattr(args, 'cache_root', None) else "cache/train"
        dataset_config = {
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = SpiderEmbeddingDataset(**dataset_config)
    elif dataset_type == "spider-breast":
        emb_dir = os.path.join(args.cache_root, split) if getattr(args, 'cache_root', None) else "cache-spider-breast/train"
        dataset_config = {
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = SpiderEmbeddingDataset(**dataset_config)
    elif dataset_type == "spider-thorax":
        emb_dir = os.path.join(args.cache_root, split) if getattr(args, 'cache_root', None) else "cache-spider-thorax/train"
        dataset_config = {
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = SpiderEmbeddingDataset(**dataset_config)
    elif dataset_type == "spider-skin":
        emb_dir = os.path.join(args.cache_root, split) if getattr(args, 'cache_root', None) else "cache-spider-skin/train"
        dataset_config = {
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = SpiderEmbeddingDataset(**dataset_config)
    elif dataset_type == "ignite":
        # When cache_root is provided we expect the user to provide the ignite cache root
        # and the script will look for the split folder under it. If not provided, fallback to default cache-ignite/<split>/org
        if getattr(args, 'cache_root', None):
            dataset_config = {
                "emb_dir": os.path.join(args.cache_root, split),
                "zscore": zscore_normalize,
                "l2_normalize": l2_normalize
            }
        else:
            base = "cache-ignite"
            dataset_config = {
                "emb_dir": f"{base}/{split}/org",  # Use original embeddings from org folder
                "zscore": zscore_normalize,
                "l2_normalize": l2_normalize
            }
        dataset = IgniteEmbeddingDataset(**dataset_config)

        # remove the org suffix from the dataset config so feature analysis uses the mask-in patches
        dataset_config["emb_dir"] = f"cache-ignite/{split}"
    elif dataset_type == "kather100k":
        emb_dir = os.path.join(args.cache_root, split) if getattr(args, 'cache_root', None) else f"cache-nctcrche100k/embeddings_uni_{split}"
        dataset_config = {
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = Kather100kEmbeddingDataset(**dataset_config)
    elif dataset_type == "surgen" or dataset_type == "surgen-sr386":
        # SurGen embeddings are written to cache-surgen/ by the conversion tool
        # For the SR386 variant, converter writes to cache-surgen-sr386/
        # If cache_root provided, expect it to point to the dataset cache root and look under <cache_root>/<split>
        if getattr(args, 'cache_root', None):
            emb_dir = os.path.join(args.cache_root, split)
        else:
            default_target = "cache-surgen" if dataset_type == "surgen" else "cache-surgen-sr386"
            emb_dir = f"{default_target}/{split}"
        dataset_config = {
            # Use split-specific subdirectory (train/test) so the dataset loader
            # finds the emb_*.pt shards created by the converter.
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = SpiderEmbeddingDataset(**dataset_config)
    elif dataset_type == "sr386-raw":
        # SR386 raw patches - embeddings extracted from raw PNGs
        # Default cache structure: cache-sr386-raw-{model}/{split}/
        # If cache_root provided, expect it to point to the dataset cache root and look under <cache_root>/<split>
        if getattr(args, 'cache_root', None):
            emb_dir = os.path.join(args.cache_root, split)
        else:
            # Default to UNI embeddings (cache-sr386-raw-uni)
            emb_dir = f"cache-sr386-raw-uni/{split}"
        dataset_config = {
            "emb_dir": emb_dir,
            "zscore": zscore_normalize,
            "l2_normalize": l2_normalize
        }
        dataset = SpiderEmbeddingDataset(**dataset_config)

    # Configure DataLoader for throughput when using GPUs
    num_workers = getattr(args, 'num_workers', 4)
    prefetch_factor = getattr(args, 'prefetch_factor', 2)
    pin_memory = True if device.type == 'cuda' else False
    persistent_workers = True if num_workers > 0 else False

    train_loader = DataLoader(dataset, batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers,
                              prefetch_factor=prefetch_factor)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/{dataset_type}", exist_ok=True)

    lambda_list = []
    loss_list = []
    avg_sparse_list = []

    def create_model_name(hidden_dim, lambda_l1, dataset_config, batch_size, learning_rate,
                           tie_weights=False, use_pre_bias=False, activation="relu", topk_k=None, dataset_type="spider"):
        l2_norm = "l2" if dataset_config.get("l2_normalize", False) else "no-l2"
        zscore_norm = "zscore" if dataset_config.get("zscore", False) else "no-zscore"
        tied_weights = "tied" if tie_weights else "untied"
        pre_bias = "prebias" if use_pre_bias else "no-prebias"
        if activation == "topk":
            activation_tag = f"topk{topk_k}"
            lambda_tag = None
        elif activation == "batchtopk":
            activation_tag = f"batchtopk{topk_k}"
            lambda_tag = None
        else:
            activation_tag = activation
            lambda_tag = f"lambda{lambda_l1}"

        name_parts = [f"model-exp{hidden_dim}"]
        if lambda_tag:
            name_parts.append(lambda_tag)
        name_parts.extend([
            l2_norm,
            zscore_norm,
            tied_weights,
            pre_bias,
            f"act{activation_tag}",
            f"bs{batch_size}",
            f"lr{learning_rate}"
        ])
        return "-".join(name_parts)

    def create_model_dir(model_name, dataset_type="spider", out_dir="sae-models"):
        return f"{out_dir}/{dataset_type}/{model_name}"

    for lambda_l1 in lambda_values:
        model_name = create_model_name(hidden_dim, lambda_l1, dataset_config, batch_size, learning_rate,
                                       tie_weights, use_pre_bias, activation, topk_k, dataset_type)
        model_dir = create_model_dir(model_name, dataset_type, out_dir)
        os.makedirs(model_dir, exist_ok=True)

        metadata = {
            "model_name": model_name,
            "dataset_type": dataset_type,
            "dataset_split": split if dataset_type in ["ignite", "kather100k"] else "train",
            "hyperparameters": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "lambda_l1": lambda_l1,
                "learning_rate": learning_rate,
                "num_epochs": NUM_EPOCHS,
                "batch_size": batch_size,
                "tie_weights": tie_weights,
                "use_pre_bias": use_pre_bias,
                "activation": activation,
                "topk_k": topk_k
            },
            "dataset_args": dataset_config,
            "final_metrics": {}
        }

        process = psutil.Process(os.getpid())
        cpu_peak_bytes = process.memory_info().rss
        samples_processed = 0
        gpu_peak_bytes = 0
        training_start_time = time.perf_counter()
        if device.type == 'cuda':
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        sae_model = SparseAutoEncoder(input_dim, hidden_dim, tie_weights=tie_weights, use_pre_bias=use_pre_bias,
                                      activation=activation, topk_k=topk_k)
        sae_model = sae_model.to(device)

        # Multi-GPU: simple DataParallel wrapper (easy to use). For best performance at scale, use torch.distributed.launch with DDP.
        if getattr(args, 'multi_gpu', False) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected ({torch.cuda.device_count()}), wrapping model with DataParallel")
            sae_model = nn.DataParallel(sae_model)

        optimizer = torch.optim.Adam(sae_model.parameters(), lr=learning_rate)
        sae_model.train()
        print(f"===== Starting training with lambda: {lambda_l1} in {model_name} =====")

        # Prepare AMP scaler if requested (per-model)
        use_amp = getattr(args, 'amp', False) and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Only enable verbose per-batch logging and tqdm for the SurGen dataset
        if dataset_type == "surgen":
            log_interval = getattr(args, 'log_interval', 50)
            for epoch in range(NUM_EPOCHS):
                running_loss = 0.0
                running_rec = 0.0
                running_sp = 0.0
                start_time = time.time()
                for batch_idx, (emb, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit='batch')):
                    input_batch = emb.to(device)
                    samples_processed += input_batch.shape[0]
                    try:
                        rss_now = process.memory_info().rss
                        if rss_now > cpu_peak_bytes:
                            cpu_peak_bytes = rss_now
                    except Exception:
                        pass
                    optimizer.zero_grad()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            x_reconstructed, h = sae_model(input_batch)
                            loss_reconstruction = F.mse_loss(input_batch, x_reconstructed)

                            if activation in {"topk", "batchtopk"}:
                                loss_sparsity = torch.tensor(0.0, device=input_batch.device)
                                total_loss = loss_reconstruction
                            else:
                                loss_sparsity = h.abs().mean()
                                total_loss = loss_reconstruction + lambda_l1 * loss_sparsity

                        # Scaled backward step
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        x_reconstructed, h = sae_model(input_batch)
                        loss_reconstruction = F.mse_loss(input_batch, x_reconstructed)

                        if activation in {"topk", "batchtopk"}:
                            loss_sparsity = torch.tensor(0.0, device=input_batch.device)
                            total_loss = loss_reconstruction
                        else:
                            # Compute L1-norm
                            loss_sparsity = h.abs().mean()
                            total_loss = loss_reconstruction + lambda_l1 * loss_sparsity

                        total_loss.backward()
                        optimizer.step()

                    # Update running metrics
                    running_loss += float(total_loss.item())
                    running_rec += float(loss_reconstruction.item())
                    try:
                        running_sp += float(loss_sparsity.item())
                    except Exception:
                        running_sp += float(loss_sparsity)

                    # Periodic logging
                    if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                        batches_done = batch_idx + 1
                        elapsed = time.time() - start_time
                        avg_loss = running_loss / batches_done
                        avg_rec = running_rec / batches_done
                        avg_sp = running_sp / batches_done
                        eta = (elapsed / batches_done) * (len(train_loader) - batches_done)
                        gpu_stats = ''
                        if torch.cuda.is_available():
                            try:
                                used = torch.cuda.memory_allocated(device) / (1024 ** 3)
                                raw_peak = torch.cuda.max_memory_allocated(device)
                                gpu_peak_bytes = max(gpu_peak_bytes, raw_peak)
                                peak = raw_peak / (1024 ** 3)
                                gpu_stats = f" gpu_used={used:.2f}GB peak={peak:.2f}GB"
                                # reset peak stats to report fresh next time
                                torch.cuda.reset_peak_memory_stats(device)
                            except Exception:
                                gpu_stats = ''
                        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Batch {batches_done}/{len(train_loader)} avg_loss={avg_loss:.6f} rec={avg_rec:.6f} sp={avg_sp:.6f} ETA={eta:.1f}s{gpu_stats}")
                epoch_time = time.time() - start_time
                avg_epoch_loss = running_loss / max(1, len(train_loader))
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete: avg_loss={avg_epoch_loss:.6f} time={epoch_time:.1f}s")
        else:
            # Non-SurGen datasets: use the original simpler loop without per-batch logging
            for epoch in range(NUM_EPOCHS):
                for emb, label in train_loader:
                    input_batch = emb.to(device)
                    samples_processed += input_batch.shape[0]
                    try:
                        rss_now = process.memory_info().rss
                        if rss_now > cpu_peak_bytes:
                            cpu_peak_bytes = rss_now
                    except Exception:
                        pass
                    optimizer.zero_grad()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            x_reconstructed, h = sae_model(input_batch)
                            loss_reconstruction = F.mse_loss(input_batch, x_reconstructed)

                            if activation in {"topk", "batchtopk"}:
                                loss_sparsity = torch.tensor(0.0, device=input_batch.device)
                                total_loss = loss_reconstruction
                            else:
                                loss_sparsity = h.abs().mean()
                                total_loss = loss_reconstruction + lambda_l1 * loss_sparsity

                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        x_reconstructed, h = sae_model(input_batch)
                        loss_reconstruction = F.mse_loss(input_batch, x_reconstructed)

                        if activation in {"topk", "batchtopk"}:
                            loss_sparsity = torch.tensor(0.0, device=input_batch.device)
                            total_loss = loss_reconstruction
                        else:
                            loss_sparsity = h.abs().mean()
                            total_loss = loss_reconstruction + lambda_l1 * loss_sparsity

                        total_loss.backward()
                        optimizer.step()
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss.item():6f}")

        if device.type == 'cuda':
            try:
                gpu_peak_bytes = max(gpu_peak_bytes, torch.cuda.max_memory_allocated(device))
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        training_wall_time = time.perf_counter() - training_start_time
        if samples_processed == 0:
            samples_processed = len(train_loader.dataset) * NUM_EPOCHS
        samples_per_sec = samples_processed / training_wall_time if training_wall_time > 0 else 0.0

        print("Training complete!")
        sae_model.eval()
        with torch.no_grad():
            eval_batch = next(iter(train_loader))[0].to("cuda" if torch.cuda.is_available() else "cpu")
            _, h_activated = sae_model(eval_batch)
            # evaluate on L0 sparsity
            num_active_features = torch.sum(h_activated > 1e-6, dim=1)
            avg_active_features = num_active_features.float().mean().item()
            print(f"Average number of active features: {avg_active_features}")

        metadata["final_metrics"] = {
            "final_loss": total_loss.item(),
            "final_reconstruction_loss": loss_reconstruction.item(),
            "final_sparsity_loss": loss_sparsity.item(),
            "avg_active_features": avg_active_features
        }
        metadata["resource_metrics"] = {
            "training": {
                "wall_time_sec": float(training_wall_time),
                "samples_processed": int(samples_processed),
                "samples_per_sec": float(samples_per_sec),
                "cpu_peak_mem_bytes": int(cpu_peak_bytes),
                "gpu_peak_mem_bytes": int(gpu_peak_bytes) if gpu_peak_bytes else None,
                "batch_size": batch_size,
                "num_epochs": NUM_EPOCHS
            }
        }

        torch.save(sae_model.state_dict(), f"{model_dir}/model.pt")
        with open(f"{model_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        lambda_list.append(lambda_l1)
        loss_list.append(total_loss.item())
        avg_sparse_list.append(avg_active_features)

    config_name = create_model_name(hidden_dim, "all", dataset_config, batch_size, learning_rate,
                                    tie_weights, use_pre_bias, activation, topk_k, dataset_type).replace("-lambdaall", "")


    # Call the plotting function
    plot_loss_vs_sparsity(lambda_list, loss_list, avg_sparse_list, lambda_values, hidden_dim, dataset_config, batch_size, learning_rate,
                          config_name, tie_weights, use_pre_bias, activation, topk_k, dataset_type, out_dir)

    # Optionally run analysis
    # Run analysis if requested and/or run evaluation optionally.
    for lambda_l1 in lambda_values:
        model_name = create_model_name(hidden_dim, lambda_l1, dataset_config, batch_size, learning_rate,
                                       tie_weights, use_pre_bias, activation, topk_k, dataset_type)
        model_dir = create_model_dir(model_name, dataset_type, out_dir)

        if args.run_analysis:
            print(f"Running analysis for {model_name} ...")
            viz_flag = " --all-viz --cluster" if args.viz else ""
            num_features_flag = f" --num-features {args.num_features}" if args.num_features else ""
            cache_flag = f" --cache-root {args.cache_root}" if getattr(args, 'cache_root', None) else ""
            os.system(f"python3 sae_feature_analysis_v2.py {model_dir}{viz_flag}{num_features_flag}{cache_flag}")

        # Optionally run classification evaluation. Allow --eval-classification to run even when --run-analysis is not set.
        if args.eval_classification:
            print(f"Running classification evaluation for {model_name} ...")
            topk_flag = f" --topk-dense {args.topk_dense}" if args.topk_dense else ""
            method_flag = f" --topk-method {args.topk_method}" if args.topk_dense else ""
            pca_flag = " --pca" if args.pca else ""
            rp_flag = " --rp" if args.rp else ""
            rp_hier_flag = " --rp-hierarchical-bootstrap" if args.rp_hierarchical_bootstrap else ""
            rp_seed_flag = f" --rp-hierarchical-seeds {args.rp_hierarchical_seeds}" if args.rp_hierarchical_seeds is not None else ""
            classifier_flag = f" --classifier {args.classifier}" if getattr(args, 'classifier', None) else ""
            classifier_params_flag = f" --classifier-params '{args.classifier_params}'" if getattr(args, 'classifier_params', None) else ""
            streaming_flag = " --streaming-topk" if getattr(args, 'streaming_topk', False) else ""

            cache_flag = f" --cache-root {args.cache_root}" if getattr(args, 'cache_root', None) else ""
            os.system(
                "python3 evaluate_classification_performance.py "
                f"{model_dir}{topk_flag}{method_flag}{pca_flag}{rp_flag}{rp_hier_flag}{rp_seed_flag}"
                f"{classifier_flag}{classifier_params_flag}{streaming_flag}{cache_flag}"
            )
