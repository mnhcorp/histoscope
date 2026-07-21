#!/usr/bin/env python3
"""
Vocabulary Builder App

Gradio interface for pathologists to build controlled vocabulary
by reviewing patches from SPIDER dataset classes.
"""
import gradio as gr
import json
import random
from pathlib import Path
from typing import Dict, List, Set
import socket
import argparse
import sys
import numpy as np
import torch

# SPIDER dataset controlled vocabulary (class names)
SPIDER_CLASSES = [
    "Adenocarcinoma HG",
    "Adenocarcinoma LG",
    "Adenoma HG",
    "Adenoma LG",
    "Fat",
    "Hyperplastic polyp",
    "Inflammation",
    "Mucus",
    "Muscle",
    "Necrosis",
    "Sessile serrated lesion",
    "Stroma healthy",
    "Vessels"
]

# Common quick-add concepts by category
QUICK_ADD_CONCEPTS = {
    "Glandular": ["Glandular", "Cribriform glands", "Tubular glands", "Dilated glands"],
    "Stromal": ["Stroma", "Desmoplastic stroma", "Loose fibrovascular stroma", "Dense collagenous stroma"],
    "Inflammatory": ["Inflammatory infiltrate", "Lymphocytes", "Plasma cells", "Neutrophils"],
    "Necrotic": ["Necrotic", "Necrotic debris", "Karyorrhectic nuclei"],
    "Vascular": ["Vascular", "Vessels", "Blood vessels", "Capillaries"],
    "Mucin": ["Mucin", "Mucin pools", "Mucin-filled spaces"]
}


class VocabBuilderApp:
    def __init__(self, export_dir: str, patches_per_set: int = 10, cache_dir: str = "cache"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.patches_per_set = patches_per_set
        self.cache_dir = Path(cache_dir)

        self.vocabulary: Set[str] = set()  # For counting unique concepts
        self.vocabulary_list: List[str] = []  # For preserving all entries (including duplicates)
        self.set_concepts: Dict[int, List[str]] = {}  # Concepts added per set
        self.set_patches: Dict[int, List[str]] = {}  # Cached patches for each set
        self.current_set = 0
        self.total_sets = 13  # One set per SPIDER class
        self.completed_sets: Set[int] = set()

        # Pre-existing vocabulary support
        self.preexisting_vocab: Dict[int, List[str]] = {}  # Original pre-existing concepts per set
        self.deleted_preexisting: Dict[int, Set[str]] = {}  # Deleted pre-existing concepts per set

        # Load patch data from feature matrix
        self.patch_data = self.load_patch_data()

        # Load excluded paths from h1/h2/h3 studies
        self.excluded_paths: Set[str] = self.load_excluded_paths()

        self.load_preexisting_vocab()
        self.load_progress()

    def load_patch_data(self) -> Dict:
        """Load patch information from feature matrix"""
        split = "train"
        feature_matrix_dir = self.cache_dir / "feature-matrix" / split

        # Load image paths
        image_paths_file = feature_matrix_dir / "image_paths.json"
        if not image_paths_file.exists():
            print(f"Warning: Image paths not found at {image_paths_file}")
            return {"image_paths": [], "labels": [], "class_names": SPIDER_CLASSES}

        with open(image_paths_file) as f:
            image_paths = json.load(f)

        # Load labels
        labels_file = feature_matrix_dir / "labels.pt"
        if labels_file.exists():
            labels_data = torch.load(labels_file)
            # Handle different label formats
            if isinstance(labels_data, list):
                labels = [int(label.item()) if hasattr(label, 'item') else int(label) for label in labels_data]
            elif hasattr(labels_data, 'numpy'):
                labels = labels_data.numpy().tolist()
            else:
                labels = list(labels_data)
        else:
            print(f"Warning: Labels not found at {labels_file}")
            labels = [0] * len(image_paths)

        # Load class names from cache metadata
        cache_file = self.cache_dir / split / "cache.npz"
        class_names = SPIDER_CLASSES  # Default
        if cache_file.exists():
            try:
                arr = np.load(cache_file, allow_pickle=True)
                if "class_names" in arr.files:
                    class_names = arr["class_names"].tolist()
            except Exception as e:
                print(f"Warning: Could not load class names: {e}")

        # Group patches by class
        class_to_indices = {i: [] for i in range(len(class_names))}
        for idx, label in enumerate(labels):
            if label < len(class_names):
                class_to_indices[label].append(idx)

        return {
            "image_paths": image_paths,
            "labels": labels,
            "class_names": class_names,
            "class_to_indices": class_to_indices
        }

    def load_excluded_paths(self) -> Set[str]:
        """Load image paths used in h1/h2/h3 studies to exclude from vocab builder"""
        excluded = set()

        # Check each study directory
        for study in ['h1', 'h2', 'h3']:
            study_dir = self.export_dir.parent / study

            # For h1, check both session A (main) and session B
            metadata_files = [study_dir / 'metadata.json']
            if study == 'h1':
                metadata_files.append(study_dir / 'session-B' / 'metadata.json')

            for metadata_file in metadata_files:
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    # Determine the base directory for resolving paths
                    # For session-B, paths are relative to session-B directory
                    base_dir = metadata_file.parent

                    study_excluded_count = 0
                    session_label = f"{study}" if metadata_file.parent == study_dir else f"{study}/session-B"

                    # H1: Extract paths from both monosemantic and polysemantic neurons
                    if 'neurons' in metadata and isinstance(metadata['neurons'], dict):
                        for neuron_type in ['monosemantic', 'polysemantic']:
                            if neuron_type in metadata['neurons']:
                                for neuron in metadata['neurons'][neuron_type]:
                                    if 'patches' in neuron and 'patches' in neuron['patches']:
                                        for patch_info in neuron['patches']['patches']:
                                            exported_path = patch_info.get('exported_path', '')
                                            if exported_path:
                                                full_path = str(base_dir / exported_path)
                                                excluded.add(full_path)
                                                study_excluded_count += 1

                    # H2: Extract paths from classes -> neurons structure
                    if 'classes' in metadata:
                        for class_info in metadata['classes']:
                            if 'neurons' in class_info:
                                for neuron in class_info['neurons']:
                                    if 'patches' in neuron and 'patches' in neuron['patches']:
                                        for patch_info in neuron['patches']['patches']:
                                            exported_path = patch_info.get('exported_path', '')
                                            if exported_path:
                                                full_path = str(base_dir / exported_path)
                                                excluded.add(full_path)
                                                study_excluded_count += 1

                    # H3: Extract paths from bridge_neurons and control_neurons
                    for neuron_list_key in ['bridge_neurons', 'control_neurons']:
                        if neuron_list_key in metadata:
                            for neuron in metadata[neuron_list_key]:
                                if 'patches' in neuron and 'patches' in neuron['patches']:
                                    for patch_info in neuron['patches']['patches']:
                                        exported_path = patch_info.get('exported_path', '')
                                        if exported_path:
                                            full_path = str(base_dir / exported_path)
                                            excluded.add(full_path)
                                            study_excluded_count += 1

                    print(f"✓ Loaded {study_excluded_count} excluded paths from {session_label}")

                except Exception as e:
                    print(f"Warning: Could not load excluded paths from {metadata_file}: {e}")

        print(f"✓ Total excluded paths: {len(excluded)}")
        return excluded

    def load_preexisting_vocab(self):
        """Load pre-existing vocabulary from vocab-per-set.json"""
        vocab_file = self.export_dir / "vocab-per-set.json"

        if not vocab_file.exists():
            print(f"Warning: Pre-existing vocabulary not found at {vocab_file}")
            return

        try:
            with open(vocab_file) as f:
                vocab_data = json.load(f)

            # Convert from "set_1" to 0-indexed keys
            for i in range(self.total_sets):
                set_key = f"set_{i + 1}"
                if set_key in vocab_data:
                    self.preexisting_vocab[i] = vocab_data[set_key]

            print(f"✓ Loaded pre-existing vocabulary: {sum(len(v) for v in self.preexisting_vocab.values())} concepts across {len(self.preexisting_vocab)} sets")
        except Exception as e:
            print(f"Warning: Could not load pre-existing vocabulary: {e}")

    def load_progress(self):
        """Load existing progress from disk"""
        progress_file = self.export_dir / "vocab_builder_progress.json"

        if progress_file.exists():
            with open(progress_file) as f:
                data = json.load(f)
                # Load vocabulary_list if available (new format), otherwise use vocabulary (old format)
                self.vocabulary_list = data.get('vocabulary_list', data.get('vocabulary', []))
                self.vocabulary = set(self.vocabulary_list)  # Derive set from list
                self.set_concepts = {int(k): v for k, v in data.get('set_concepts', {}).items()}
                self.completed_sets = set(data.get('completed_sets', []))
                self.current_set = data.get('current_set', 0)
                # Load deleted pre-existing concepts
                deleted_data = data.get('deleted_preexisting', {})
                self.deleted_preexisting = {int(k): set(v) for k, v in deleted_data.items()}
                print(f"✓ Loaded progress: {len(self.vocabulary_list)} concepts, {len(self.completed_sets)} sets completed, current_set={self.current_set}")
        else:
            # On first load, pre-populate vocabulary with all pre-existing concepts
            for set_idx, concepts in self.preexisting_vocab.items():
                self.vocabulary_list.extend(concepts)
                self.vocabulary.update(concepts)
            if self.preexisting_vocab:
                print(f"✓ Pre-populated vocabulary with {len(self.vocabulary_list)} pre-existing concepts")

    def has_progress(self) -> bool:
        """Return True if any progress has been recorded."""
        return bool(self.vocabulary_list or self.completed_sets or self.current_set > 0)

    def is_complete(self) -> bool:
        """Return True if all sets have been completed."""
        return self.current_set >= self.total_sets

    def save_progress(self):
        """Save progress to disk"""
        progress_file = self.export_dir / "vocab_builder_progress.json"

        data = {
            'vocabulary': sorted(list(self.vocabulary)),  # Unique concepts (for backward compatibility)
            'vocabulary_list': self.vocabulary_list,  # All concepts including duplicates
            'set_concepts': {str(k): v for k, v in self.set_concepts.items()},
            'completed_sets': sorted(list(self.completed_sets)),
            'current_set': self.current_set,
            'total_sets': self.total_sets,
            'patches_per_set': self.patches_per_set,
            'deleted_preexisting': {str(k): sorted(list(v)) for k, v in self.deleted_preexisting.items()}
        }

        with open(progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_random_patches_for_set(self, set_num: int, use_cache: bool = True) -> tuple:
        """Get random patches for a specific set, cycling through all 13 classes

        Args:
            set_num: Current set number (0-indexed)
            use_cache: If True, return cached patches if available

        Returns:
            Tuple of (list of image paths, class_name)
        """
        if not self.patch_data["image_paths"]:
            return [], "Unknown"

        class_to_indices = self.patch_data["class_to_indices"]
        class_names = self.patch_data["class_names"]
        image_paths = self.patch_data["image_paths"]

        # Get classes that have samples
        available_classes = [cls_idx for cls_idx in range(len(class_names)) if class_to_indices.get(cls_idx)]

        if not available_classes:
            return [], "Unknown"

        # Determine which class to show for this set
        # Cycle through all 13 classes
        class_idx = available_classes[set_num % len(available_classes)]
        class_name = class_names[class_idx]

        # Return cached patches if available and requested
        if use_cache and set_num in self.set_patches:
            return self.set_patches[set_num], class_name

        # Generate new random patches
        class_indices = class_to_indices[class_idx]

        # Filter out indices that point to excluded paths
        filtered_indices = [
            idx for idx in class_indices
            if image_paths[idx] not in self.excluded_paths
        ]

        # If we filtered out too many, warn but proceed
        if len(filtered_indices) < self.patches_per_set:
            print(f"Warning: Only {len(filtered_indices)} non-excluded patches available for class {class_name}")

        # Get n random patches from this class (using filtered indices)
        if len(filtered_indices) >= self.patches_per_set:
            selected_indices = random.sample(filtered_indices, self.patches_per_set)
        elif len(filtered_indices) > 0:
            # If not enough, sample with replacement from available
            selected_indices = random.choices(filtered_indices, k=self.patches_per_set)
        else:
            # Fallback: if no non-excluded patches, use original indices (shouldn't happen)
            print(f"Warning: No non-excluded patches for class {class_name}, using any available")
            if len(class_indices) >= self.patches_per_set:
                selected_indices = random.sample(class_indices, self.patches_per_set)
            else:
                selected_indices = random.choices(class_indices, k=self.patches_per_set)

        patches = [image_paths[idx] for idx in selected_indices]

        # Cache the patches for this set
        self.set_patches[set_num] = patches

        return patches, class_name

    def add_concept(self, concept: str) -> tuple:
        """Add a concept to vocabulary and current set"""
        concept = concept.strip()

        if not concept:
            return False, "⚠️ Please enter a concept"

        # Allow duplicates - we'll deduplicate post-hoc
        # Add to vocabulary (using list instead of set to allow duplicates)
        if not hasattr(self, 'vocabulary_list'):
            # For backward compatibility, convert set to list if needed
            self.vocabulary_list = list(self.vocabulary) if isinstance(self.vocabulary, set) else []

        self.vocabulary_list.append(concept)

        # Also keep in set for quick counting of unique concepts
        self.vocabulary.add(concept)

        # Track which set this was added in
        if self.current_set not in self.set_concepts:
            self.set_concepts[self.current_set] = []
        self.set_concepts[self.current_set].append(concept)

        # Auto-save
        self.save_progress()

        return True, f"✓ Added '{concept}'"

    def complete_set(self):
        """Mark current set as complete and move to next"""
        self.completed_sets.add(self.current_set)
        # Always increment to move to next (even if beyond total_sets)
        # This allows save_and_next to detect completion properly
        self.current_set += 1
        self.save_progress()

    def delete_preexisting_concept(self, concept: str) -> tuple:
        """Delete a pre-existing concept from current set"""
        if self.current_set not in self.preexisting_vocab:
            return False, "⚠️ No pre-existing concepts for this set"

        if concept not in self.preexisting_vocab[self.current_set]:
            return False, f"⚠️ Concept not found in pre-existing vocabulary"

        # Track deletion
        if self.current_set not in self.deleted_preexisting:
            self.deleted_preexisting[self.current_set] = set()
        self.deleted_preexisting[self.current_set].add(concept)

        # Remove from global vocabulary tracking
        if concept in self.vocabulary:
            # Count how many times this concept appears in vocabulary_list
            count = self.vocabulary_list.count(concept)
            if count > 0:
                # Remove one instance from vocabulary_list
                self.vocabulary_list.remove(concept)
                # If no more instances remain, remove from set
                if concept not in self.vocabulary_list:
                    self.vocabulary.discard(concept)

        # Auto-save
        self.save_progress()

        return True, f"✓ Deleted '{concept}' from pre-existing concepts"

    def format_progress(self) -> str:
        """Format progress as HTML"""
        completed = len(self.completed_sets)
        # Current set display (cap at total_sets for display purposes)
        current_display = min(self.current_set + 1, self.total_sets)
        pct = (completed / self.total_sets) * 100.0

        return f"""
<div style="margin-bottom: 0.5rem;">
  <span style="font-size: 0.9rem; font-weight: 600; color: #666;">Progress: Set {current_display}/{self.total_sets} | {len(self.vocabulary)} concepts</span>
</div>
<div class="progress-container" role="progressbar" aria-valuenow="{completed}" aria-valuemin="0" aria-valuemax="{self.total_sets}">
  <div class="progress-fill" style="width: {pct:.1f}%"></div>
</div>
"""

    def format_preexisting_concepts(self) -> str:
        """Format pre-existing concepts for current set as markdown list"""
        if self.current_set not in self.preexisting_vocab:
            return ""

        # Get active pre-existing concepts (not deleted)
        deleted = self.deleted_preexisting.get(self.current_set, set())
        active_concepts = [c for c in self.preexisting_vocab[self.current_set] if c not in deleted]

        if not active_concepts:
            return ""

        # Build simple markdown list
        items = "\n".join([f"  • {c}" for c in active_concepts])

        return f"**Pre-existing Concepts for This Set:**\n{items}\n\n_To remove a concept that doesn't match, select it below and click Delete._"

    def get_active_preexisting_concepts(self) -> list:
        """Get list of active pre-existing concepts for current set (for dropdown)"""
        if self.current_set not in self.preexisting_vocab:
            return []

        deleted = self.deleted_preexisting.get(self.current_set, set())
        return [c for c in self.preexisting_vocab[self.current_set] if c not in deleted]

    def format_current_set_concepts(self) -> str:
        """Format concepts added in current set (user-added only)"""
        if self.current_set not in self.set_concepts or not self.set_concepts[self.current_set]:
            return ""

        concepts = self.set_concepts[self.current_set]
        items = "\n".join([f"  • {c}" for c in concepts])
        return f"**User-Added Concepts for This Set:**\n{items}"

    def format_vocabulary_list(self, deduplicate: bool = False) -> str:
        """Format complete vocabulary as HTML list

        Args:
            deduplicate: If True, show only unique concepts (deduplicated)
        """
        if not self.vocabulary:
            return "<p><em>No concepts added yet</em></p>"

        if deduplicate:
            sorted_vocab = sorted(list(self.vocabulary))
            header = f"<p style='color: #666; font-style: italic; margin-bottom: 0.5rem;'>Showing {len(sorted_vocab)} unique concepts (duplicates across classes removed)</p>"
        else:
            sorted_vocab = sorted(self.vocabulary_list)
            header = f"<p style='color: #666; font-style: italic; margin-bottom: 0.5rem;'>Showing all {len(sorted_vocab)} concepts (including duplicates)</p>"

        items = "\n".join([f"{i+1}. {concept}" for i, concept in enumerate(sorted_vocab)])

        return f"""
{header}
<div style="max-height: 400px; overflow-y: auto; padding: 1rem; background: #f9f9f9; border-radius: 8px;">
  <pre style="margin: 0; white-space: pre-wrap;">{items}</pre>
</div>
"""

    def export_final_vocabulary(self) -> str:
        """Export vocabulary as JSON and text"""
        # JSON export
        json_file = self.export_dir / "vocabulary.json"
        with open(json_file, 'w') as f:
            json.dump({
                'vocabulary_unique': sorted(list(self.vocabulary)),  # Unique concepts
                'vocabulary_all': self.vocabulary_list,  # All concepts including duplicates
                'total_unique_concepts': len(self.vocabulary),
                'total_all_concepts': len(self.vocabulary_list),
                'sets_completed': len(self.completed_sets),
                'total_sets': self.total_sets
            }, f, indent=2)

        # Text export for easy reading (all concepts)
        txt_file = self.export_dir / "vocabulary_all.txt"
        with open(txt_file, 'w') as f:
            for i, concept in enumerate(self.vocabulary_list, 1):
                f.write(f"{i}. {concept}\n")

        # Text export for unique concepts
        txt_file_unique = self.export_dir / "vocabulary_unique.txt"
        with open(txt_file_unique, 'w') as f:
            for i, concept in enumerate(sorted(list(self.vocabulary)), 1):
                f.write(f"{i}. {concept}\n")

        return f"✓ Vocabulary exported to:\n  • {json_file}\n  • {txt_file}\n  • {txt_file_unique}"


def create_interface(export_dir: str = "study-export/vocab-builder", patches_per_set: int = 10, cache_dir: str = "cache"):
    """Create Gradio interface"""

    app = VocabBuilderApp(export_dir, patches_per_set, cache_dir)

    # Custom CSS
    custom_css = """
    .progress-container {
        position: relative;
        width: 100%;
        height: 1.5rem;
        border-radius: 999px;
        background-color: #f0f0f0;
        overflow: hidden;
        margin: 0rem 0 1.25rem 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #1f6feb, #6f9fff);
        transition: width 0.3s ease;
    }
    .instruction-box {
        padding: 1rem;
        background: #f0f7ff;
        border-left: 4px solid #1f6feb;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .gradio-container label,
    .gradio-container .gr-form,
    .gradio-container .gr-input,
    .gradio-container .gr-box,
    .gradio-container p,
    .gradio-container button,
    .gradio-container textarea,
    .gradio-container select {
        font-size: calc(1rem + 2px) !important;
    }
    .gradio-container h1 {
        font-size: 2em !important;
    }
    .gradio-container h2 {
        font-size: 1.5em !important;
    }
    .patch-grid {
        display: grid;
        gap: 0.5rem;
    }
    .quick-add-btn {
        font-size: 0.9rem !important;
        padding: 0.25rem 0.5rem !important;
    }
    /* Hide footer */
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(title="UNI-SAE Vocabulary Builder", theme=gr.themes.Monochrome(), css=custom_css) as demo:

        # Header
        gr.Markdown("# 🔬 UNI-SAE Study: Vocabulary Builder", elem_id="top-of-app")

        # Check if user has already started (has concepts or completed sets)
        # This checks the loaded progress from disk
        has_started = app.has_progress()

        # Welcome/Instructions Screen
        with gr.Column(visible=not has_started) as welcome_col:
            gr.Markdown(f"""
<div class="instruction-box">

## Welcome, Dr. [Name]

---

**Purpose:** Build a list **of** morphological concepts that you observe **in** histopathology patches.

You will review **{app.total_sets} sets of** diverse patches (**{app.patches_per_set} patches per set**) and identify the morphological features you see.

**Guidelines:**

• Be **specific** (e.g., "cribriform glands" not just "glands")
• Focus on **morphology**, not diagnosis
• Add concepts **as** you see them
• Don't worry about being exhaustive

**Estimated time:** 15-20 minutes

This is **INDEPENDENT** work - do **not** consult **with** other participants.

</div>
""")

            with gr.Row():
                start_btn = gr.Button("🚀 Start Vocabulary Builder", size="lg", variant="primary")

        # Main Builder Screen (visible if started but not completed)
        show_builder = has_started and not app.is_complete()
        with gr.Column(visible=show_builder) as builder_col:

            # Progress
            progress_html = gr.HTML(app.format_progress())

            # Class indicator
            class_name_display = gr.Markdown(f"## Vocabulary Builder (Set {app.current_set + 1}/{app.total_sets})")

            gr.Markdown("""
**Review these patches and add morphological concepts you observe.** [Click any patch to view larger]
""")

            # Patch gallery - grid of patches
            with gr.Row():
                patch_gallery = gr.Gallery(
                    label="Patches",
                    columns=5,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                    show_label=False
                )

            with gr.Row():
                refresh_images_btn = gr.Button("🔄 Refresh Images", variant="secondary", size="sm")

            gr.Markdown("---")

            # Add concept section
            gr.Markdown("### Add morphological concepts you observe:")

            with gr.Row():
                concept_input = gr.Textbox(
                    label="",
                    placeholder="Type concept here...",
                    scale=4
                )
                add_btn = gr.Button("+ Add", variant="primary", scale=1)

            # Status message
            status_msg = gr.Markdown("")

            gr.Markdown("---")

            # Pre-existing concepts display
            preexisting_concepts = gr.Markdown(app.format_preexisting_concepts())

            # Delete pre-existing concept section
            with gr.Row(visible=True) as delete_preexisting_row:
                preexisting_dropdown = gr.Dropdown(
                    choices=app.get_active_preexisting_concepts(),
                    label="Select pre-existing concept to delete",
                    value=None,
                    interactive=True,
                    scale=4
                )
                delete_preexisting_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)

            # Concepts added this set (user-added)
            current_set_concepts = gr.Markdown(app.format_current_set_concepts())

            # Navigation
            with gr.Row():
                clear_set_btn = gr.Button("🗑️ Clear This Set", variant="secondary")
                nav_prev_btn = gr.Button("← Previous", variant="secondary")
                save_next_btn = gr.Button("💾 Save & Next →", variant="primary", scale=2)

        # Completion Screen (visible if all completed)
        show_completion = has_started and app.is_complete()
        with gr.Column(visible=show_completion) as completion_col:
            gr.Markdown("# Vocabulary Builder Complete! 🎉")

            completion_summary = gr.Markdown(f"""
**Sets reviewed:** {len(app.completed_sets)}/{app.total_sets}

**Total unique concepts identified:** {len(app.vocabulary)}

**Total concepts (with duplicates):** {len(app.vocabulary_list)}

---

### Your Final Vocabulary:
""")

            final_vocab_display = gr.HTML(app.format_vocabulary_list(deduplicate=True))

            gr.Markdown("---")

            gr.Markdown("**Review or modify classes:**")

            with gr.Row():
                review_prev_btn = gr.Button("← Previous Class", variant="secondary", scale=1)
                submit_btn = gr.Button("✅ Submit Final Vocabulary", variant="primary", size="lg", scale=2)

            final_status = gr.Markdown("")

        # === Callback Functions ===

        def start_session():
            # Check if all sets are already completed
            if app.is_complete():
                summary_text = f"""
**Sets reviewed:** {len(app.completed_sets)}/{app.total_sets}

**Total unique concepts identified:** {len(app.vocabulary)}

**Total concepts (with duplicates):** {len(app.vocabulary_list)}

---

### Your Final Vocabulary:
"""
                return {
                    welcome_col: gr.update(visible=False),
                    builder_col: gr.update(visible=False),
                    completion_col: gr.update(visible=True),
                    class_name_display: "",
                    patch_gallery: [],
                    progress_html: "",
                    preexisting_concepts: "",
                    preexisting_dropdown: gr.update(choices=[], value=None),
                    current_set_concepts: "",
                    completion_summary: summary_text,
                    final_vocab_display: app.format_vocabulary_list(deduplicate=True),
                    nav_prev_btn: gr.update(interactive=True)
                }

            # Load patches for current set (resume from saved state)
            patches, class_name = app.get_random_patches_for_set(app.current_set, use_cache=True)

            # Disable previous button if at first class
            prev_btn_interactive = app.current_set > 0

            return {
                welcome_col: gr.update(visible=False),
                builder_col: gr.update(visible=True),
                completion_col: gr.update(visible=False),
                class_name_display: f"## Vocabulary Builder (Set {app.current_set + 1}/{app.total_sets}) - **{class_name}**",
                patch_gallery: patches,
                progress_html: app.format_progress(),
                preexisting_concepts: app.format_preexisting_concepts(),
                preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                current_set_concepts: app.format_current_set_concepts(),
                completion_summary: "",
                final_vocab_display: "",
                nav_prev_btn: gr.update(interactive=prev_btn_interactive)
            }

        def add_concept_fn(concept_text):
            success, msg = app.add_concept(concept_text)

            return {
                concept_input: "",
                status_msg: msg,
                current_set_concepts: app.format_current_set_concepts(),
                progress_html: app.format_progress()
            }

        def delete_preexisting_fn(concept_to_delete):
            if not concept_to_delete:
                return {
                    status_msg: "⚠️ Please select a concept to delete",
                    preexisting_concepts: app.format_preexisting_concepts(),
                    preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                    progress_html: app.format_progress()
                }

            success, msg = app.delete_preexisting_concept(concept_to_delete)

            return {
                status_msg: msg,
                preexisting_concepts: app.format_preexisting_concepts(),
                preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                progress_html: app.format_progress()
            }

        def save_and_next(pending_concept: str = ""):
            # Auto-save pending concept if not empty
            if pending_concept and pending_concept.strip():
                app.add_concept(pending_concept.strip())

            app.complete_set()

            # Check if we've completed all sets (current_set has been incremented by complete_set)
            if app.is_complete():
                # Go to completion screen
                summary_text = f"""
**Sets reviewed:** {len(app.completed_sets)}/{app.total_sets}

**Total unique concepts identified:** {len(app.vocabulary)}

**Total concepts (with duplicates):** {len(app.vocabulary_list)}

---

### Your Final Vocabulary:
"""
                return {
                    builder_col: gr.update(visible=False),
                    completion_col: gr.update(visible=True),
                    class_name_display: "",
                    patch_gallery: [],
                    progress_html: "",
                    preexisting_concepts: "",
                    preexisting_dropdown: gr.update(choices=[], value=None),
                    current_set_concepts: "",
                    status_msg: "",
                    completion_summary: summary_text,
                    final_vocab_display: app.format_vocabulary_list(deduplicate=True),
                    nav_prev_btn: gr.update(interactive=True),
                    concept_input: ""
                }

            # Load next set of patches
            patches, class_name = app.get_random_patches_for_set(app.current_set, use_cache=True)

            return {
                builder_col: gr.update(visible=True),
                completion_col: gr.update(visible=False),
                class_name_display: f"## Vocabulary Builder (Set {app.current_set + 1}/{app.total_sets}) - **{class_name}**",
                patch_gallery: patches,
                progress_html: app.format_progress(),
                preexisting_concepts: app.format_preexisting_concepts(),
                preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                current_set_concepts: app.format_current_set_concepts(),
                status_msg: f"✓ Set {app.current_set} saved. Now reviewing set {app.current_set + 1}/{app.total_sets}",
                completion_summary: "",
                final_vocab_display: "",
                nav_prev_btn: gr.update(interactive=True),
                concept_input: ""
            }

        def clear_set():
            # Remove concepts added in current set
            if app.current_set in app.set_concepts:
                for concept in app.set_concepts[app.current_set]:
                    app.vocabulary.discard(concept)
                app.set_concepts[app.current_set] = []
                app.save_progress()

            return {
                current_set_concepts: "",
                status_msg: "⚠️ Concepts cleared for this set",
                progress_html: app.format_progress()
            }

        def nav_to_previous_class():
            """Navigate to previous class in builder"""
            new_idx = max(0, app.current_set - 1)
            app.current_set = new_idx
            patches, class_name = app.get_random_patches_for_set(new_idx, use_cache=True)

            # Disable previous button if at first class
            prev_btn_interactive = new_idx > 0

            return {
                class_name_display: f"## Vocabulary Builder (Set {new_idx + 1}/{app.total_sets}) - **{class_name}**",
                patch_gallery: patches,
                progress_html: app.format_progress(),
                preexisting_concepts: app.format_preexisting_concepts(),
                preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                current_set_concepts: app.format_current_set_concepts(),
                status_msg: f"📝 Now viewing Set {new_idx + 1}/{app.total_sets}",
                nav_prev_btn: gr.update(interactive=prev_btn_interactive)
            }

        def refresh_images():
            """Refresh the images for the current set"""
            patches, class_name = app.get_random_patches_for_set(app.current_set, use_cache=False)
            return {
                patch_gallery: patches,
                status_msg: "🔄 Images refreshed"
            }

        def go_to_class(class_idx):
            """Navigate back to builder to review/modify a specific class"""
            if class_idx < 0 or class_idx >= app.total_sets:
                class_idx = 0

            # Set app state to this class
            app.current_set = class_idx

            # Load patches for this class (use cache)
            patches, class_name = app.get_random_patches_for_set(class_idx, use_cache=True)

            # Disable previous button if at first class
            prev_btn_interactive = class_idx > 0

            return {
                completion_col: gr.update(visible=False),
                builder_col: gr.update(visible=True),
                class_name_display: f"## Vocabulary Builder (Set {class_idx + 1}/{app.total_sets}) - **{class_name}**",
                patch_gallery: patches,
                progress_html: app.format_progress(),
                preexisting_concepts: app.format_preexisting_concepts(),
                preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                current_set_concepts: app.format_current_set_concepts(),
                status_msg: f"📝 Reviewing Set {class_idx + 1}/{app.total_sets}. You can add more concepts or proceed to the next set.",
                nav_prev_btn: gr.update(interactive=prev_btn_interactive)
            }

        def review_previous_class():
            """Go to previous class in builder"""
            new_idx = max(0, app.current_set - 1)
            return go_to_class(new_idx)

        def review_next_class():
            """Go to next class in builder"""
            new_idx = min(app.total_sets - 1, app.current_set + 1)
            return go_to_class(new_idx)

        def export_vocabulary():
            app.export_final_vocabulary()
            return {
                final_status: """
✅ **Vocabulary submitted successfully!**

Thank you for completing the vocabulary builder.
"""
            }

        # === Wire up events ===

        start_btn.click(
            fn=start_session,
            outputs=[welcome_col, builder_col, completion_col, class_name_display,
                    patch_gallery, progress_html, preexisting_concepts, preexisting_dropdown,
                    current_set_concepts, completion_summary, final_vocab_display, nav_prev_btn]
        )

        add_btn.click(
            fn=add_concept_fn,
            inputs=[concept_input],
            outputs=[concept_input, status_msg, current_set_concepts, progress_html]
        )

        concept_input.submit(
            fn=add_concept_fn,
            inputs=[concept_input],
            outputs=[concept_input, status_msg, current_set_concepts, progress_html]
        )

        delete_preexisting_btn.click(
            fn=delete_preexisting_fn,
            inputs=[preexisting_dropdown],
            outputs=[status_msg, preexisting_concepts, preexisting_dropdown, progress_html]
        )

        clear_set_btn.click(
            fn=clear_set,
            outputs=[current_set_concepts, status_msg, progress_html]
        )

        nav_prev_btn.click(
            fn=nav_to_previous_class,
            outputs=[class_name_display, patch_gallery, progress_html, preexisting_concepts,
                    preexisting_dropdown, current_set_concepts, status_msg, nav_prev_btn]
        )

        refresh_images_btn.click(
            fn=refresh_images,
            outputs=[patch_gallery, status_msg]
        )

        save_next_event = save_next_btn.click(
            fn=save_and_next,
            inputs=[concept_input],
            outputs=[builder_col, completion_col, class_name_display, patch_gallery,
                    progress_html, preexisting_concepts, preexisting_dropdown, current_set_concepts,
                    status_msg, completion_summary, final_vocab_display, nav_prev_btn, concept_input]
        )

        # Run AFTER DOM updates, and only scroll if no warning (mirrors H1 behavior)
        save_next_event.then(
            None,
            inputs=[status_msg],
            js="""
        (status) => {
        // If status starts with the warning symbol, a validation error occurred -> don't scroll
        const hasError = typeof status === 'string' && status.trim().startsWith('⚠️');
        if (hasError) return;

        const top = document.getElementById('top-of-app');
        if (top && top.scrollIntoView) {
            top.setAttribute('tabindex', '-1');
            top.focus({preventScroll:true});
            top.scrollIntoView({behavior:'smooth', block:'start'});
        } else {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        }
        """
        )

        submit_btn.click(
            fn=export_vocabulary,
            outputs=[final_status]
        )

        # Review navigation in completion screen - go back to builder (previous class)
        review_prev_btn.click(
            fn=review_previous_class,
            outputs=[completion_col, builder_col, class_name_display, patch_gallery,
                    progress_html, preexisting_concepts, preexisting_dropdown, current_set_concepts,
                    status_msg, nav_prev_btn]
        )

        # Load initial view when page loads (handles welcome, builder, and completion states)
        def load_initial_view():
            if app.is_complete():
                summary_text = f"""
**Sets reviewed:** {len(app.completed_sets)}/{app.total_sets}

**Total unique concepts identified:** {len(app.vocabulary)}

**Total concepts (with duplicates):** {len(app.vocabulary_list)}

---

### Your Final Vocabulary:
"""
                return {
                    welcome_col: gr.update(visible=False),
                    builder_col: gr.update(visible=False),
                    completion_col: gr.update(visible=True),
                    class_name_display: "",
                    patch_gallery: [],
                    progress_html: "",
                    preexisting_concepts: "",
                    preexisting_dropdown: gr.update(choices=[], value=None),
                    current_set_concepts: "",
                    completion_summary: summary_text,
                    final_vocab_display: app.format_vocabulary_list(deduplicate=True),
                    nav_prev_btn: gr.update(interactive=True),
                    status_msg: ""
                }

            if app.has_progress():
                patches, class_name = app.get_random_patches_for_set(app.current_set, use_cache=True)
                prev_btn_interactive = app.current_set > 0

                return {
                    welcome_col: gr.update(visible=False),
                    builder_col: gr.update(visible=True),
                    completion_col: gr.update(visible=False),
                    class_name_display: f"## Vocabulary Builder (Set {app.current_set + 1}/{app.total_sets}) - **{class_name}**",
                    patch_gallery: patches,
                    progress_html: app.format_progress(),
                    preexisting_concepts: app.format_preexisting_concepts(),
                    preexisting_dropdown: gr.update(choices=app.get_active_preexisting_concepts(), value=None),
                    current_set_concepts: app.format_current_set_concepts(),
                    completion_summary: "",
                    final_vocab_display: "",
                    nav_prev_btn: gr.update(interactive=prev_btn_interactive),
                    status_msg: f"🔁 Resumed at Set {app.current_set + 1}/{app.total_sets}"
                }

            # Default to welcome screen
            return {
                welcome_col: gr.update(visible=True),
                builder_col: gr.update(visible=False),
                completion_col: gr.update(visible=False),
                class_name_display: "",
                patch_gallery: [],
                progress_html: "",
                preexisting_concepts: "",
                preexisting_dropdown: gr.update(choices=[], value=None),
                current_set_concepts: "",
                completion_summary: "",
                final_vocab_display: "",
                nav_prev_btn: gr.update(interactive=False),
                status_msg: ""
            }

        demo.load(
            fn=load_initial_view,
            outputs=[welcome_col, builder_col, completion_col, class_name_display, patch_gallery,
                     progress_html, preexisting_concepts, preexisting_dropdown, current_set_concepts,
                     completion_summary, final_vocab_display, nav_prev_btn, status_msg]
        )

    return demo


if __name__ == "__main__":
    import sys

    # Allow custom export directory and patches per set
    export_dir = sys.argv[1] if len(sys.argv) > 1 else "study-export/vocab-builder"
    patches_per_set = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    cache_dir = sys.argv[3] if len(sys.argv) > 3 else "cache"

    print(f"Starting Vocabulary Builder")
    print(f"Export directory: {export_dir}")
    print(f"Patches per set: {patches_per_set}")
    print(f"Cache directory: {cache_dir}")

    demo = create_interface(export_dir, patches_per_set, cache_dir)
    def find_available_port(start_port: int = 7862, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("", port))
                    return port
            except socket.error:
                continue
        raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

    # Allow optional command-line override for starting port
    parser = argparse.ArgumentParser(description="Vocabulary Builder App")
    parser.add_argument("export_dir", nargs="?", default=export_dir,
                        help="Export directory for vocab builder (default: study-export/vocab-builder)")
    parser.add_argument("--port-start", dest="port_start", type=int, default=7862,
                        help="Starting port to search for an available port (default: 7862)")
    args = parser.parse_args()

    # Find an available port starting from args.port_start
    try:
        available_port = find_available_port(args.port_start)
        print(f"✓ Using port: {available_port}")
    except RuntimeError as e:
        print(f"Error: {e}")
        raise

    demo.launch(
        server_name="0.0.0.0",
        server_port=available_port,
        share=True,
        show_error=True,
        show_api=False,
        allowed_paths=[str(Path(export_dir).resolve()), str(Path(cache_dir).resolve())]
    )
