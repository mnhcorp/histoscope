#!/usr/bin/env python3
"""
H1 Interpretability Study App

Gradio interface for pathologists to rate neuron monosemanticity
by viewing top-activating patches.
"""

import gradio as gr
import json
import socket
from pathlib import Path
from typing import Dict, List
from copy import deepcopy

def find_available_port(start_port: int = 7860, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', port))
                return port
        except socket.error:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

def load_vocabulary(vocab_path: str = "study-export/vocab-builder/vocabulary-pathologist-validated.json") -> List[str]:
    """Load unique vocabulary from vocabulary.json (concepts only, no class labels)"""
    try:
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        unique_vocab = vocab_data.get('vocabulary_unique', [])
        # Return only unique vocab concepts + Mixed + Other (no class labels)
        return ["Mixed", "Other"] + unique_vocab
    except Exception as e:
        print(f"Warning: Could not load vocabulary from {vocab_path}: {e}")
        # Fallback to just Mixed and Other
        return ["Mixed", "Other"]

# Load the full controlled vocabulary (concepts only)
CONTROLLED_VOCABULARY = load_vocabulary()

class H1StudyApp:
    def __init__(self, export_dir: str, neuron_limit: float = None, username: str = None):
        self.export_dir = Path(export_dir)
        self.metadata = self.load_metadata()
        self.neuron_limit = neuron_limit
        self.username = username if username else "nouser"
        self.responses: Dict[int, Dict] = {}
        self.neurons: List[Dict] = []
        self.current_idx = 0
        self.next_index = 0
        self.refresh_state_from_disk()

    def load_metadata(self) -> Dict:
        """Load study metadata"""
        with open(self.export_dir / "metadata.json") as f:
            return json.load(f)

    def load_responses(self) -> tuple:
        """Load existing responses and saved neuron order if available.
        Returns (responses_dict, neuron_order_list_or_None, next_index_or_None)
        """
        response_file = self.export_dir / f"h1_{self.username}_responses.json"
        if response_file.exists():
            with open(response_file) as f:
                data = json.load(f)
                # Convert string keys back to integers
                responses = {int(k): v for k, v in data.get('responses', {}).items()}
                neuron_order = data.get('neuron_order', None)
                next_index = data.get('next_index')  # may be None
                if isinstance(next_index, str):
                    try:
                        next_index = int(next_index)
                    except ValueError:
                        next_index = None
                return responses, neuron_order, next_index
        return {}, None, None

    def refresh_state_from_disk(self):
        """Reload responses/order pointer from disk and update current index."""
        responses, saved_order, saved_next_index = self.load_responses()
        self.responses = responses
        self.neurons = self.prepare_neurons(saved_order)
        if not self.neurons:
            self.current_idx = 0
            self.next_index = 0
            return
        if saved_next_index is not None and 0 <= saved_next_index < len(self.neurons):
            start_idx = saved_next_index
        else:
            start_idx = self.find_first_unrated_neuron()
        self.set_current_index(start_idx, persist=False)

    def set_current_index(self, idx: int, persist: bool = True):
        """Update current and next indices (optionally persisting state)."""
        if not self.neurons:
            self.current_idx = 0
            self.next_index = 0
            return
        safe_idx = max(0, min(idx, len(self.neurons) - 1))
        self.current_idx = safe_idx
        self.next_index = safe_idx
        if persist:
            # Persist current progress without mutating responses
            self.export_responses()

    def find_first_unrated_neuron(self) -> int:
        """Find the first neuron that hasn't been rated yet"""
        for idx, neuron in enumerate(self.neurons):
            if neuron['neuron_id'] not in self.responses:
                return idx
        return 0  # If all rated, start from beginning

    def prepare_neurons(self, saved_order=None) -> List[Dict]:
        """Combine mono and poly neurons into single list and randomize order"""
        import random

        neurons = []

        # Add monosemantic neurons
        for neuron in self.metadata['neurons']['monosemantic']:
            neuron_copy = deepcopy(neuron)
            neuron_copy['ground_truth'] = 'monosemantic'
            neurons.append(neuron_copy)

        # Add polysemantic neurons
        for neuron in self.metadata['neurons']['polysemantic']:
            neuron_copy = deepcopy(neuron)
            neuron_copy['ground_truth'] = 'polysemantic'
            neurons.append(neuron_copy)

        # If we have a saved order, use it to maintain consistency
        if saved_order:
            # Create a dict for quick lookup
            neuron_dict = {n['neuron_id']: n for n in neurons}
            # Reorder according to saved order
            ordered = [neuron_dict[nid] for nid in saved_order if nid in neuron_dict]
            # Append any neurons that may be new/not in saved order
            saved_set = set(saved_order)
            ordered.extend([n for n in neurons if n['neuron_id'] not in saved_set])
            neurons = ordered
        else:
            # First time: randomize order to mix mono and poly neurons
            random.seed(42)  # Fixed seed for reproducibility across sessions
            random.shuffle(neurons)

        # Apply limit if specified (for debugging)
        if self.neuron_limit is not None:
            if self.neuron_limit < 1:
                # Treat as fraction
                limit_count = int(len(neurons) * self.neuron_limit)
            else:
                # Treat as absolute count
                limit_count = int(self.neuron_limit)
            neurons = neurons[:limit_count]
            print(f"🔍 Debug mode: Limited to {len(neurons)} neurons (limit={self.neuron_limit})")

        return neurons

    def get_current_neuron(self) -> Dict:
        """Get current neuron data"""
        return self.neurons[self.current_idx]

    def get_patch_paths(self, neuron: Dict) -> List[str]:
        """Get full paths to patch images"""
        patches = []
        for patch_info in neuron['patches']['patches']:
            patch_path = self.export_dir / patch_info['exported_path']
            patches.append(str(patch_path))
        return patches

    def format_neuron_info(self, neuron: Dict) -> str:
        """Format neuron metadata for display"""
        # Display neuron number (1-indexed) instead of actual SAE index for blinded study
        neuron_number = self.current_idx + 1
        info = f"""
## 🔬 Patch Set #{neuron_number}
"""
        return info

    def format_progress_bar(self) -> str:
        """Render progress as an HTML progress bar"""
        completed = len(self.responses)
        total = len(self.neurons)
        total = max(1, total)
        pct = (completed / total) * 100.0
        return f"""
<div style="margin-bottom: 0.5rem;">
  <span style="font-size: 0.9rem; font-weight: 600; color: #666;">Progress {completed}/{total}</span>
</div>
<div class="progress-container" role="progressbar" aria-valuenow="{completed}" aria-valuemin="0" aria-valuemax="{total}" aria-valuetext="{completed} of {total} neurons rated">
  <div class="progress-fill" style="width: {pct:.1f}%"></div>
</div>
"""

    def save_response(self, neuron_id: int, morphology: List[str], morphology_other: str,
                      monosemantic_rating: int, diagnostic_relevance: int, notes: str):
        """Save pathologist's response"""
        neuron = self.get_current_neuron()
        self.responses[neuron_id] = {
            'morphology': morphology,  # Now a list of selected morphologies
            'morphology_other': morphology_other if 'Other' in (morphology or []) else None,
            'monosemantic_rating': monosemantic_rating,
            'diagnostic_relevance': diagnostic_relevance,
            'notes': notes,
            'ground_truth': neuron['ground_truth'],
            'is_overlap': bool(neuron.get('is_overlap', False)),
            'overlap_source_session': neuron.get('overlap_source_session')
        }
        # Compute next index (point to next unrated or stay at last)
        if self.current_idx < len(self.neurons) - 1:
            self.next_index = self.current_idx + 1
        else:
            self.next_index = self.current_idx
        # Auto-save after each response including next_index
        self.export_responses()

    def export_responses(self) -> str:
        """Export all responses to JSON"""
        output_file = self.export_dir / f"h1_{self.username}_responses.json"

        # Save neuron order to ensure consistency across sessions
        neuron_order = [n['neuron_id'] for n in self.neurons]

        export_data = {
            'study': 'h1_interpretability',
            'username': self.username,
            'session': self.metadata.get('session'),
            'total_neurons': len(self.neurons),
            'completed': len(self.responses),
            'overlap_total': sum(1 for n in self.neurons if n.get('is_overlap')),
            'overlap_completed': sum(1 for r in self.responses.values() if r.get('is_overlap')),
            'neuron_order': neuron_order,  # Save the order
            'next_index': getattr(self, 'next_index', 0),  # Where to resume
            'responses': self.responses
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        return f"✓ Responses saved to {output_file}"


def create_interface(export_dir: str = "study-export/h1", neuron_limit: float = None, username: str = None):
    """Create Gradio interface"""

    app = H1StudyApp(export_dir, neuron_limit=neuron_limit, username=username)
    is_session_b = app.metadata.get('session') == 'session_b'
    study_title = "UNI-SAE Interpretability Study: H1 (Session B)" if is_session_b else "UNI-SAE Interpretability Study: H1"

    # Custom CSS for better styling
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
    .instruction-box p strong {
        font-size: 1.15em;
        font-weight: 700;
    }
    /* Increase font sizes for all interactive elements */
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
    /* Keep headers at their proper large sizes */
    .gradio-container h1 {
        font-size: 2em !important;
    }
    .gradio-container h2 {
        font-size: 1.5em !important;
    }
    /* Hide footer */
    footer {
        display: none !important;
    }
    /* Hide fullscreen button in Gallery preview modal */
    .gallery-item button[aria-label*="Fullscreen"],
    .gallery-item button[aria-label*="fullscreen"],
    .preview-controls button:nth-child(2),
    button[title*="Fullscreen"],
    button[title*="fullscreen"] {
        display: none !important;
    }
    """

    completion_modal_html = """
        <div id="completion-modal" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        ">
            <div style="
                background: white;
                border-radius: 12px;
                padding: 2.5rem;
                max-width: 500px;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            ">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎉</div>
                <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem; color: #1f6feb;">All Patch Sets Rated!</h2>
                <p style="margin: 0 0 1.5rem 0; font-size: 1.1rem; color: #57606a;">
                    You have successfully completed the H1 Interpretability Study.
                    <br><br>
                    Your responses have been saved. Thank you for your contribution! You can now close this tab.
                </p>
            </div>
        </div>
        <script>
            const completionModal = document.getElementById('completion-modal');
            if (completionModal) {
                completionModal.style.display = 'flex';
            }
        </script>
    """

    with gr.Blocks(title=study_title, theme=gr.themes.Monochrome(), css=custom_css) as demo:

        # Completion modal (hidden by default, no close button - final state)
        completion_modal = gr.HTML(visible=False, value="")

        # Header
        header = gr.Markdown(f"# 🔬 {study_title}")

        # Instructions - collapsible and collapsed by default
        with gr.Accordion("📋 Instructions", open=True):
            gr.Markdown("""
**Task:**
Identify the main **morphological concepts/patterns** in the patches shown, and evaluate how **consistent** and **clinically relevant** they are.

1. **Morphological Concepts/Patterns:**
   Select one or more concepts/patterns that describe the **main recurring morphology** in these patches.
   - Use **“Mixed”** if the patches show **unrelated** concepts/patterns.
   - Use **“Other”** if you need to describe a concept/pattern not in the list.

2. **Consistency (1–5):**
   How consistently does the **same concept or a tightly related set of concepts** appear across the patches?
   - **5** = Very consistent
   - **3** = Moderately consistent
   - **1** = Very mixed or diffuse

3. **Diagnostic Relevance (1–5):**
   How useful is this concept/theme in **real diagnostic reasoning**?
   - **5** = Highly relevant (key diagnostic hallmark)
   - **3** = Moderately relevant (supportive/contextual)
   - **1** = Not diagnostic (background or artifact)

4. **Notes (Optional):**
   Add any clarifications, uncertainties, or comments you think are helpful.

---

**Use the patches below to make your assessment.**

            """)

        # Get the initial neuron (handles resume from saved state)
        initial_neuron = app.get_current_neuron()
        initial_response = app.responses.get(initial_neuron['neuron_id'])

        # Neuron info (placeholder updated on load)
        neuron_info = gr.Markdown(app.format_neuron_info(initial_neuron), elem_id="top-of-app")
        progress_display = gr.HTML(app.format_progress_bar())

        # Image gallery - always show both rows without scrolling
        with gr.Row():
            patch_gallery = gr.Gallery(
                #label="🖼️ Top 12 Activating Patches (click to enlarge)",
                columns=6,
                rows=2,
                height="auto",
                object_fit="contain",
                show_label=True,
                value=app.get_patch_paths(initial_neuron)
            )

        # Rating interface
        gr.HTML('<div class="rating-section">')

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Morphology Classification")
                gr.Markdown("*Select one or more morphologies that describe these patches*")

                # Pre-populate if already rated
                initial_morph = initial_response['morphology'] if initial_response else []
                # Ensure it's a list for multi-select
                if isinstance(initial_morph, str):
                    initial_morph = [initial_morph] if initial_morph else []

                morphology = gr.Dropdown(
                    choices=CONTROLLED_VOCABULARY,
                    label="Select tissue/morphology type(s)",
                    value=initial_morph,
                    multiselect=True,
                    interactive=True
                )

                initial_morph_other = initial_response.get('morphology_other', "") if initial_response else ""
                morphology_other = gr.Textbox(
                    label="If 'Other' is selected, please specify all other morphologies:",
                    placeholder="Describe any morphologies not in the vocabulary...",
                    visible=("Other" in initial_morph),
                    value=initial_morph_other,
                    lines=2
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 2️⃣ Concept Consistency Rating")
                gr.Markdown("*How specific and focused is this pattern?*")

                initial_mono = str(initial_response['monosemantic_rating']) if initial_response else None
                monosemantic_rating = gr.Radio(
                    choices=["1", "2", "3", "4", "5"],
                    label="",
                    value=initial_mono,
                    info="1 = Very broad/diffuse (many unrelated concepts) | 5 = Very specific (single concept or tightly related concepts)"
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 3️⃣ Diagnostic Relevance")
                gr.Markdown("*How valuable are the identified concepts for clinical or diagnostic purposes?*")

                initial_diag = str(initial_response['diagnostic_relevance']) if initial_response else None
                diagnostic_relevance = gr.Radio(
                    choices=["1", "2", "3", "4", "5"],
                    label="",
                    value=initial_diag,
                    info="1 = Not valuable (background/artifact features) | 5 = Highly valuable (key diagnostic features)"
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Notes (Optional)")

                initial_notes = initial_response['notes'] if initial_response else ""
                notes = gr.Textbox(
                    label="Additional observations or comments",
                    placeholder="Any observations about this patch set...",
                    value=initial_notes,
                    lines=3
                )

        gr.HTML('</div>')

        # Status message
        initial_status = f"ℹ️ You've already rated this patch set (Morphology: {initial_response['morphology']})" if initial_response else ""
        status_msg = gr.Markdown(initial_status)

        # Navigation buttons
        with gr.Row():
            initial_prev_enabled = app.current_idx > 0
            prev_btn = gr.Button("← Previous", variant="secondary", interactive=initial_prev_enabled)
            submit_btn = gr.Button("Submit & Next →", variant="primary", size="lg")


        # === Callback Functions ===

        def toggle_other_textbox(morphology_val):
            """Show/hide 'Other' textbox based on dropdown selection"""
            if isinstance(morphology_val, list):
                return gr.update(visible=("Other" in morphology_val))
            return gr.update(visible=(morphology_val == "Other"))

        def load_neuron(idx: int, persist: bool = True):
            """Load neuron at given index"""
            if 0 <= idx < len(app.neurons):
                app.set_current_index(idx, persist=persist)
                neuron = app.get_current_neuron()

                # Check if already rated
                existing_response = app.responses.get(neuron['neuron_id'])

                if existing_response:
                    morph_val = existing_response['morphology']
                    # Ensure morphology is a list for multi-select
                    if isinstance(morph_val, str):
                        morph_val = [morph_val] if morph_val else []

                    mono_val = existing_response['monosemantic_rating']
                    diag_val = existing_response['diagnostic_relevance']

                    # Convert numeric ratings back to string format (now just "1", "2", etc.)
                    mono_label = str(mono_val)
                    diag_label = str(diag_val)

                    morph_display = ", ".join(morph_val) if morph_val else "None"

                    return (
                        app.format_neuron_info(neuron),
                        app.format_progress_bar(),
                        app.get_patch_paths(neuron),
                        morph_val,
                        existing_response.get('morphology_other', ""),
                        gr.update(visible=("Other" in morph_val)),
                        mono_label,
                        diag_label,
                        existing_response['notes'],
                        f"ℹ️ You've already rated this patch set (Morphology: {morph_display})",
                        gr.update(interactive=(idx > 0)),  # Enable Previous if not first neuron
                        gr.update(value="", visible=False)   # Hide completion modal
                    )
                else:
                    return (
                        app.format_neuron_info(neuron),
                        app.format_progress_bar(),
                        app.get_patch_paths(neuron),
                        [],  # Empty list for multi-select
                        "",
                        gr.update(visible=False),
                        None,
                        None,
                        "",
                        "",
                        gr.update(interactive=(idx > 0)),  # Enable Previous if not first neuron
                        gr.update(value="", visible=False)   # Hide completion modal
                    )
            return None

        def submit_and_next(morph_val, morph_other_val, mono_val, diag_val, notes_val):
            """Save response and move to next neuron"""
            def _apply_form_state(outputs_to_update):
                current_morph = morph_val if isinstance(morph_val, list) else ([morph_val] if morph_val else [])
                outputs_to_update[3] = current_morph
                outputs_to_update[4] = morph_other_val
                outputs_to_update[5] = gr.update(visible=("Other" in current_morph))
                outputs_to_update[6] = mono_val
                outputs_to_update[7] = diag_val
                outputs_to_update[8] = notes_val

            # Validation
            if not morph_val or len(morph_val) == 0:
                outputs = list(load_neuron(app.current_idx, persist=False))
                outputs[9] = "⚠️ Please select at least one morphology type"
                _apply_form_state(outputs)
                outputs[11] = gr.update(value="", visible=False)
                return tuple(outputs)

            if "Other" in (morph_val or []) and not (morph_other_val or "").strip():
                outputs = list(load_neuron(app.current_idx, persist=False))
                outputs[9] = "⚠️ Please specify the morphology type in the text box"
                _apply_form_state(outputs)
                outputs[11] = gr.update(value="", visible=False)
                return tuple(outputs)

            if mono_val is None:
                outputs = list(load_neuron(app.current_idx, persist=False))
                outputs[9] = "⚠️ Please provide a concept consistency rating"
                _apply_form_state(outputs)
                outputs[11] = gr.update(value="", visible=False)
                return tuple(outputs)

            if diag_val is None:
                outputs = list(load_neuron(app.current_idx, persist=False))
                outputs[9] = "⚠️ Please provide a diagnostic relevance rating"
                _apply_form_state(outputs)
                outputs[11] = gr.update(value="", visible=False)
                return tuple(outputs)

            # Save response
            mono_rating = int(mono_val)
            diag_rating = int(diag_val)

            neuron = app.get_current_neuron()
            app.save_response(neuron['neuron_id'], morph_val, morph_other_val,
                            mono_rating, diag_rating, notes_val)

            # Check if this is the last neuron - show completion modal
            is_last_neuron = (app.current_idx == len(app.neurons) - 1)

            if is_last_neuron:
                # Show completion modal
                outputs = list(load_neuron(app.current_idx, persist=False))
                outputs[9] = ""  # Clear status
                outputs[11] = gr.update(value=completion_modal_html, visible=True)
                return tuple(outputs)

            # Not the last neuron - move to next
            next_outputs = list(load_neuron(app.current_idx + 1))
            next_outputs[11] = gr.update(value="", visible=False)
            return tuple(next_outputs)

        def go_previous():
            """Go to previous neuron"""
            if app.current_idx > 0:
                prev_outputs = list(load_neuron(app.current_idx - 1))
                prev_outputs[11] = gr.update(value="", visible=False)
                return tuple(prev_outputs)
            outputs = list(load_neuron(app.current_idx, persist=False))
            outputs[9] = "⚠️ Already at first neuron"
            outputs[11] = gr.update(value="", visible=False)
            return tuple(outputs)

        def initialize_session():
            """Ensure UI reflects latest saved progress when a client connects."""
            app.refresh_state_from_disk()
            # Check if all neurons are rated - if so, show completion modal
            all_rated = len(app.responses) == len(app.neurons)
            outputs = list(load_neuron(app.current_idx))
            outputs[11] = gr.update(
                value=completion_modal_html if all_rated else "",
                visible=all_rated
            )
            return tuple(outputs)

        # === Wire up events ===

        # Show/hide 'Other' textbox
        morphology.change(
            fn=toggle_other_textbox,
            inputs=morphology,
            outputs=morphology_other
        )

        output_components = [
            neuron_info, progress_display, patch_gallery, morphology, morphology_other, morphology_other,
            monosemantic_rating, diagnostic_relevance, notes, status_msg, prev_btn,
            completion_modal
        ]

        event = submit_btn.click(
            fn=submit_and_next,
            inputs=[morphology, morphology_other, monosemantic_rating, diagnostic_relevance, notes],
            outputs=output_components
        )

        # Run AFTER DOM updates, and only scroll if no warning
        event.then(
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

        prev_btn.click(
            fn=go_previous,
            inputs=None,
            outputs=output_components
        ).then(
            None,
            js="() => document.getElementById('top-of-app')?.scrollIntoView({behavior:'smooth', block:'start'})"
        )

        demo.load(
            fn=initialize_session,
            inputs=None,
            outputs=output_components
        )

    return demo


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="H1 Neuron Interpretability Study")
    parser.add_argument("export_dir", nargs="?", default="study-export/h1",
                        help="Path to study export directory (default: study-export/h1)")
    parser.add_argument("--session-b", action="store_true",
                        help="Load Session B from study-export/h1/session-B")
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit to a fraction of neurons for debugging (e.g., 0.66 for 2/3, or integer for exact count)")
    parser.add_argument("--username", type=str, default=None,
                        help="Username for tagging responses (default: check .user file or nouser)")
    args = parser.parse_args()

    export_dir = args.export_dir
    if args.session_b:
        export_dir = "study-export/h1/session-B"

    # Check for .user file in current directory if no username provided
    if args.username is None:
        user_file = Path(".user")
        if user_file.exists():
            args.username = user_file.read_text().strip()
            print(f"📝 Found .user file, using username: {args.username}")

    if export_dir == "study-export/h1" and len(sys.argv) == 1:
        print("Usage: python h1_study_app.py [--session-b] [--limit FRACTION]")
        print("Example: python h1_study_app.py")
        print("Example: python h1_study_app.py --session-b")
        print("Example: python h1_study_app.py study-export/h1 --limit 0.66  # Review 2/3 of neurons")
        print("Example: python h1_study_app.py study-export/h1/session-B")
        print("Example: python h1_study_app.py study-export/h1 --limit 5  # Review only 5 neurons")
        print("\nUsing default: study-export/h1")

    # Validate directory exists
    export_path = Path(export_dir)
    if not export_path.exists():
        print(f"Error: Directory not found: {export_dir}")
        sys.exit(1)

    metadata_file = export_path / "metadata.json"
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {export_dir}")
        print("Make sure this is a valid H1 study export directory.")
        sys.exit(1)

    username_display = args.username if args.username else "nouser"
    print(f"✓ Loading H1 study from: {export_dir}")
    print(f"✓ Username: {username_display}")
    print(f"✓ Responses will be saved to: {export_dir}/h1_{username_display}_responses.json")
    demo = create_interface(export_dir, neuron_limit=args.limit, username=args.username)

    # Resolve absolute path for allowed_paths
    export_abs = Path(export_dir).resolve()

    # Find an available port starting from 7860
    try:
        available_port = find_available_port(7860)
        print(f"✓ Using port: {available_port}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=available_port,
        share=True,
        show_error=True,
        allowed_paths=[str(export_abs)]
    )
