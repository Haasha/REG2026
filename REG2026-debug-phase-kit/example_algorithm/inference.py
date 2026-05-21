"""
REG2026 Challenge - Example Submission Algorithm
=================================================

This starter kit implements the two input-output interfaces required for all
REG2026 challenge phases (Debug, Test 1, Test 2).

  Interface 0 - Visual Grounding (Metric B)
  ------------------------------------------
  Inputs:
    * histopathology-region-of-interest-thumbnail  (JPEG image)
    * visual-context-question                      (JSON string)
  Output:
    * visual-context-response.json                 (JSON string - the answer only)

  Interface 1 - Workflow Reasoning (Metric A)
  --------------------------------------------
  Input:
    * whole-slide-image  (medical image file, e.g. .mha / .tif)
  Output:
    * chain-of-thought.json  (JSON array of {question, answer, next_question} steps)

Notes
-----
* The platform determines which interface to run by reading /input/inputs.json.
* For Interface 0, background (non-tissue) ROIs must receive the answer
  "not_informative" - returning any diagnostic claim for such regions is penalised.
* For Interface 1, use the EXACT canonical question / next_question strings from
  the training annotations. Minor whitespace / capitalisation differences are
  normalised, but paraphrasing is penalised.
* Model weights should be uploaded to Grand Challenge under Algorithm > Models as
  a tarball. At runtime they are available at /opt/ml/model.

To test locally:
  ./do_test_run.sh

To build and save for upload:
  ./do_save.sh

Reference:
  https://grand-challenge.org/documentation/runtime-environment/
"""

import glob
import json
from pathlib import Path

import SimpleITK
import torch
from PIL import Image

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")


# =============================================================================
# Entry point
# =============================================================================

def run():
    # Determine which interface is active for this job
    interface_key = get_interface_key()

    handler = {
        (
            "histopathology-region-of-interest-thumbnail",
            "visual-context-question",
        ): interf0_handler,
        ("whole-slide-image",): interf1_handler,
    }[interface_key]

    return handler()


# =============================================================================
# Interface handlers
# =============================================================================

def interf0_handler():
    """
    Visual Grounding - Metric B
    Reads an ROI thumbnail and a free-text question, then writes a single
    string answer to visual-context-response.json.
    """
    # --- Read inputs ---------------------------------------------------------
    question = load_json_file(
        location=INPUT_PATH / "visual-context-question.json",
    )
    roi_image = load_jpeg_image(
        location=INPUT_PATH / "histopathology-region-of-interest-thumbnail.jpeg",
    )

    print(f"[interf0] Question  : {question}")
    print(f"[interf0] ROI size  : {roi_image.size}  mode: {roi_image.mode}")

    _show_torch_cuda_info()

    # --- Load model ----------------------------------------------------------
    # Replace the block below with your own model loading logic.
    # At runtime on Grand Challenge your uploaded weights will be at MODEL_PATH.
    # Example:
    #   model = YourModel()
    #   model.load_state_dict(torch.load(MODEL_PATH / "weights.pt"))
    #   model.eval()

    # --- Run inference -------------------------------------------------------
    # TODO: replace with your model actual inference.
    # * For non-tissue (background) ROIs the correct answer is "not_informative".
    # * For tissue ROIs return a clinically grounded free-text answer.
    answer = predict_visual_context_response(roi_image=roi_image, question=question)

    # --- Write output --------------------------------------------------------
    # Output is a plain JSON string - just the answer text, nothing else.
    write_json_file(
        location=OUTPUT_PATH / "visual-context-response.json",
        content=answer,
    )
    print(f"[interf0] Answer written: {answer}")
    return 0


def interf1_handler():
    """
    Workflow Reasoning - Metric A
    Reads a Whole Slide Image and writes a structured chain-of-thought JSON
    array of {question, answer, next_question} steps.
    Do NOT wrap the list in an object with an "id" key - the platform adds that.
    """
    # --- Read input ----------------------------------------------------------
    wsi_array = load_image_file_as_array(
        location=INPUT_PATH / "images/whole-slide-image",
    )
    print(f"[interf1] WSI shape : {wsi_array.shape}  dtype: {wsi_array.dtype}")

    _show_torch_cuda_info()

    # --- Load model ----------------------------------------------------------
    # Replace with your own model loading logic (see interf0_handler for hints).

    # --- Run inference -------------------------------------------------------
    # TODO: replace with your model actual inference.
    # Use the EXACT canonical question / next_question strings from the training
    # dataset annotations. Do not paraphrase them.
    chain_of_thought = predict_chain_of_thought(wsi_array=wsi_array)

    # --- Write output --------------------------------------------------------
    # Output is a JSON array of steps - no "id" wrapper.
    # Each step: {"question": "...", "answer": "...", "next_question": "..."}
    write_json_file(
        location=OUTPUT_PATH / "chain-of-thought.json",
        content=chain_of_thought,
    )
    print(f"[interf1] Chain-of-thought written ({len(chain_of_thought)} steps)")
    return 0


# =============================================================================
# Dummy inference - replace with your actual model
# =============================================================================

def predict_visual_context_response(*, roi_image, question):
    """
    Dummy Visual Grounding prediction.

    Returns "not_informative" as a safe baseline placeholder.
    Replace this function body with your VQA / vision-language model inference.
    For non-tissue background regions the correct answer IS "not_informative".
    """
    # TODO: implement real inference
    return "not_informative"


def predict_chain_of_thought(*, wsi_array):
    """
    Dummy Workflow Reasoning prediction.

    Returns a minimal chain-of-thought with placeholder text.
    Replace this function body with your reasoning model inference.
    Use the EXACT canonical question strings from the training annotations.
    """
    # TODO: implement real inference - use canonical questions from training data
    return [
        {
            "question": "What type of specimen is this?",
            "answer": "The specimen is a surgically resected tissue section prepared for histopathological examination.",
            "next_question": "What is the predominant tissue architecture observed in this specimen?",
        },
        {
            "question": "What is the predominant tissue architecture observed in this specimen?",
            "answer": "The tissue shows predominantly glandular structures embedded within a fibrous stroma.",
            "next_question": "Are there morphological features suggestive of malignancy?",
        },
        {
            "question": "Are there morphological features suggestive of malignancy?",
            "answer": "There are features including nuclear pleomorphism, prominent nucleoli, and increased mitotic figures.",
            "next_question": "null",
        },
    ]


# =============================================================================
# I/O helpers
# =============================================================================

def get_interface_key():
    """
    Reads /input/inputs.json (injected by the platform) to determine which
    set of input sockets is active for this job.
    """
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_jpeg_image(*, location):
    """Load a JPEG/PNG thumbnail as a PIL Image (RGB)."""
    return Image.open(location).convert("RGB")


def load_image_file_as_array(*, location):
    """Load a medical image (.mha / .tif / .tiff) via SimpleITK -> NumPy array."""
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    if not input_files:
        raise FileNotFoundError(f"No supported image file found in {location}")
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Torch CUDA available:", (available := torch.cuda.is_available()))
    if available:
        print(f"  devices          : {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        print(f"  current device   : {current}")
        print(f"  device properties: {torch.cuda.get_device_properties(current)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
