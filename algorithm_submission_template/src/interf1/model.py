"""
Interface 1 — Workflow Reasoning (template implementation).

This is where you plug in your chain-of-thought / reasoning model.

Steps:
  1. Load your model weights (MODEL_PATH is available from core.py).
  2. Replace the body of predict_chain_of_thought with your inference logic.
  3. If your model lives in a separate repository, add it under src/ and import it here:

       from src.your_repo.your_module import YourReasoningModel

Platform rules for this interface:
  - Use the EXACT canonical question / next_question strings from the training
    annotations. Minor whitespace/capitalisation differences are normalised, but
    paraphrasing is penalised.
  - Output is a bare JSON array of steps — do NOT wrap it in an object with an
    "id" key. The platform adds that wrapper automatically.
  - The last step must have "next_question": "".

Output schema per step:
  {
    "question":      "<exact canonical question string>",
    "answer":        "<your model's free-text answer>",
    "next_question": "<exact canonical next question string, or empty string>"
  }
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TypedDict

import tifffile

# from src.your_repo.your_module import YourReasoningModel   # <-- uncomment and adapt


class ChainOfThoughtStep(TypedDict):
    """One reasoning step — keys must match the platform schema exactly."""

    question: str
    answer: str
    next_question: str


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """
    Run Workflow Reasoning inference for a single whole-slide image.

    Args:
        wsi_path: Directory containing the WSI .tiff (platform-fixed).

    Returns:
        A list of steps, each with keys: question, answer, next_question.
        Use exact canonical question strings from the training annotations.

    IMPORTANT — do not change the return type (must stay
    ``list[ChainOfThoughtStep]``). inference.py serialises this list directly
    to chain-of-thought.json; wrapping it or changing field names will break
    submission validation.
    """
    # Load from wsi_path — swap for memmap, OpenSlide, cuCIM, etc. if needed.
    tiff_paths = sorted(glob.glob(str(wsi_path / "*.tiff")))
    if not tiff_paths:
        raise FileNotFoundError(f"No .tiff file found in {wsi_path}")
    wsi_array = tifffile.imread(tiff_paths[0])

    # TODO: replace placeholder return with model inference, e.g.:
    #   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   model = YourReasoningModel().to(device)
    #   model.load_state_dict(torch.load(MODEL_PATH / "weights.pt", map_location=device))
    #   model.eval()
    #   return model.reason(wsi_array)

    # placeholder — remove once your model is implemented
    chain_of_thought = [
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
            "next_question": "",
        },
    ]
    return chain_of_thought
