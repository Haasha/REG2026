"""
Interface 0 — Visual Grounding (template implementation).

This is where you plug in your VQA / vision-language model.

Steps:
  1. Load your model weights (MODEL_PATH is available from core.py).
  2. Replace the body of predict_visual_context_response with your inference logic.
  3. If your model lives in a separate repository, add it under src/ and import it here:

       from src.your_repo.your_module import YourVQAModel

Platform rules for this interface:
  - If the ROI shows non-tissue background, the answer MUST be "not_informative".
    Returning any diagnostic claim for a background region is penalised.
  - For tissue ROIs, return a clinically grounded free-text answer.
  - The output is a plain JSON string — just the answer text, nothing else.
"""

from __future__ import annotations

from pathlib import Path

from core import load_json_file, load_jpg_image

# from src.your_repo.your_module import YourVQAModel   # <-- uncomment and adapt


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    """
    Run Visual Grounding inference for a single ROI.

    Args:
        question_path:  Path to visual-context-question.json (platform-fixed).
        roi_image_path: Path to histopathology-region-of-interest-thumbnail.jpg.

    Returns:
        A plain string answer (written as a JSON string by inference.py).
        Return "not_informative" if the ROI contains no tissue.

    IMPORTANT — do not change the return type (must stay ``str``). The platform
    expects a plain JSON string in visual-context-response.json; changing this
    will break submission validation.
    """
    # Load from paths — replace with your own pipeline if you prefer (paths are stable).
    question: str = load_json_file(location=question_path)
    roi_image = load_jpg_image(location=roi_image_path)

    # TODO: replace with your model inference, then remove the hardcoded return below.
    # Example skeleton:
    #   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   model = YourVQAModel().to(device)
    #   model.load_state_dict(torch.load(MODEL_PATH / "weights.pt", map_location=device))
    #   model.eval()
    #   answer = model.generate(image=roi_image, question=question)

    answer = "Yes"  # placeholder — remove this line once your model is implemented
    return answer
