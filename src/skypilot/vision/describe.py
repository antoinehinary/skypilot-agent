"""Open-ended questions about a camera frame, answered by a vision-language model.

Uses LiteLLM, like the agent, so a cloud vision model is a config change. Positions
should come from `detect.find_objects`; VLM answers are for qualitative questions.
"""

from __future__ import annotations

import base64

import cv2
import litellm

from skypilot.config import DescribeConfig
from skypilot.drone.backend import CameraFrame

CAMERA_CONTEXT = (
    "This image comes from a camera under a drone, looking straight down at the ground. "
    "The top of the image is the drone's forward direction and the right of the image is "
    "the drone's right. Answer briefly and only describe what is visible."
)


class VisionModelError(RuntimeError):
    """The vision model could not be reached or did not answer in time."""


def encode_png_data_url(image_rgb) -> str:
    """Encode an RGB image as a base64 PNG data URL."""
    ok, png = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode image as PNG")
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def describe_scene(frame: CameraFrame, question: str, config: DescribeConfig) -> str:
    """Ask the vision model `question` about `frame` and return its answer."""
    api_base = config.api_base if config.model_id.startswith("ollama") else None
    try:
        response = litellm.completion(
            model=config.model_id,
            api_base=api_base,
            messages=[
                {"role": "system", "content": CAMERA_CONTEXT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_png_data_url(frame.image)},
                        },
                    ],
                },
            ],
            max_tokens=config.max_tokens,
            timeout=config.request_timeout_s,
            temperature=0.0,
        )
    except (litellm.exceptions.APIConnectionError, litellm.exceptions.Timeout) as e:
        raise VisionModelError(
            f"vision model {config.model_id} unavailable ({type(e).__name__}); is Ollama running?"
        ) from e
    return (response.choices[0].message.content or "").strip()
