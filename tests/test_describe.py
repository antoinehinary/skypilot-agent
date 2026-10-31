"""describe_scene with the LLM call replaced: checks the request, not the model."""

import base64
from types import SimpleNamespace

import cv2
import litellm
import numpy as np
import pytest

from skypilot.config import load_vision_config
from skypilot.drone.backend import CameraFrame
from skypilot.vision import describe
from skypilot.vision.describe import VisionModelError, describe_scene

CONFIG = load_vision_config().describe
FRAME = CameraFrame(np.full((24, 32, 3), (200, 30, 30), np.uint8), np.eye(3), np.eye(4), 0.0)


def _reply(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_sends_question_and_image_and_returns_answer(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        describe.litellm, "completion", lambda **kw: calls.append(kw) or _reply(" A red box. ")
    )

    answer = describe_scene(FRAME, "What do you see?", CONFIG)

    assert answer == "A red box."
    request = calls[0]
    assert request["model"] == CONFIG.model_id
    assert request["api_base"] == CONFIG.api_base
    user_content = request["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "What do you see?"}
    url = user_content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    png = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
    decoded = cv2.cvtColor(cv2.imdecode(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded, FRAME.image)  # lossless, colours not swapped


def test_unreachable_model_gives_a_readable_error(monkeypatch) -> None:
    def refuse(**_):
        raise litellm.exceptions.APIConnectionError(
            message="connection refused", llm_provider="ollama", model=CONFIG.model_id
        )

    monkeypatch.setattr(describe.litellm, "completion", refuse)

    with pytest.raises(VisionModelError, match="is Ollama running"):
        describe_scene(FRAME, "anything?", CONFIG)
