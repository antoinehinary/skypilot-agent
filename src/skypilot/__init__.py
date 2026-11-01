"""SkyPilot Agent: plain-language missions for a simulated Crazyflie."""

import os

# NOTE: LiteLLM otherwise downloads a model price list from GitHub every time it is imported.
# The project runs locally and offline, so use the copy bundled with LiteLLM (overridable).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "1.0.0"
