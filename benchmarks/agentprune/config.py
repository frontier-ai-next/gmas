import os
from pathlib import Path


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        message = f"Required environment variable {name} is not set"
        raise RuntimeError(message)
    return value


HERE = Path(__file__).resolve().parent
AGENTS_FILE = HERE / "agents.json"
TASKS_FILE = HERE / "tasks.json"
LOG_FILE = HERE / "log.csv"
API_KEY = _required_env("LLM_API_KEY")
BASE_URL = _required_env("LLM_BASE_URL")
MODEL = _required_env("LLM_MODEL")
PARSE_MAX_RETRIES = 3
CALL_MAX_RETRIES = 10
TIMEOUT = 300.0

# ~40K tokens without optimization
# the value multiplied by TOKEN_COEFF
# is approximately equal to 5
# the same scale with other scores
TOKEN_COEFF = 1 / 8000
# ~300 seconds without optimization
TIME_COEFF = 1 / 60
# before scaling score value is between -5 and 15 approximately
SCORE_COEFF = 1 / 10

PRINT_LEN = 72

NUM_PRUNING = 14

RL_NUM_ITERATIONS = 3
MC_NUM_ITERATIONS = 5
PRUNE_RATIO = 0.8
