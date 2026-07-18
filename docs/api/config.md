# Config API

## FrameworkSettings

Pydantic-settings based configuration with environment variable support.

```python
from gmas.config import FrameworkSettings

settings = FrameworkSettings()
```

### Settings Fields

| Field | Type | Env Variable | Default | Description |
|-------|------|-------------|---------|-------------|
| `api_key` | `str \| None` | `GMAS_API_KEY` | `None` | API key for LLM provider |
| `api_key_file` | `str \| None` | `GMAS_API_KEY_FILE` | `None` | Path to API key file |
| `base_url` | `str \| None` | `GMAS_BASE_URL` | `None` | API base URL |
| `model` | `str` | `GMAS_MODEL` | `"gpt-4"` | Default model name |
| `temperature` | `float` | `GMAS_TEMPERATURE` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `GMAS_MAX_TOKENS` | `1000` | Max tokens per request |
| `timeout` | `float` | `GMAS_TIMEOUT` | `30.0` | Request timeout (seconds) |
| `max_retries` | `int` | `GMAS_MAX_RETRIES` | `3` | Max retry attempts |
| `retry_delay` | `float` | `GMAS_RETRY_DELAY` | `1.0` | Retry delay (seconds) |
| `log_level` | `str` | `GMAS_LOG_LEVEL` | `"INFO"` | Logging level |

### resolved_api_key

The `resolved_api_key` property resolves the API key in this order:

1. Direct `api_key` value
2. Content of the file at `api_key_file`
3. Raises `RuntimeError` if neither is set and a key is required

```python
settings = FrameworkSettings()
api_key = settings.resolved_api_key  # str or RuntimeError
```

## Environment Variables

```bash
# API Configuration
export GMAS_API_KEY="sk-..."
export GMAS_BASE_URL="https://api.provider.example"
export GMAS_API_KEY_FILE=/path/to/keyfile

# LLM Configuration
export GMAS_MODEL="gpt-4"
export GMAS_TEMPERATURE="0.7"
export GMAS_MAX_TOKENS="1000"

# Request Configuration
export GMAS_TIMEOUT="30"
export GMAS_MAX_RETRIES="3"
export GMAS_RETRY_DELAY="1"

# Logging
export GMAS_LOG_LEVEL="INFO"
```

## Settings Usage

```python
from gmas.config import FrameworkSettings
from gmas.execution import create_openai_caller

settings = FrameworkSettings()

# Access settings
api_key = settings.resolved_api_key
model = settings.model
temperature = settings.temperature

# Create LLM caller from settings
caller = create_openai_caller(
    api_key=settings.resolved_api_key,
    model=settings.model,
    temperature=settings.temperature,
    max_tokens=settings.max_tokens,
    base_url=settings.base_url,
)
```

## File-Based API Keys

For secure environments, load API keys from files:

```bash
# Write key to file
echo "sk-..." > /run/secrets/gmas_api_key
export GMAS_API_KEY_FILE=/run/secrets/gmas_api_key
```

```python
settings = FrameworkSettings()
# resolved_api_key reads the file content
api_key = settings.resolved_api_key
```

## Validation

Settings are validated on instantiation. Invalid values raise errors:

```python
# Missing required key when accessed
settings.resolved_api_key
# RuntimeError: GMAS_API_KEY is required

# Invalid temperature
FrameworkSettings(temperature=5.0)
# ValidationError: temperature must be between 0 and 2
```

Invalid API keys block startup without silent fallback — there is no default key.
