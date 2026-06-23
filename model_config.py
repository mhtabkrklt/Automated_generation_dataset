import os
import warnings
from dotenv import load_dotenv
from llm import BuildLLM

load_dotenv()


MODEL_REGISTRY = {
    "google/gemini-2.5-pro": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "google/gemini-2.5-pro",
        "temperature": 0.1,
    },
    "google/gemini-2.5-flash": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "google/gemini-2.5-flash",
        "temperature": 0.1,
    },
    "openai/gpt-4o-mini": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "openai/gpt-4o-mini",
        "temperature": 0.1,
    },
    "openai/gpt-4o": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "openai/gpt-4o",
        "temperature": 0.1,
    },
    "openai/gpt-5-mini": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "openai/gpt-5-mini",
        "temperature": 0.1,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "meta-llama/llama-3.3-70b-instruct",
        "temperature": 0.1,
    },
    "qwen/qwen3-32b": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "qwen/qwen3-32b",
        "temperature": 0.1,
    },
    "kimi-k2p6": {
        "api_key_env": "HYDRA_API_KEY",
        "base_url_env": "HYDRA_BASE_URL",
        "model_name": "kimi-k2p6",
        "temperature": 0.1,
    },
    "google/gemini-3.1-flash-lite-preview": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_name": "google/gemini-3.1-flash-lite-preview",
        "temperature": 0.0,
    },
}

_DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
_DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def build_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Модель '{name}' не найдена в MODEL_REGISTRY. Доступные: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[name]

    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        if _DEFAULT_API_KEY:
            msg = (
                f"[model_config] {cfg['api_key_env']} не задан — используется OPENAI_API_KEY "
                f"для модели '{name}'. Убедитесь, что ключ подходит для этого провайдера."
            )
            print(f"⚠️  WARNING: {msg}")
            warnings.warn(msg, UserWarning, stacklevel=2)
            api_key = _DEFAULT_API_KEY
        else:
            raise ValueError(f"API-ключ не найден для модели '{name}' (env: {cfg['api_key_env']}). Проверьте .env файл.")

    base_url = os.environ.get(cfg["base_url_env"], "")
    if not base_url:
        if _DEFAULT_BASE_URL:
            msg = (
                f"[model_config] {cfg['base_url_env']} не задан — используется OPENAI_BASE_URL "
                f"для модели '{name}'. Убедитесь, что endpoint подходит для этого провайдера."
            )
            print(f"⚠️  WARNING: {msg}")
            warnings.warn(msg, UserWarning, stacklevel=2)
            base_url = _DEFAULT_BASE_URL
        else:
            raise ValueError(f"BASE_URL не задан для модели '{name}' (env: {cfg['base_url_env']}). Проверьте .env файл.")

    builder = BuildLLM(
        api_key=api_key,
        base_url=base_url,
        model_name=cfg["model_name"],
        temperature=cfg["temperature"],
    )
    return builder.build_llm()


def get_available_models():
    available = []
    for name, cfg in MODEL_REGISTRY.items():
        api_key = os.environ.get(cfg["api_key_env"], _DEFAULT_API_KEY)
        base_url = os.environ.get(cfg["base_url_env"], _DEFAULT_BASE_URL)
        if api_key and base_url:
            available.append(name)
    return available
