import os
import time
import random
from typing import Union
from dotenv import load_dotenv

load_dotenv()

class FakeLangChainResponse:
    def __init__(self, text):
        self.content = text if text else ""

class VertexAIAdapter:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, input_data, **kwargs):
        system_instruction = None
        messages = []

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})

        elif isinstance(input_data, list):
            for msg in input_data:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                elif hasattr(msg, 'content'):
                    role_attr = getattr(msg, 'type', 'user')
                    if role_attr == 'human':
                        role = 'user'
                    elif role_attr == 'ai':
                        role = 'assistant'
                    elif role_attr == 'system':
                        role = 'system'
                    else:
                        role = 'user'
                    content = msg.content
                else:
                    role = 'user'
                    content = str(msg)
                messages.append({"role": role, "content": content})

        system_parts = [m for m in messages if m["role"] == "system"]
        chat_messages = [m for m in messages if m["role"] != "system"]

        if system_parts:
            system_instruction = "\n".join(m["content"] for m in system_parts)

        max_tokens = kwargs.get('max_tokens', 32000)

        generation_config = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=max_tokens,
        )

        history = []
        for msg in chat_messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append(Content(role=role, parts=[Part.from_text(msg["content"])]))

        last_message = chat_messages[-1]["content"] if chat_messages else ""

        max_retries = 5
        base_delay = 5
        max_backoff = 60
        all_errors = []

        for attempt in range(max_retries):
            try:
                model = GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
                chat = model.start_chat(history=history)
                response = chat.send_message(
                    last_message,
                    generation_config=generation_config
                )
                return FakeLangChainResponse(response.text)

            except Exception as e:
                error_str = str(e)
                all_errors.append(f"[попытка {attempt + 1}] {error_str}")
                print(f"⚠️ API Error (Попытка {attempt + 1}/{max_retries}): {error_str}")

                if "429" in error_str or "rate limit" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = min(base_delay * (2 ** attempt) + random.uniform(1, 5), max_backoff)
                    print(f"   Rate limit — ожидание {wait_time:.1f} сек...")
                    time.sleep(wait_time)
                else:
                    wait_time = min(2 ** attempt + random.uniform(0, 1), max_backoff)
                    time.sleep(wait_time)

        errors_summary = " | ".join(all_errors[-2:])
        raise RuntimeError(
            f"API max retries exceeded for model {self.model_name}. "
            f"История ошибок: [{errors_summary}]"
        )

TOKEN_USAGE = {
    "input": 0,
    "output": 0,
    "calls": 0,
    "retry_calls": 0,
    "retry_input": 0,
    "rate_limit_hits": 0,
}

def reset_token_usage():
    for k in TOKEN_USAGE:
        TOKEN_USAGE[k] = 0

def get_token_usage():
    return dict(TOKEN_USAGE)

def _estimate_tokens(messages: list) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(m.get("content", "") if isinstance(m, dict) else str(m))) for m in messages)
    except Exception:
        return sum(len((m.get("content", "") if isinstance(m, dict) else str(m))) // 4 for m in messages)

class OpenAICompatibleAdapter:
    def __init__(self, api_key: str, base_url: str, model_name: str, temperature: float):
        from openai import OpenAI
        import httpx
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self.model_name = model_name
        self.temperature = temperature

    def _call_api(self, messages, max_tokens):
        import threading
        result = [None]
        error = [None]

        def _do_call():
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                result[0] = resp
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_do_call, daemon=True)
        t.start()
        t.join(timeout=180)

        if t.is_alive():
            raise TimeoutError("LLM не ответил за 180 секунд")
        if error[0] is not None:
            raise error[0]
        return result[0]

    def invoke(self, input_data, **kwargs):
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            messages = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                if isinstance(m, dict) else {"role": "user", "content": str(m)}
                for m in input_data
            ]
        else:
            messages = [{"role": "user", "content": str(input_data)}]

        max_retries = 3
        base_delay = 3
        max_backoff = 30
        all_errors = []

        for attempt in range(max_retries):
            try:
                response = self._call_api(messages, kwargs.get("max_tokens", 16384))
                if hasattr(response, "usage") and response.usage:
                    TOKEN_USAGE["input"] += response.usage.prompt_tokens or 0
                    TOKEN_USAGE["output"] += response.usage.completion_tokens or 0
                    TOKEN_USAGE["calls"] += 1
                return FakeLangChainResponse(response.choices[0].message.content or "")
            except Exception as e:
                error_str = str(e)
                all_errors.append(f"[попытка {attempt + 1}] {error_str}")
                print(f"⚠️ API Error (Попытка {attempt + 1}/{max_retries}): {error_str}")
                if any(code in error_str for code in ("401", "403", "invalid_api_key", "Unauthorized")):
                    raise RuntimeError(f"Auth error — ретрай бесполезен: {error_str}")
                if "429" in error_str or "rate limit" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                    TOKEN_USAGE["retry_calls"] += 1
                    TOKEN_USAGE["rate_limit_hits"] += 1
                    wait_time = min(base_delay * (2 ** attempt) + random.uniform(1, 5), max_backoff)
                    print(f"   Rate limit — ожидание {wait_time:.1f} сек...")
                    time.sleep(wait_time)
                else:
                    TOKEN_USAGE["retry_calls"] += 1
                    TOKEN_USAGE["retry_input"] += _estimate_tokens(messages)
                    wait_time = min(2 ** attempt + random.uniform(0, 1), max_backoff)
                    time.sleep(wait_time)

        errors_summary = " | ".join(all_errors[-2:])
        raise RuntimeError(
            f"API max retries exceeded for model {self.model_name}. "
            f"История ошибок: [{errors_summary}]"
        )

class BuildLLM:
    def __init__(self, api_key: str, base_url: str, model_name: str, temperature: float):
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self._temperature = temperature

    def build_llm(self) -> OpenAICompatibleAdapter:
        return OpenAICompatibleAdapter(
            api_key=self._api_key,
            base_url=self._base_url,
            model_name=self._model_name,
            temperature=self._temperature,
        )

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemini-2.5-flash")
VALIDATOR_MODEL = os.environ.get("VALIDATOR_MODEL", "google/gemini-2.5-pro")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

_AdapterType = Union[VertexAIAdapter, OpenAICompatibleAdapter]

class _UninitLLM:
    def invoke(self, *args, **kwargs):
        raise RuntimeError(
            "LLM не инициализирован. Заполните OPENAI_API_KEY / OPENAI_BASE_URL в .env"
        )

gemini_base: _AdapterType = _UninitLLM()  # type: ignore[assignment]
validator_llm: _AdapterType = _UninitLLM()  # type: ignore[assignment]

if PROJECT_ID:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
    vertexai.init(project=PROJECT_ID, location=LOCATION, api_transport="rest")
    gemini_base = VertexAIAdapter(model_name=MODEL_NAME, temperature=0.3)
    validator_llm = VertexAIAdapter(model_name=VALIDATOR_MODEL, temperature=0.0)
elif OPENAI_API_KEY and OPENAI_BASE_URL:
    gemini_base = OpenAICompatibleAdapter(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model_name=MODEL_NAME,
        temperature=0.3,
    )
    validator_llm = OpenAICompatibleAdapter(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model_name=VALIDATOR_MODEL,
        temperature=0.0,
    )
    print(f"✅ LLM инициализирован: {MODEL_NAME} / validator: {VALIDATOR_MODEL}")
else:
    import warnings
    warnings.warn(
        "Ни GOOGLE_CLOUD_PROJECT, ни OPENAI_API_KEY не заданы — LLM недоступен. "
        "Заполните .env файл.",
        UserWarning,
        stacklevel=1,
    )
