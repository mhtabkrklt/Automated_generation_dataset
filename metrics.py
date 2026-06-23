import re
import warnings
import threading
from collections import Counter
import json_repair


def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r'["""«»\'`]', '', text)
    text = re.sub(r'(?<!\d)[.,](?!\d)', ' ', text)  # точка/запятая НЕ между цифрами
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def exact_match(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# Загружается один раз; workaround для SIGBUS на Apple Silicon + Python 3.13
# (AutoModel.from_pretrained использует safetensors mmap, который падает в этой конфигурации)
_bert_model = None
_bert_tokenizer = None
_bert_init_lock = threading.Lock()
_bert_call_lock = threading.Lock()

_BERT_MODEL_NAME = "bert-base-multilingual-cased"


def _load_bert_manually():
    import os
    try:
        from safetensors.torch import load_file as st_load_file
        from transformers import BertModel, BertTokenizer, BertConfig
    except ImportError:
        return None, None

    cache_dir = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{_BERT_MODEL_NAME.replace('/', '--')}/snapshots"
    )
    snap = None
    if os.path.exists(cache_dir):
        snaps = sorted(os.listdir(cache_dir))
        if snaps:
            snap = os.path.join(cache_dir, snaps[0])

    try:
        if snap:
            tok = BertTokenizer.from_pretrained(snap)
            config = BertConfig.from_pretrained(snap)
            model = BertModel(config)
            st_path = os.path.join(snap, "model.safetensors")
            if os.path.exists(st_path):
                weights = st_load_file(st_path)
            else:
                import torch
                weights = torch.load(
                    os.path.join(snap, "pytorch_model.bin"), map_location="cpu", weights_only=True
                )
            model.load_state_dict(weights, strict=False)
        else:
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(_BERT_MODEL_NAME)
            model = AutoModel.from_pretrained(_BERT_MODEL_NAME)

        model.eval()
        return model, tok
    except Exception:
        return None, None


def compute_bert_score(pred, gold):
    global _bert_model, _bert_tokenizer
    try:
        import torch
        import torch.nn.functional as F

        if _bert_model is None:
            with _bert_init_lock:
                if _bert_model is None:
                    model, tok = _load_bert_manually()
                    if model is None:
                        return -1.0
                    _bert_model = model
                    _bert_tokenizer = tok

        with _bert_call_lock:
            enc_pred = _bert_tokenizer(
                str(pred), return_tensors="pt", truncation=True, max_length=128
            )
            enc_gold = _bert_tokenizer(
                str(gold), return_tensors="pt", truncation=True, max_length=128
            )
            with torch.no_grad():
                emb_pred = _bert_model(**enc_pred).last_hidden_state[0]
                emb_gold = _bert_model(**enc_gold).last_hidden_state[0]

            # убираем [CLS] и [SEP], нормализуем по L2
            emb_pred = F.normalize(emb_pred[1:-1], dim=-1)
            emb_gold = F.normalize(emb_gold[1:-1], dim=-1)

            if emb_pred.size(0) == 0 or emb_gold.size(0) == 0:
                return 0.0

            sim = torch.mm(emb_pred, emb_gold.T)
            precision = sim.max(dim=1).values.mean().item()
            recall = sim.max(dim=0).values.mean().item()

            if precision + recall == 0.0:
                return 0.0
            f1 = 2 * precision * recall / (precision + recall)

        return max(0.0, min(1.0, round(f1, 4)))
    except Exception:
        return -1.0


def compute_all_metrics(pred, gold):
    return {
        "exact_match": exact_match(pred, gold),
        "token_f1": token_f1(pred, gold),
        "bert_score": compute_bert_score(pred, gold),
    }


def final_accuracy(metrics: dict) -> float:
    em = metrics.get("exact_match", 0.0)
    if em == 1.0:
        return 1.0
    judge = metrics.get("llm_judge", -1.0)
    if judge >= 0.0:
        return judge
    return em


def should_call_judge(metrics: dict) -> bool:
    if metrics.get("exact_match", -1.0) == 1.0:
        return False

    bert = metrics.get("bert_score", -1.0)
    f1 = metrics.get("token_f1", -1.0)

    bert_available = bert >= 0.0
    f1_available = f1 >= 0.0

    if not bert_available and not f1_available:
        return False

    if not bert_available or not f1_available:
        return True  # неполная информация — лучше вызвать судью

    # не вызываем только если обе метрики уверенно говорят «неверно»
    return not (bert < 0.5 and f1 < 0.3)


def extract_answer_from_json(response: str) -> str:
    if not response or not response.strip():
        return ""

    try:
        parsed = json_repair.loads(response)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "answer" in parsed[0]:
            return str(parsed[0]["answer"]).strip()
    except Exception:
        pass

    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1 and start < end:
        try:
            parsed = json_repair.loads(response[start:end + 1])
            if isinstance(parsed, dict) and "answer" in parsed:
                return str(parsed["answer"]).strip()
        except Exception:
            pass

    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
    if code_match:
        try:
            parsed = json_repair.loads(code_match.group(1).strip())
            if isinstance(parsed, dict) and "answer" in parsed:
                return str(parsed["answer"]).strip()
        except Exception:
            pass

    match = re.search(r'"answer"\s*:\s*"([^"]+)"', response)
    if not match:
        match = re.search(r'"answer"\s*:\s*\'([^\']+)\'', response)
    if not match:
        match = re.search(r'"answer"\s*:\s*([^"\'}\n,]+)', response)
    if match:
        return match.group(1).strip()

    # короткий ответ — модель ответила plain text; длинный — reasoning или мусор
    result = response.strip()
    return result if len(result) <= 100 else ""


def path_accuracy(predicted: str, gold: str) -> float:
    def normalize_step(s: str) -> set:
        return set(re.sub(r'[^\w\s]', '', s.lower()).split())

    pred_steps = [s.strip() for s in predicted.split("→") if s.strip()]
    gold_steps = [s.strip() for s in gold.split("→") if s.strip()]

    if not gold_steps:
        return 1.0 if not pred_steps else 0.0
    if not pred_steps:
        return 0.0

    n_gold = len(gold_steps)
    n_pred = len(pred_steps)

    recall_total = 0.0
    for i, gold_step in enumerate(gold_steps):
        if i < n_pred:
            gold_tokens = normalize_step(gold_step)
            pred_tokens = normalize_step(pred_steps[i])
            if gold_tokens:
                recall_total += len(gold_tokens & pred_tokens) / len(gold_tokens)
    recall = recall_total / n_gold

    precision_total = 0.0
    for i, pred_step in enumerate(pred_steps):
        if i < n_gold:
            gold_tokens = normalize_step(gold_steps[i])
            pred_tokens = normalize_step(pred_step)
            if pred_tokens:
                precision_total += len(gold_tokens & pred_tokens) / len(pred_tokens)
    precision = precision_total / n_pred

    if precision + recall == 0.0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def avg_metrics(results: list, include_path: bool = False) -> dict:
    def _safe(lst):
        return round(sum(lst) / len(lst), 4) if lst else -1.0

    if not results:
        base = {"count": 0, "exact_match": -1.0, "token_f1": -1.0,
                "bert_score": -1.0, "llm_judge": -1.0, "final_accuracy": -1.0}
        if include_path:
            base["path_accuracy"] = -1.0
        return base

    em_values    = [r["metrics"]["exact_match"]  for r in results if r["metrics"].get("exact_match",  -1.0) >= 0]
    f1_values    = [r["metrics"]["token_f1"]      for r in results if r["metrics"].get("token_f1",     -1.0) >= 0]
    bert_values  = [r["metrics"]["bert_score"]    for r in results if r["metrics"].get("bert_score",   -1.0) >= 0]
    judge_values = [r["metrics"].get("llm_judge", -1.0) for r in results if r["metrics"].get("llm_judge", -1.0) >= 0]
    fa_values    = [r["metrics"]["final_accuracy"] for r in results if r["metrics"].get("final_accuracy", -1.0) >= 0]

    result = {
        "count":          len(results),
        "exact_match":    _safe(em_values),
        "token_f1":       _safe(f1_values),
        "bert_score":     _safe(bert_values),
        "llm_judge":      _safe(judge_values),
        "final_accuracy": _safe(fa_values),
    }
    if include_path:
        path_values = [r["metrics"]["path_accuracy"] for r in results if "path_accuracy" in r["metrics"]]
        result["path_accuracy"] = _safe(path_values)
    return result


def _call_judge(judge_adapter, prompt, expected_key: str) -> float:
    messages = [
        {"role": "system", "content": "Ты — строгий судья качества ответов. Отвечай только в JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = judge_adapter.invoke(messages).content
        parsed = json_repair.loads(response)
        if isinstance(parsed, dict):
            if expected_key in parsed:
                return max(0.0, min(1.0, float(parsed[expected_key])))
            for key in ("score", "blind_score", "reference_score"):
                if key in parsed:
                    return max(0.0, min(1.0, float(parsed[key])))
        return -1.0
    except Exception:
        return -1.0


def run_llm_judge(judge_adapter, question, gold_answer, predicted_answer, task_type):
    reference_prompt = f"""Ты — строгий судья, сравнивающий ответ модели с эталонным ответом.

Тип задачи: {task_type}
Вопрос: {question}
Эталонный ответ: {gold_answer}
Ответ модели: {predicted_answer}

Оцени фактическую точность:
- 1.0 — совпадает по смыслу и значению с эталоном (допустима вариация форматирования: "14.5%" = "14,5% годовых")
- 0.5 — допустимо ТОЛЬКО для нефактических ответов (статус, категория); для числовых/дат — только 1.0 или 0.0
- 0.0 — не совпадает с эталоном, приблизительный ответ на точный вопрос, отсутствует

Ответь СТРОГО в JSON: {{"reference_score": <число 0.0/0.5/1.0>, "reasoning": "<1 предложение>"}}"""

    return _call_judge(judge_adapter, reference_prompt, "reference_score")
