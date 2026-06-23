import tiktoken
import json
import pickle
import os
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import gemini_base as llm

TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")

NO_TRUNCATE_LIMITS = {100_000}

MIN_RATIO = {
    25_000:  0.80,   # easy:   добиваем если < 20K (80% от 25K)
    60_000:  0.83,   # medium: добиваем если < 50K (83% от 60K)
    100_000: 0.60,   # hard:   добиваем если < 60K (60% от 100K)
}

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def count_message_tokens(turn: list) -> int:
    serialized = "".join(f"{m['role']}: {m['content']}\n" for m in turn)
    return count_tokens(serialized)

def count_chat_tokens(data: list) -> int:
    all_text = "".join(
        f"{m['role']}: {m.get('content') or ''}\n"
        for batch in data
        for turn in batch
        for m in turn
    )
    return count_tokens(all_text) if all_text else 0

def get_last_id(data: list) -> int:
    max_id = 0
    for batch in data:
        for turn in batch:
            for msg in turn:
                msg_id = msg.get("id", 0)
                if msg_id > max_id:
                    max_id = msg_id
    return max_id

def _extract_plan_facts(plan_text: str) -> str:
    facts = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        has_data = any([
            re.search(r'\d+[\s.,]*\d*\s*(руб|тыс|млн|%|год|мес|дн)', line, re.I),
            re.search(r'\d{2}\.\d{2}\.\d{4}', line),
            re.search(r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+', line),  # ФИО
        ])
        if has_data:
            facts.append(line)
    return "\n".join(facts[:50])

def _collect_chat_summary(data: list, max_turns: int = 20) -> str:
    all_msgs = []
    for batch in data:
        for turn in batch:
            for msg in turn:
                content = msg.get('content') or ''
                all_msgs.append(f"{msg['role']}: {content[:200]}")

    if len(all_msgs) > max_turns:
        summary = all_msgs[:10] + ["...(пропущено)..."] + all_msgs[-10:]
    else:
        summary = all_msgs
    return "\n".join(summary)

def generate_padding_turns(data: list, plan_text: str, plans_list: list,
                           profile_data: dict, tokens_needed: int, iteration: int) -> list:
    main_spec = profile_data.get("main_spec", {})
    client_name = main_spec.get("name", "Клиент")
    client_job = main_spec.get("job_title", "")
    client_income = main_spec.get("monthly_income", "")

    plan_facts = _extract_plan_facts(plan_text)
    chat_summary = _collect_chat_summary(data)

    batch_idx = iteration % len(plans_list) if plans_list else 0
    focus_plan = plans_list[batch_idx] if plans_list else plan_text[:3000]

    pairs_needed = max(1, min(tokens_needed // 600, 15))

    prompt = f"""Ты — генератор банковских диалогов на РУССКОМ языке.

═══════════════ ПРОФИЛЬ КЛИЕНТА ═══════════════
Имя: {client_name}
Профессия: {client_job}
Доход: {client_income}

═══════════════ ПЛАН ЭТОГО ЭТАПА (детально обсуди) ═══════════════
{focus_plan}

═══════════════ КЛЮЧЕВЫЕ ФАКТЫ ИЗ ПЛАНА ═══════════════
{plan_facts}

═══════════════ КОНТЕКСТ ДИАЛОГА ═══════════════
{chat_summary}

═══════════════ ЗАДАЧА ═══════════════
Сгенерируй {pairs_needed} пар сообщений (клиент + оператор), которые ПРОДОЛЖАЮТ этот диалог.

СТРОГИЕ ПРАВИЛА:
1. Клиент обращается по ИМЕНИ (используй "{client_name.split()[0]}").
2. Каждый вопрос клиента ДОЛЖЕН ссылаться на КОНКРЕТНЫЙ факт из плана:
   - конкретную сумму, дату, ставку, документ или событие
   - "А вы говорили про 12.9% годовых — это фиксированная ставка?"
   - "Когда точно нужно подать документы? Вы сказали до 15 июля?"
3. Оператор отвечает РАЗВЁРНУТО (8-12 предложений):
   - ссылается на конкретные цифры из плана
   - объясняет процедуры, сроки, условия
   - упоминает юридические нюансы (115-ФЗ, НДФЛ, договор и т.д.)
4. НЕ выдумывай новые факты — используй ТОЛЬКО данные из плана.
5. Диалог должен быть естественным: клиент переживает, уточняет, переспрашивает.

ФОРМАТ — JSON список:
[
  {{
    "user": "Вопрос клиента (со ссылкой на факт из плана)",
    "assistant": "Развёрнутый ответ оператора (8-12 предложений с конкретными данными из плана)"
  }}
]

Верни ТОЛЬКО JSON."""

    try:
        raw = llm.invoke(prompt).content
        clean = re.sub(r'```json|```', '', raw).strip()
        from json_repair import repair_json
        pairs = json.loads(repair_json(clean))
        if isinstance(pairs, dict):
            pairs = [pairs]
        return pairs
    except Exception as e:
        print(f"  ⚠️ Ошибка генерации padding: {e}")
        return []

def truncate_chats(chat_directory: str, token_limit: int,
                   no_truncate: Optional[bool] = None, min_ratio: Optional[float] = None):
    chat_path = os.path.join(chat_directory, "chat.json")
    plan_path = os.path.join(chat_directory, "plan.pickle")
    profile_path = os.path.join(chat_directory, "user_profile.json")
    output_chat_path = os.path.join(chat_directory, "chat_truncated.json")

    if os.path.exists(output_chat_path):
        stats_path = os.path.join(chat_directory, "token_stats.json")
        try:
            with open(stats_path, 'r', encoding='utf-8') as _f:
                saved = json.load(_f)
            orig = saved.get("original_tokens", 0)
            final = saved.get("final_tokens", 0)
        except Exception:
            orig, final = 0, 0
        return {"action": "skipped", "reason": "chat_truncated.json already exists",
                "meets_requirement": True, "original_tokens": orig, "final_tokens": final}

    if not os.path.exists(chat_path):
        return {"error": f"Файл не найден: {chat_path}"}

    with open(chat_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plans_list = []
    plan_text = ""
    if os.path.exists(plan_path):
        try:
            with open(plan_path, 'rb') as f:
                plans_list = pickle.load(f)
            plans_list = [p for p in plans_list if p != "[SYNTHETIC PADDING BATCH]"]
            if not plans_list:
                print(f"  ⚠️ plan.pickle пуст или содержит только синтетические батчи — добивка без контекста плана")
            plan_text = "\n\n".join(plans_list)
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            print(f"  ⚠️ Не удалось загрузить plan.pickle: {e}. Продолжаем без плана.")

    profile_data = {}
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
        except Exception as e:
            print(f"  ⚠️ Не удалось загрузить user_profile.json: {e}. Продолжаем без профиля.")

    original_tokens = count_chat_tokens(data)
    _no_truncate = no_truncate if no_truncate is not None else (token_limit in NO_TRUNCATE_LIMITS)
    _min_ratio = min_ratio if min_ratio is not None else MIN_RATIO.get(token_limit, 0.90)
    min_threshold = int(token_limit * _min_ratio)
    action = "none"

    if _no_truncate:
        if original_tokens < token_limit:
            action = "pad"
            data = _pad_chat(data, plan_text, plans_list, profile_data, original_tokens, token_limit)
        else:
            action = "none"
    elif original_tokens > token_limit:
        action = "truncate"
        data = _truncate_chat(data, token_limit)
    elif original_tokens < min_threshold:
        action = "pad"
        data = _pad_chat(data, plan_text, plans_list, profile_data, original_tokens, token_limit)

    final_tokens = count_chat_tokens(data)
    valid_batch_count = len(data)

    if os.path.exists(plan_path):
        try:
            with open(plan_path, 'rb') as f:
                original_plans = pickle.load(f)
            truncated_plans = list(original_plans[:valid_batch_count])
            if valid_batch_count > len(original_plans):
                padded_extra = valid_batch_count - len(original_plans)
                truncated_plans.extend(["[SYNTHETIC PADDING BATCH]"] * padded_extra)
            output_plan_path = os.path.join(chat_directory, "plan_truncated.pickle")
            with open(output_plan_path, 'wb') as f:
                pickle.dump(truncated_plans, f)
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            print(f"  ⚠️ Не удалось синхронизировать план: {e}")
    else:
        print(f"  ⚠️ WARNING: plan.pickle не найден в {chat_directory} — план не синхронизирован с обрезанным чатом!")

    with open(output_chat_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    meets = (final_tokens >= token_limit) if token_limit in NO_TRUNCATE_LIMITS \
        else (min_threshold <= final_tokens <= token_limit)

    if not meets and _no_truncate:
        print(f"  ⚠️ HARD LIMIT NOT MET: итого {final_tokens:,} токенов, требуется ≥ {token_limit:,}. "
              f"Проверьте LLM-соединение и повторите стадию truncate.")

    stats = {
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "token_limit": token_limit,
        "min_threshold": min_threshold,
        "action": action,
        "batches": valid_batch_count,
        "meets_requirement": meets,
    }
    stats_path = os.path.join(chat_directory, "token_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats

def _truncate_chat(data: list, token_limit: int) -> list:
    new_messages = []
    total = 0

    for batch in data:
        valid_turns = []
        for turn in batch:
            turn_tokens = count_message_tokens(turn)
            if total + turn_tokens <= token_limit:
                total += turn_tokens
                valid_turns.append(turn)
            else:
                if valid_turns:
                    new_messages.append(valid_turns)
                return new_messages
        if valid_turns:
            new_messages.append(valid_turns)

    return new_messages

def _pad_chat(data: list, plan_text: str, plans_list: list, profile_data: dict,
              current_tokens: int, target_tokens: int) -> list:
    tokens_needed = target_tokens - current_tokens
    next_id = get_last_id(data) + 1

    print(f"  📝 Добивка: {current_tokens:,} → {target_tokens:,} токенов (нужно +{tokens_needed:,})")

    max_iterations = 20
    iteration = 0
    consecutive_failures = 0

    while tokens_needed > 500 and iteration < max_iterations:
        iteration += 1

        pairs = generate_padding_turns(data, plan_text, plans_list, profile_data, tokens_needed, iteration)
        if not pairs:
            consecutive_failures += 1
            print(f"  ⚠️ Не удалось сгенерировать padding (итерация {iteration}, сбоев подряд: {consecutive_failures})")
            if consecutive_failures >= 3:
                print(f"  ❌ Слишком много сбоев подряд, прекращаем добивку")
                break
            continue

        padding_batch = []
        oversized_skips = 0
        for pair in pairs:
            user_text = pair.get("user", "")
            asst_text = pair.get("assistant", "")
            if not user_text or not asst_text:
                continue

            turn = [
                {"role": "user", "content": user_text, "id": next_id, "padding": True},
                {"role": "assistant", "content": asst_text, "id": next_id + 1, "padding": True}
            ]
            turn_tokens = count_message_tokens(turn)

            if tokens_needed - turn_tokens < -100:
                oversized_skips += 1
                continue

            padding_batch.append(turn)
            tokens_needed -= turn_tokens
            next_id += 2

        if padding_batch:
            consecutive_failures = 0
            data.append(padding_batch)
            print(f"  📝 Итерация {iteration}: +{len(padding_batch)} пар, осталось {tokens_needed:,} токенов")
        else:
            consecutive_failures += 1
            print(f"  ⚠️ Все пары слишком велики ({oversized_skips} шт., итерация {iteration}, сбоев подряд: {consecutive_failures})")
            if consecutive_failures >= 3:
                print(f"  ❌ Слишком много сбоев подряд, прекращаем добивку")
                break

        if tokens_needed <= 500:
            break

    return data

def run_truncation_pipeline(input_directory: str, token_limit: int,
                            no_truncate: Optional[bool] = None, min_ratio: Optional[float] = None):
    dirs = sorted(
        [d for d in os.listdir(input_directory)
         if d.isdigit() and os.path.isdir(os.path.join(input_directory, d))],
        key=lambda x: int(x)
    )

    print(f"--- Обработка чатов: лимит {token_limit:,} токенов ---")

    for d in dirs:
        chat_dir = os.path.join(input_directory, d)
        print(f"\n📁 Папка {d}:")
        try:
            stats = truncate_chats(chat_dir, token_limit, no_truncate=no_truncate, min_ratio=min_ratio)
            if isinstance(stats, dict) and "error" not in stats:
                status = "OK" if stats["meets_requirement"] else "WARN"
                print(f"  [{status}] {stats['original_tokens']:,} → {stats['final_tokens']:,} токенов "
                      f"(действие: {stats['action']})")
            else:
                print(f"  {stats}")
        except Exception as e:
            print(f"  Ошибка: {e}")

if __name__ == "__main__":
    pass
