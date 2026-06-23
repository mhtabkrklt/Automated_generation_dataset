import pickle
import json
import os
import re
import time
from datetime import datetime
from collections import OrderedDict
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore[assignment]

RUSSIAN_MONTHS = {
    'января': '01', 'январь': '01', 'january': '01', 'jan': '01',
    'февраля': '02', 'февраль': '02', 'february': '02', 'feb': '02',
    'марта': '03', 'март': '03', 'march': '03', 'mar': '03',
    'апреля': '04', 'апрель': '04', 'april': '04', 'apr': '04',
    'мая': '05', 'май': '05', 'may': '05',
    'июня': '06', 'июнь': '06', 'june': '06', 'jun': '06',
    'июля': '07', 'июль': '07', 'july': '07', 'jul': '07',
    'августа': '08', 'август': '08', 'august': '08', 'aug': '08',
    'сентября': '09', 'сентябрь': '09', 'september': '09', 'sep': '09',
    'октября': '10', 'октябрь': '10', 'october': '10', 'oct': '10',
    'ноября': '11', 'ноябрь': '11', 'november': '11', 'nov': '11',
    'декабря': '12', 'декабрь': '12', 'december': '12', 'dec': '12'
}


def get_token_number(text):
    if not text:
        return 0

    if tiktoken:
        for encoding_name in ["cl100k_base", "p50k_base", "r50k_base"]:
            try:
                encoder = tiktoken.get_encoding(encoding_name)
                return len(encoder.encode(str(text)))
            except Exception:
                continue

    # кириллица: ~3 символа на токен
    return len(str(text)) // 3


def parse_russian_date(date_str: str):
    date_str = date_str.strip()

    m = re.match(r'^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$', date_str)
    if m:
        day, month, year = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        return f"{year}-{month}-{day}"

    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_str)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    date_lower = date_str.lower().replace(",", "")
    parts = date_lower.split()

    day, month, year = None, None, None
    for part in parts:
        if part.isdigit():
            if len(part) == 4:
                year = part
            elif len(part) <= 2:
                if not day:
                    day = part.zfill(2)
                elif int(part) <= 12 and not month:
                    month = part.zfill(2)
        elif part in RUSSIAN_MONTHS:
            month = RUSSIAN_MONTHS[part]

    if day and month and year:
        return f"{year}-{month}-{day}"

    return None


def extract_time_anchor(plan_text: str):
    lines = plan_text.strip().split('\n')
    first_bullet = ""
    for line in lines:
        if "Time Anchor" in line or "Временная метка" in line:
            first_bullet = line
            break

    if not first_bullet:
        return "Unknown Date"

    patterns = [
        r'(\d{1,2}\s+[а-яА-Яa-zA-Z]+\s+\d{4})',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'([а-яА-Яa-zA-Z]+\s+\d{1,2},?\s*\d{4})',
        r'(\d{1,2}[\./-]\d{1,2}[\./-]\d{4})',
    ]

    for pattern in patterns:
        match = re.search(pattern, first_bullet, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            parsed = parse_russian_date(date_str)
            return parsed if parsed else date_str

    return "Unknown Date"


def convert_user_messages_pickle_to_json(input_address: str, output_address: str) -> None:
    with open(input_address, 'rb') as f:
        batches = pickle.load(f)

    all_batches = []
    for batch_index, batch in enumerate(batches, start=1):
        time_anchor = batch.get('time_anchor', 'Unknown')

        raw_msgs = batch.get('messages', [])
        if raw_msgs and isinstance(raw_msgs[0], list):
            raw_msgs = raw_msgs[0]

        batch_messages = []
        for message in raw_msgs:
            content = message
            if isinstance(message, dict):
                content = message.get('content', '')
            if not content.strip():
                continue
            batch_messages.append({"role": "user", "content": content.strip()})

        all_batches.append({
            "batch": batch_index,
            "time_anchor": time_anchor,
            "messages": batch_messages
        })

    with open(output_address, "w", encoding="utf-8") as f:
        json.dump(all_batches, f, indent=4, ensure_ascii=False)


def convert_chats_pickle_to_json(input_address: str, output_address: str) -> None:
    if not os.path.exists(input_address):
        print(f"File not found: {input_address}")
        return

    with open(input_address, 'rb') as f:
        batches = pickle.load(f)

    all_dialogues = []
    # плоский список сообщений — оборачиваем в один батч
    if batches and isinstance(batches[0], dict) and 'role' in batches[0]:
        batches = [batches]

    for index, batch in enumerate(batches):
        batch_number = index + 1
        turns = []
        per_batch_turns: dict[str, Any] = {
            'batch_number': batch_number,
            'time_anchor': None,
            'turns': []
        }

        current_turn = []
        for msg in batch:
            if msg.get('role') == 'user' and 'time_anchor' in msg:
                per_batch_turns['time_anchor'] = msg['time_anchor']
            current_turn.append(msg)
            if msg.get('role') == 'assistant':
                turns.append(current_turn)
                current_turn = []

        if current_turn:
            turns.append(current_turn)

        per_batch_turns['turns'] = turns
        all_dialogues.append(per_batch_turns)

    with open(output_address, 'w', encoding="utf-8") as f:
        json.dump(all_dialogues, f, indent=2, ensure_ascii=False)


def add_ids_to_chats(input_file: str, output_address: str):
    if not os.path.exists(input_file):
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    msg_id = 0
    iterable = data if isinstance(data, list) else []

    for batch in iterable:
        for turn in batch.get('turns', []):
            for msg in turn:
                new_msg = OrderedDict()
                new_msg['id'] = msg_id
                for key, val in msg.items():
                    if key != 'id':
                        new_msg[key] = val
                msg.clear()
                msg.update(new_msg)
                msg_id += 1

    with open(output_address, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_all_chat_dirs(base_dir: str) -> list:
    chat_dirs: list[str] = []
    if not os.path.exists(base_dir):
        return chat_dirs
    for task_type in sorted(os.listdir(base_dir)):
        type_dir = os.path.join(base_dir, task_type)
        if not os.path.isdir(type_dir):
            continue
        for chat_id in sorted(
            [x for x in os.listdir(type_dir) if x.isdigit()],
            key=lambda x: int(x)
        ):
            chat_dir = os.path.join(type_dir, chat_id)
            if os.path.isdir(chat_dir):
                chat_dirs.append(chat_dir)
    return chat_dirs


_EVAL_RESULT_DIRS = {"eval_results", "eval_results_graph", "eval_results_memory", "eval_results_memory_graph"}


def get_eval_chat_dirs(base_dir: str) -> list:
    chat_dirs: list[tuple[str, str]] = []
    if not os.path.isdir(base_dir):
        return chat_dirs
    for task_type in sorted(os.listdir(base_dir)):
        if task_type in _EVAL_RESULT_DIRS:
            continue
        type_dir = os.path.join(base_dir, task_type)
        if not os.path.isdir(type_dir):
            continue
        for seq_id in sorted(
            [d for d in os.listdir(type_dir)
             if d.isdigit() and os.path.isdir(os.path.join(type_dir, d))],
            key=lambda x: int(x)
        ):
            chat_dirs.append((f"{task_type}/{seq_id}", os.path.join(type_dir, seq_id)))
    return chat_dirs


def invoke_with_retries(model_adapter, messages: list, max_retries: int = 3) -> str:
    response = ""
    for attempt in range(max_retries):
        try:
            response = model_adapter.invoke(messages).content
            break
        except Exception as e:
            err_str = str(e).lower()
            if any(marker in err_str for marker in ("401", "403", "invalid api key", "authentication")):
                print(f"  ❌ Ошибка аутентификации: {type(e).__name__}: {e}")
                break
            if "contextwindowexceeded" in err_str or "context_length_exceeded" in err_str or "input tokens exceed" in err_str:
                print(f"  ❌ Превышен контекст модели: {type(e).__name__}")
                break
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  ⚠️ Ошибка (попытка {attempt+1}/{max_retries}): {type(e).__name__}: {e}. Ожидание {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Ошибка вызова модели после {max_retries} попыток: {type(e).__name__}: {e}")
    return response


def convert_txt_plan_to_pickle(chat_directory: str):
    input_plan_address = os.path.join(chat_directory, "plan_new.txt")
    if not os.path.exists(input_plan_address):
        return

    with open(input_plan_address, 'r', encoding="utf-8") as f:
        plans_text = f.read()

    raw_batches = re.split(r'(BATCH \d+ PLAN)', plans_text)

    plans = []
    for i in range(1, len(raw_batches), 2):
        header = raw_batches[i].strip()
        content = raw_batches[i + 1].strip()
        plans.append(f"{header}\n{content}")

    output_plan_address = os.path.join(chat_directory, "plan_new.pickle")
    with open(output_plan_address, 'wb') as f:
        pickle.dump(plans, f)
