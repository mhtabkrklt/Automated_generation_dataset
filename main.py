import random
import os
import re
import time
import pickle
import json
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from json_repair import repair_json

from llm import gemini_base as llm, validator_llm
from prompts import (
    label_generation_prompt_template,
    plan_generation_prompt_profile_and_topic_given_detailed_template,
    message_generation_prompt_focused_template,
    ai_assistant_llm_template,
    FAMILY_INTERFERENCE_EASY_INSTRUCTION,
    FAMILY_INTERFERENCE_MEDIUM_INSTRUCTION,
    FAMILY_INTERFERENCE_HARD_INSTRUCTION,
    information_extraction_selection_easy,
    information_extraction_selection_medium,
    information_extraction_selection_hard,
    knowledge_update_selection_easy,
    knowledge_update_selection_medium,
    knowledge_update_selection_hard,
    temporal_reasoning_selection_easy,
    temporal_reasoning_selection_medium,
    temporal_reasoning_selection_hard,
    interference_selection_easy,
    interference_selection_medium,
    interference_selection_hard,
    composite_selection_easy,
    composite_selection_medium,
    composite_selection_hard,
    information_extraction,
    information_extraction_gen_easy,
    information_extraction_gen_medium,
    information_extraction_gen_hard,
    knowledge_update_gen_easy,
    knowledge_update_gen_medium,
    knowledge_update_gen_hard,
    temporal_reasoning_gen_easy,
    temporal_reasoning_gen_medium,
    temporal_reasoning_gen_hard,
    interference_gen_easy,
    interference_gen_medium,
    interference_gen_hard,
    composite_gen_easy,
    composite_gen_medium,
    composite_gen_hard,
    task_validation_prompt,
)
from profile_creation import create_profile, extract_profile_from_events
from utils import extract_time_anchor, get_token_number

class ConversationSummaryBuffer:
    def __init__(self, llm_model, max_tokens=20000, recent_messages_count=10):
        self.llm = llm_model
        self.max_tokens = max_tokens
        self.recent_messages_count = recent_messages_count
        self.messages = []
        self.summary = ""
        self.key_facts_log: list[str] = []

    @staticmethod
    def _extract_key_facts(content: str) -> list[str]:
        """Извлекает точные числа, суммы, даты и имена из текста сообщения."""
        facts = []
        for m in re.finditer(
            r'\b\d[\d\s]*(?:[.,]\d+)?\s*(?:руб(?:лей|ля|\.)?|тыс(?:\.|\b)|млн(?:\.|\b)|%|₽)',
            content, re.IGNORECASE
        ):
            facts.append(m.group(0).strip())
        for m in re.finditer(r'\b\d{2}\.\d{2}\.(?:\d{4}|\d{2})\b', content):
            facts.append(m.group(0))
        for m in re.finditer(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b', content):
            candidate = m.group(0)
            if len(candidate.split()) >= 2:
                facts.append(candidate)
        return list(dict.fromkeys(facts))  # дедупликация с сохранением порядка

    def add_message(self, role, content):
        content_str = str(content) if content else ""
        self.messages.append({"role": role, "content": content_str})

        new_facts = self._extract_key_facts(content_str)
        for f in new_facts:
            if f not in self.key_facts_log:
                self.key_facts_log.append(f)

        if len(self.messages) > self.recent_messages_count + 5:
            self._compress_memory()

    def _compress_memory(self):
        to_summarize = self.messages[:-self.recent_messages_count]
        self.messages = self.messages[-self.recent_messages_count:]

        history_text = ""
        for msg in to_summarize:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = str(msg.get("content", ""))
            history_text += f"{role}: {content}\n"

        prompt = f"""Суммируй следующие сообщения в краткий, но информативный текст.
        Сохрани ключевые факты, цифры и договоренности.
        Текущее резюме (если есть): {self.summary}
        Новые сообщения:
        {history_text}
        Новое итоговое резюме:"""

        try:
            summary_response = self.llm.invoke(prompt)
            if summary_response and summary_response.content and summary_response.content.strip():
                self.summary = summary_response.content.strip()
                print(f"✨ Память сжата. Новое саммари: {self.summary[:50]}...")
            else:
                raise ValueError("Пустой ответ от LLM при сжатии памяти")
        except Exception as e:
            print(f"⚠️ Ошибка сжатия памяти, применяем обрезку: {e}")
            digest_parts = []
            for msg in to_summarize[-5:]:
                role = "User" if msg["role"] == "user" else "Asst"
                text = str(msg.get("content", ""))[:120]
                digest_parts.append(f"{role}: {text}")
            if digest_parts:
                self.summary = (self.summary + " | " if self.summary else "") + " | ".join(digest_parts)
                self.summary = self.summary[-2000:]  # не раздуваем резюме

    def _key_facts_context(self) -> str:
        """Формирует строку со всеми зафиксированными фактами для включения в промпт."""
        if not self.key_facts_log:
            return ""
        facts_sample = self.key_facts_log[-100:]
        return ("[ВНУТРЕННИЙ КОНТЕКСТ — НЕ ВОСПРОИЗВОДИТЬ В ОТВЕТЕ] "
                "Числа/даты/имена из истории диалога: "
                + " | ".join(facts_sample))

    def get_messages_for_llm(self, system_prompt=None):
        formatted_messages = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        if self.summary:
            context_msg = f"Краткий контекст прошлых событий: {self.summary}"
            formatted_messages.append({"role": "system", "content": context_msg})

        facts_ctx = self._key_facts_context()
        if facts_ctx:
            formatted_messages.append({"role": "system", "content": facts_ctx})

        formatted_messages.extend(self.messages)
        return formatted_messages

    def get_conversation_history_text(self):
        history_text = f"КОНТЕКСТ ПРОШЛОГО: {self.summary}\n\n" if self.summary else ""
        facts_ctx = self._key_facts_context()
        if facts_ctx:
            history_text += f"{facts_ctx}\n\n"
        for msg in self.messages:
            role = "Клиент" if msg["role"] == "user" else "Оператор"
            history_text += f"{role}: {msg['content']}\n"
        return history_text.strip()

    def estimate_tokens(self, text):
        return len(str(text)) // 3

def generate_labels(topic: str, theme: str, domain: str) -> str:
    prompt = label_generation_prompt_template.replace("<topic>", topic).replace("<theme>", theme)
    response = llm.invoke(prompt).content
    return response

def extract_labels(labels_text: str) -> List[Dict]:
    HEADER_RE = re.compile(r"^\*{0,2}(?P<title>.+? Labels):\*{0,2}$", re.MULTILINE)

    lines = [ln.strip() for ln in labels_text.splitlines() if ln.strip()]
    records: list[dict] = []
    current: Optional[dict] = None

    for ln in lines:
        header = HEADER_RE.match(ln)
        if header:
            if current: records.append(current)
            current = {"category": header.group("title"), "description": "", "sublabels": []}
            continue

        if ln.startswith("-") and current:
            bullet = ln.lstrip("- ").strip()
            if not current["description"]:
                current["description"] = bullet
            else:
                current["sublabels"] = [s.strip() for s in bullet.split(",")]

    if current: records.append(current)
    return records

def format_labels_for_llm(records):
    formatted = []
    for i, rec in enumerate(records, 1):
        sub = ', '.join(rec['sublabels']) if rec['sublabels'] else "Общие вопросы"
        formatted.append(f"{i}. CATEGORY: {rec['category']}\n   DESC: {rec['description']}\n   FOCUS: {sub}")
    return "\n\n".join(formatted)

def get_labels_pipeline(topic: str, theme: str) -> str:
    raw_text = generate_labels(topic, theme, "finance")
    parsed = extract_labels(raw_text)
    return format_labels_for_llm(parsed)

def get_profile_pipeline(events=None, name_collision_prob: float = 0.0, client_name: Optional[str] = None):
    events_overrides: Optional[dict] = None
    if client_name:
        events_overrides = {"name": client_name}
    elif events:
        events_overrides = extract_profile_from_events(events) or None

    main_spec, relationships = create_profile(
        events_overrides=events_overrides,
        name_collision_prob=name_collision_prob,
    )

    profile_text = f"""
    ИМЯ: {main_spec['name']}
    ВОЗРАСТ: {main_spec['age']}
    ПОЛ: {main_spec['gender']}
    РАБОТА: {main_spec['job_title']}
    ЛОКАЦИЯ: {main_spec.get('location', 'Россия')}
    ДОХОД: {main_spec.get('monthly_income', 'Не указан')}

    ПСИХОЛОГИЧЕСКИЙ ПОРТРЕТ:
    {main_spec['personality_traits']}
    """

    rel_text = ""
    for rel_type, people in relationships.items():
        rel_text += f"\n{rel_type.upper()}:\n"
        for p in people:
            rel_text += f"- {p['name']} ({p['age']} лет)\n"

    return profile_text, rel_text, main_spec, relationships

def generate_plans(topic: str, theme: str,
                   num_batches: int, num_bullets: int,
                   save_address: str, noise_settings: Optional[dict] = None,
                   timeline: str = "3 месяца",
                   events: Optional[list] = None, task_type: Optional[str] = None,
                   client_name: Optional[str] = None):
    chat_dir = os.path.dirname(save_address)
    os.makedirs(chat_dir, exist_ok=True)
    labels = get_labels_pipeline(topic, theme)
    profile_save_path = os.path.join(chat_dir, "user_profile.json")

    if os.path.exists(profile_save_path):
        with open(profile_save_path, 'r', encoding='utf-8') as f:
            full_profile_data = json.load(f)
        raw_profile_data = full_profile_data['main_spec']
        raw_rels_data = full_profile_data['relationships']
        main_spec = raw_profile_data
        user_profile_text = f"""
    ИМЯ: {main_spec['name']}
    ВОЗРАСТ: {main_spec['age']}
    ПОЛ: {main_spec['gender']}
    РАБОТА: {main_spec['job_title']}
    ЛОКАЦИЯ: {main_spec.get('location', 'Россия')}
    ДОХОД: {main_spec.get('monthly_income', 'Не указан')}

    ПСИХОЛОГИЧЕСКИЙ ПОРТРЕТ:
    {main_spec['personality_traits']}
    """
        user_rels_text = ""
        for rel_type, people in raw_rels_data.items():
            user_rels_text += f"\n{rel_type.upper()}:\n"
            for p in people:
                user_rels_text += f"- {p['name']} ({p['age']} лет)\n"
        print(f"👤 Профиль загружен из кэша: {profile_save_path}")
    else:
        name_collision_prob = noise_settings.get("name_collision_prob", 0.0) if noise_settings else 0.0

        user_profile_text, user_rels_text, raw_profile_data, raw_rels_data = get_profile_pipeline(
            events=events,
            name_collision_prob=name_collision_prob,
            client_name=client_name,
        )
        full_profile_data = {
            "main_spec": raw_profile_data,
            "relationships": raw_rels_data
        }
        with open(profile_save_path, "w", encoding="utf-8") as f:
            json.dump(full_profile_data, f, indent=4, ensure_ascii=False)
        print(f"👤 Профиль пользователя сохранен в: {profile_save_path}")

    _narrative_events_block = ""
    if events:
        _narrative_events_block = "\n    ОБЯЗАТЕЛЬНЫЕ СОБЫТИЯ (должны войти в сюжет в указанном порядке):\n"
        for ev in events:
            _narrative_events_block += f"    - {ev}\n"

    narrative_prompt = f"""
    ТЫ: Сценарист.
    ЗАДАЧА: Напиши краткий сюжет (Narrative Arc) для диалога.

    ВХОДНЫЕ ДАННЫЕ (Seed):
    - Тема: {topic} ({theme})
    - Ключевые аспекты: {labels}
    - Профиль клиента: {user_profile_text}
    - Длительность: {timeline}
    {_narrative_events_block}
    ТРЕБОВАНИЕ:
    Напиши связную историю развития событий, которая ляжет в основу {num_batches} этапов диалога.
    Сюжет ОБЯЗАН включать все перечисленные события в их логическом порядке.
    Не пиши сам диалог, только сюжетную линию: завязка -> развитие -> кульминация -> развязка.
    """

    narrative_lambda = llm.invoke(narrative_prompt).content

    with open(f'{save_address}_narrative.txt', 'w', encoding='utf-8') as f:
        f.write(narrative_lambda)

    if noise_settings is None:
        noise_settings = {"enabled": False, "level": 0.0, "description": "", "distractor_enabled": False, "distractor_ratio": 0.0}

    noise_instruction = ""
    if noise_settings.get("enabled"):
        noise_level = noise_settings.get("level", 0.3)
        noise_desc = noise_settings.get("description", "противоречия в суммах, смена цели ипотеки, неверные даты документов")
        noise_instruction = f"""

        ═══════════════ ИНСТРУКЦИЯ ПО ГЕНЕРАЦИИ ШУМА ═══════════════
        В этом плане ТЫ ДОЛЖЕН намеренно создать логические ошибки примерно в {max(1, round(num_batches * noise_level))} этапах.
        Для таких пунктов используй пометку: [SEMANTIC NOISE].

        Типы разрешенного шума: {noise_desc}.
        Сделай так, чтобы противоречие было четким, но выглядело как естественная ошибка человека.
        """

    distractor_instruction = ""
    if noise_settings.get("distractor_enabled"):
        distractor_ratio = noise_settings.get("distractor_ratio", 0.15)
        distractor_instruction = f"""

        ═══════════════ ИНСТРУКЦИЯ ПО ОТВЛЕКАЮЩИМ ТЕМАМ ═══════════════
        Примерно {int(distractor_ratio * 100)}% буллетов должны содержать отвлекающие элементы:
        клиент уходит от темы (обсуждает погоду, жалуется на очереди, рассказывает о своих делах,
        задает нерелевантные вопросы). Пометь такие буллеты как [DISTRACTOR].
        """

    decoy_instruction = ""
    if noise_settings.get("decoy_enabled"):
        decoy_ratio = noise_settings.get("decoy_ratio", 0.3)
        decoy_instruction = f"""

        ═══════════════ ИНСТРУКЦИЯ ПО ФАКТАМ-ДВОЙНИКАМ (DECOY) ═══════════════
        Примерно {int(decoy_ratio * 100)}% буллетов должны содержать ФАКТЫ-ДВОЙНИКИ — данные,
        ОЧЕНЬ ПОХОЖИЕ на ключевые факты этапа, но отличающиеся в 1-2 деталях.

        ТИПЫ ДВОЙНИКОВ:
        1. **Суммы-двойники**: Если ключевой факт — перевод 47 500 руб., добавь рядом перевод 47 800 руб. другому человеку.
        2. **Даты-двойники**: Если заявка подана 15.03, добавь событие 15.04 того же типа.
        3. **Имена-двойники**: Если участник — Иванов А.С., добавь Иванова А.П. в другом контексте.
        4. **Параметры-двойники**: Если ставка 14.5%, добавь обсуждение ставки 14.8% по другому продукту.

        Цель: создать МАКСИМАЛЬНУЮ ИНТЕРФЕРЕНЦИЮ. Модель должна точно различать похожие факты.
        Пометь такие буллеты как [DECOY].
        ВАЖНО: Двойники должны стоять РЯДОМ с целевыми фактами (в том же батче), а не в отдельном блоке.
        """

    events_text = "Нет обязательных событий."
    if events:
        per_batch = len(events) / num_batches
        batch_events = []
        for b in range(num_batches):
            start = int(b * per_batch)
            end = int((b + 1) * per_batch) if b < num_batches - 1 else len(events)
            batch_events.append(events[start:end])
        lines_ev = []
        for b_idx, b_evs in enumerate(batch_events):
            lines_ev.append(f"BATCH {b_idx + 1} — обязательные события:")
            for ev in b_evs:
                lines_ev.append(f"  - {ev}")
        events_text = "\n".join(lines_ev)

    family_instruction = ""
    if noise_settings and noise_settings.get("family_interference"):
        _level_map = {
            "easy": FAMILY_INTERFERENCE_EASY_INSTRUCTION,
            "medium": FAMILY_INTERFERENCE_MEDIUM_INSTRUCTION,
            "hard": FAMILY_INTERFERENCE_HARD_INSTRUCTION,
        }
        family_instruction = _level_map.get(
            noise_settings.get("family_level", "hard"),
            FAMILY_INTERFERENCE_HARD_INSTRUCTION,
        )

    prompt = plan_generation_prompt_profile_and_topic_given_detailed_template \
                 .replace("<topic>", topic) \
                 .replace("<theme>", theme) \
                 .replace("<timeline>", timeline) \
                 .replace("<num_batches>", str(num_batches)) \
                 .replace("<provided_labels>", labels) \
                 .replace("<user_profile>", user_profile_text) \
                 .replace("<user_relationships>", user_rels_text) \
                 .replace("<num_bullets>", str(num_bullets)) \
                 .replace("<events_list>", events_text) \
                 .replace("<task_type>", task_type or "general") \
             + f"\n\nОСНОВНОЙ СЮЖЕТ (NARRATIVE ARC):\n{narrative_lambda}\n\nИспользуй этот сюжет как основу для плана." \
             + noise_instruction \
             + distractor_instruction \
             + decoy_instruction \
             + family_instruction

    print(f"--- [Algorithm Step 4] Генерация Плана P (используя Lambda) ---")

    def _parse_plans(text):
        batches = re.split(r'((?:#{1,3}\s*)?BATCH\s+\d+\s*(?:[:\-]\s*)?PLAN\w*)', text, flags=re.IGNORECASE)
        result = []
        for i in range(1, len(batches), 2):
            if i + 1 < len(batches):
                result.append(f"{batches[i].strip()}\n{batches[i+1].strip()}")
        return result

    def _check_markers(plans_list, ns):
        """Возвращает список типов маркеров, которые полностью отсутствуют."""
        full = "\n".join(plans_list)
        missing = []
        if ns.get("enabled") and len(re.findall(r'\[SEMANTIC[\s_]NOISE\]', full, re.IGNORECASE)) == 0:
            missing.append("SEMANTIC NOISE")
        if ns.get("distractor_enabled") and len(re.findall(r'\[DISTRACTOR\]', full, re.IGNORECASE)) == 0:
            missing.append("DISTRACTOR")
        if ns.get("decoy_enabled") and len(re.findall(r'\[DECOY\]', full, re.IGNORECASE)) == 0:
            missing.append("DECOY")
        return missing

    MAX_PLAN_RETRIES = 3
    plans = []
    current_prompt = prompt

    for plan_attempt in range(MAX_PLAN_RETRIES):
        _raw = llm.invoke(current_prompt)
        response = _raw.content if (_raw and _raw.content) else ""
        if not response.strip():
            print(f"❌ LLM вернул пустой ответ при генерации плана (попытка {plan_attempt+1})")
            continue
        plans = _parse_plans(response)

        if len(plans) == 0:
            print(f"❌ ОШИБКА: Парсинг плана не нашёл ни одного BATCH X PLAN (попытка {plan_attempt+1}).")
            continue
        if len(plans) < num_batches:
            print(f"⚠️ WARNING: Получено {len(plans)} батчей вместо {num_batches} (попытка {plan_attempt+1}).")
            if plan_attempt < MAX_PLAN_RETRIES - 1:
                current_prompt = prompt + (
                    f"\n\n⚠️ КРИТИЧЕСКИ ВАЖНО: в предыдущем ответе было только {len(plans)} батчей "
                    f"вместо требуемых {num_batches}. Сгенерируй РОВНО {num_batches} секций "
                    f"с заголовками 'BATCH 1 PLAN', 'BATCH 2 PLAN', ..., 'BATCH {num_batches} PLAN'."
                )
                continue

        ns = noise_settings or {}
        missing_markers = _check_markers(plans, ns) if any([
            ns.get("enabled"), ns.get("distractor_enabled"), ns.get("decoy_enabled")
        ]) else []

        if not missing_markers:
            break

        if plan_attempt < MAX_PLAN_RETRIES - 1:
            missing_str = ", ".join(f"[{m}]" for m in missing_markers)
            print(f"⚠️ ПЛАН RETRY {plan_attempt+1}: Отсутствуют маркеры {missing_str}. Усиливаю инструкцию...")
            retry_injection = f"""

⚠️ КРИТИЧЕСКИ ВАЖНО: В предыдущей попытке в плане полностью отсутствовали маркеры: {missing_str}.
Без этих маркеров план НЕ БУДЕТ ПРИНЯТ. Обязательно расставь их в тексте батчей:
"""
            if "SEMANTIC NOISE" in missing_markers:
                expected = max(1, round(num_batches * ns.get("level", 0.3)))
                retry_injection += f"- Минимум {expected} пунктов ДОЛЖНЫ заканчиваться пометкой [SEMANTIC NOISE]\n"
            if "DISTRACTOR" in missing_markers:
                expected = max(1, round(num_batches * ns.get("distractor_ratio", 0.2)))
                retry_injection += f"- Минимум {expected} пунктов ДОЛЖНЫ заканчиваться пометкой [DISTRACTOR]\n"
            if "DECOY" in missing_markers:
                expected = max(1, round(num_batches * ns.get("decoy_ratio", 0.3)))
                retry_injection += f"- Минимум {expected} пунктов ДОЛЖНЫ заканчиваться пометкой [DECOY]\n"
            retry_injection += "Пример правильной строки: '• Перевод 5 000 руб. Ивану Петрову 15.01. [DECOY]'\n"
            current_prompt = prompt + retry_injection
        else:
            print(f"⚠️ После {MAX_PLAN_RETRIES} попыток маркеры {missing_markers} так и не появились. Используем последний вариант.")

    if plans:
        full_plan = "\n".join(plans)
        ns = noise_settings or {}
        for marker, key, ratio_key in [
            ("SEMANTIC NOISE", "enabled", "level"),
            ("DISTRACTOR", "distractor_enabled", "distractor_ratio"),
            ("DECOY", "decoy_enabled", "decoy_ratio"),
        ]:
            if ns.get(key):
                count = len(re.findall(rf'\[{marker}\]', full_plan, re.IGNORECASE))
                expected = max(1, round(len(plans) * ns.get(ratio_key, 0.0)))
                status = "✅" if count > 0 else "❌"
                print(f"  {status} [{marker}]: {count} (ожидалось ~{expected})")

    _save_dir = os.path.dirname(save_address)
    if _save_dir:
        os.makedirs(_save_dir, exist_ok=True)
    with open(f'{save_address}.pickle', 'wb') as fb:
        pickle.dump(plans, fb)

    with open(f'{save_address}.txt', 'w', encoding='utf-8') as ft:
        ft.write(response)

    return plans

def extract_plan_bullets_text(plan_text: str) -> list[str]:
    """Извлекает текст буллетов, очищая от Markdown"""
    if not plan_text:
        return []

    strict_pattern = r"(?:•|-|\*|\d+\.)\s*\**(.+?)\**:\s*(.+)$"
    bullets = re.findall(strict_pattern, plan_text, re.MULTILINE)

    if bullets:
        return [f"{cat.strip()}: {desc.strip()}" for cat, desc in bullets]

    simple_pattern = r"^\s*(?:•|-|\*|\d+\.)\s*(.+)$"
    simple_bullets = re.findall(simple_pattern, plan_text, re.MULTILINE)

    return [b.strip() for b in simple_bullets if len(b) > 5]

def process_single_batch_messages(batch_idx, cur_plan, batch_size, sub_batches_per_batch,
                                   profile_context: str = ""):
    try:
        plan_bullets = extract_plan_bullets_text(cur_plan)
        if not plan_bullets:
            print(f"⚠️ Батч {batch_idx + 1}: пункты плана не найдены.")
            return None

        time_anchor = extract_time_anchor(plan_bullets[0])
        has_noise = bool(re.search(r'\[SEMANTIC[\s_]NOISE\]', cur_plan, re.IGNORECASE))

        k, m = divmod(len(plan_bullets), sub_batches_per_batch)
        bullet_groups = [plan_bullets[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in
                         range(sub_batches_per_batch)]

        batch_messages_text = []
        previous_context = ""

        for sub_idx, group in enumerate(bullet_groups):
            if not group: continue

            focused_bullets_text = "\n".join([f"{i + 1}) {b}" for i, b in enumerate(group)])

            noise_context = ""
            if has_noise:
                noise_context = "\nВАЖНО: В этом этапе клиент совершает ошибку. Отыграй её, не исправляй."

            prompt = message_generation_prompt_focused_template \
                         .replace("<PROFILE_CONTEXT>", profile_context) \
                         .replace("<FOCUSED_BULLETS>", focused_bullets_text) \
                         .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_context) \
                         .replace("<SUB_BATCH_SIZE>", str(batch_size)) \
                     + f"{noise_context}\n" \
                       f"ОБЯЗАТЕЛЬНО: В конце каждой фразы пиши ' ->-> ' и номер пункта плана.\n" \
                       f"ПРИМЕР: 'Я хочу открыть вклад ->-> 1'"

            MAX_MSG_RETRIES = 2
            accepted_msgs = []

            for attempt in range(MAX_MSG_RETRIES):
                response = llm.invoke(prompt).content

                raw_msgs = response.split(
                    "---MESSAGE_SEPARATOR---") if "---MESSAGE_SEPARATOR---" in response else response.split("\n")

                candidate_msgs = []
                for msg in raw_msgs:
                    if not msg.strip(): continue

                    if "->->" in msg:
                        parts = msg.rsplit("->->", 1)
                        text_part = parts[0].strip()
                        bullet_id = re.sub(r'[^\d]', '', parts[1]) or "?"
                    else:
                        text_part = msg.strip()
                        bullet_id = "?"

                    cleaned = re.sub(r'^\d+[\).]\s*|^(User|Клиент):\s*', '', text_part, flags=re.IGNORECASE)

                    if len(cleaned) > 5:
                        candidate_msgs.append(f"{cleaned} [ID:{bullet_id}]")

                if not candidate_msgs:
                    continue

                clean_texts = [re.sub(r'\[ID:[^\]]+\]', '', m).strip() for m in candidate_msgs]
                if check_message_quality(clean_texts, focused_bullets_text):
                    accepted_msgs = candidate_msgs
                    break
                else:
                    accepted_msgs = candidate_msgs  # сохраняем на случай провала всех попыток
                    if attempt < MAX_MSG_RETRIES - 1:
                        print(f"⚠️ Батч {batch_idx + 1}, под-батч {sub_idx + 1}: низкое качество сообщений, перегенерация ({attempt + 1}/{MAX_MSG_RETRIES})...")
                    else:
                        print(f"⚠️ Батч {batch_idx + 1}, под-батч {sub_idx + 1}: сообщения добавлены без подтверждения качества после {MAX_MSG_RETRIES} попыток")

            if not accepted_msgs:
                print(f"⚠️ Батч {batch_idx + 1}, под-батч {sub_idx + 1}: не удалось сгенерировать сообщения после {MAX_MSG_RETRIES} попыток — под-батч пропущен")
            batch_messages_text.extend(accepted_msgs)
            previous_context += f"\nОтработано: {focused_bullets_text}"

        return {
            "batch_index": batch_idx + 1,
            "time_anchor": time_anchor,
            "messages": batch_messages_text,
            "contains_noise": has_noise
        }

    except Exception as e:
        print(f"❌ Критическая ошибка в Batch {batch_idx + 1}: {e}")
        return None

def user_messages_generation(plans: list, save_address: str,
                                      sub_batches_per_batch: int = 3,
                                      batch_size: int = 5,
                                      max_workers: int = 4,
                                      chat_dir: Optional[str] = None):
    print(f"--- Запуск ПАРАЛЛЕЛЬНОЙ генерации ({len(plans)} этапов) ---")

    profile_context = ""
    if chat_dir:
        profile_path = os.path.join(chat_dir, "user_profile.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                main_spec = profile_data.get("main_spec", {})
                relationships = profile_data.get("relationships", {})
                rel_labels = {
                    "parent": "Родитель", "partner": "Супруг/Супруга",
                    "children": "Ребёнок", "friends": "Друг", "acquaintances": "Знакомый"
                }
                lines = [
                    f"Клиент: {main_spec.get('name', '?')} "
                    f"({main_spec.get('age', '?')} лет, {main_spec.get('job_title', '?')})"
                ]
                for rel_type, people in relationships.items():
                    label = rel_labels.get(rel_type, rel_type)
                    for p in people:
                        lines.append(f"- {label}: {p['name']} ({p['age']} лет)")
                lines.append(
                    "ВАЖНО: если клиент упоминает родственника или знакомого — "
                    "используй имена из этого списка. Не придумывай новых персонажей."
                )
                profile_context = "\n".join(lines)
            except Exception:
                pass

    all_messages = [None] * len(plans)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_batch_messages, i, plan, batch_size,
                            sub_batches_per_batch, profile_context): i
            for i, plan in enumerate(plans)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result:
                    all_messages[idx] = result
                    print(f"✅ Этап {idx + 1} завершен")
            except Exception as exc:
                print(f"❌ Этап {idx + 1} упал с исключением: {exc}")

    failed = [i for i, m in enumerate(all_messages) if m is None]
    if failed:
        raise RuntimeError(
            f"Генерация сообщений провалилась для батчей {[i + 1 for i in failed]}. "
            f"Исправьте ошибки и перезапустите --stage messages."
        )

    os.makedirs(os.path.dirname(save_address), exist_ok=True)
    with open(save_address, 'wb') as f:
        pickle.dump(all_messages, f)

    print(f"🚀 Параллельная генерация окончена. Сохранено в {save_address}")

def check_message_quality(messages: list, plan_bullets_text: str) -> bool:
    """Проверяет качество пакета сообщений клиента.

    Стратегия: сначала дешёвые rule-based проверки.
    LLM вызывается ТОЛЬКО при явном браке — экономия токенов.
    В норме (>90% случаев) LLM не вызывается вообще.
    """
    if not messages or all(len(m.strip()) < 10 for m in messages):
        return False

    operator_markers = ("оператор:", "менеджер:", "банк:", "сотрудник:", "assistant:")
    avg_len = sum(len(m) for m in messages) / len(messages)

    has_role_violation = all(
        any(m.lower().startswith(marker) for marker in operator_markers) for m in messages
    )
    has_too_short = avg_len < 15
    has_technical_jargon = any(
        phrase in m.lower() for m in messages
        for phrase in ("инициировать процедуру", "осуществить транзакцию", "произвести верификацию")
    )

    if not has_role_violation and not has_too_short and not has_technical_jargon:
        return True

    messages_text = "\n".join([f"- {m}" for m in messages])
    prompt = f"""Ты — контролёр качества банковского чата. Оцени сообщения клиента.

ПУНКТЫ ПЛАНА (что должны отражать сообщения):
{plan_bullets_text}

СГЕНЕРИРОВАННЫЕ СООБЩЕНИЯ КЛИЕНТА:
{messages_text}

Проверь по 3 критериям:
1. ПОКРЫТИЕ — сообщения в совокупности отражают содержание пунктов плана.
2. РОЛЬ — текст написан от лица клиента (не оператора, не LLM).
3. ЕСТЕСТВЕННОСТЬ — сообщения звучат как реальные фразы клиента банка.

Если ВСЕ 3 критерия выполнены — ответь YES. Иначе — NO.
Ответь ТОЛЬКО одним словом: YES или NO."""

    try:
        res = validator_llm.invoke(prompt).content.strip().lower()
        return "yes" in res
    except Exception:
        return True  # При ошибке API принимаем — не блокируем генерацию

def check_answer_quality(assistant_response: str, user_message: str,
                         current_plan_text: str = "", has_noise: bool = False,
                         difficulty: str = "easy",
                         last_response: str = "",
                         recent_turns: Optional[list] = None) -> bool:
    """Проверяет качество ответа оператора.

    easy: 3 базовых критерия.
    medium/hard: +4. ТОЧНОСТЬ ФАКТОВ (числа и даты из плана).
    medium/hard + has_noise: +5. ОБРАБОТКА ШУМА (оператор фиксирует противоречие).
    hard: минимум 4 предложения, нет повтора блоков, нет выхода из роли.
    recent_turns: последние 2-3 пары реплик для проверки повторов и противоречий.

    Возвращает True если ответ проходит проверку, False если нужна перегенерация.
    """
    if not assistant_response or len(assistant_response.strip()) < 20:
        return False

    extra_criteria = ""
    plan_block = ""

    if difficulty in ("medium", "hard") and current_plan_text:
        plan_block = f"\nПЛАН БАТЧА (контекст для оценки ответа):\n{current_plan_text[:1500]}\n"
        extra_criteria += (
            "\n4. ТОЧНОСТЬ ФАКТОВ — если оператор называет конкретные числа, суммы или даты, "
            "они НЕ должны противоречить плану батча. "
            "Если оператор не называет цифр — критерий считается выполненным."
        )
        if has_noise and difficulty == "hard":
            extra_criteria += (
                "\n5. ОБРАБОТКА ШУМА — в плане есть противоречивый или ошибочный факт [SEMANTIC NOISE]. "
                "Оператор должен зафиксировать несоответствие или запросить уточнение, "
                "а не молча подтверждать ошибочную информацию."
            )

    resp = assistant_response.strip()
    resp_lower = resp.lower()

    if len(resp) < 50:
        return False

    role_break_phrases = (
        "не имею доступа к счётам", "не имею доступа к данным",
        "не являюсь сотрудником", "я языковая модель", "я искусственный интеллект",
        "я ии", "я ai", "как языковая модель", "как ai-ассистент",
        "не могу получить доступ к", "у меня нет доступа к",
    )
    if any(ph in resp_lower for ph in role_break_phrases):
        return False

    fact_echo_phrases = (
        "зафиксированные факты диалога:", "зафиксированные факты:",
        "для справки:", "напоминаю:", "сводка:",
    )
    if any(ph in resp_lower for ph in fact_echo_phrases):
        return False

    if re.match(r'^\*\*[^*]+\*\*', resp) and re.search(r'\n[-*•]', resp):
        return False

    deflection_phrases = (
        "обратитесь в отделение", "позвоните на горячую линию",
        "не могу помочь", "не имею информации", "уточните у менеджера",
    )
    if any(ph in resp_lower for ph in deflection_phrases) and len(resp) < 200:
        return False

    client_role_markers = ("я клиент", "я хочу", "я прошу", "моя карта", "мой счёт")
    if any(resp_lower.startswith(m) for m in client_role_markers):
        return False

    if resp.startswith("```") or resp.startswith("{") or resp.startswith("["):
        return False

    if difficulty == "hard":
        sentence_count = len(re.findall(r'[.!?]+', resp))
        if sentence_count < 4:
            return False

    if difficulty == "hard" and last_response:
        prev = last_response.strip().lower()
        cur = resp_lower
        fingerprint = prev[:60]
        if len(fingerprint) >= 30 and fingerprint in cur:
            return False

    if not extra_criteria:
        return True

    if difficulty == "medium" and not re.search(r'\d', resp):
        return True

    num_noise_criteria = extra_criteria.count("\n5.")
    criteria_count = 4 + num_noise_criteria

    hard_criteria = ""
    if difficulty == "hard":
        criteria_count += 1
        hard_criteria = (
            f"\n{criteria_count}. КОНКРЕТНОСТЬ — ответ содержит хотя бы один конкретный факт "
            "(число, дату, имя, процент или действие). Общие фразы без деталей — FAIL."
        )

    recent_block = ""
    if recent_turns:
        lines = []
        for m in recent_turns:
            role = "Клиент" if m.get("role") == "user" else "Оператор"
            lines.append(f"{role}: {m.get('content', '')[:300]}")
        recent_block = "\nПОСЛЕДНИЕ РЕПЛИКИ ДИАЛОГА:\n" + "\n".join(lines) + "\n"

    prompt = f"""Ты — контролёр качества банковского чата. Оцени ответ оператора.
{plan_block}{recent_block}
СООБЩЕНИЕ КЛИЕНТА: "{user_message}"
ОТВЕТ ОПЕРАТОРА: "{assistant_response}"

Проверь по {criteria_count} критериям:
1. РЕЛЕВАНТНОСТЬ — ответ относится к вопросу клиента.
2. СОДЕРЖАТЕЛЬНОСТЬ — ответ содержит конкретную информацию.
3. РОЛЬ — оператор не ломает роль (не говорит "я не сотрудник", "нет доступа к счёту").{extra_criteria}{hard_criteria}

Если ВСЕ {criteria_count} критерия выполнены — ответь YES. Иначе — NO.
Ответь ТОЛЬКО одним словом: YES или NO."""

    try:
        res = validator_llm.invoke(prompt).content.strip().lower()
        return "yes" in res
    except Exception:
        return True  # При ошибке API принимаем

def answer_generation(input_address: str, output_address: str,
                      plans_address: str, topic: str, chat_size: str = "easy",
                      token_range: Optional[list] = None):
    print(f"--- Запуск симуляции ответов ({chat_size}) с валидацией ---")

    if token_range:
        token_min, token_max = token_range[0], token_range[1]
    else:
        token_defaults = {"easy": (20_000, 25_000), "medium": (50_000, 60_000), "hard": (90_000, 150_000)}
        token_min, token_max = token_defaults.get(chat_size, (20_000, 25_000))

    verbosity_map = {
        "easy": (
            f"Давай краткие, но содержательные ответы (2-4 предложения).\n"
            f"ЦЕЛЕВОЙ РАЗМЕР ДИАЛОГА: {token_min:,}-{token_max:,} токенов. Ориентируйся на этот объём."
        ),
        "medium": (
            f"Давай умеренно подробные ответы (4-6 предложений).\n"
            f"Объясняй термины и условия.\n"
            f"ЦЕЛЕВОЙ РАЗМЕР ДИАЛОГА: {token_min:,}-{token_max:,} токенов. Ориентируйся на этот объём."
        ),
        "hard": (
            f"Давай развёрнутые ответы (8-12 предложений).\n"
            f"Подробно объясняй процедуры, сроки, юридические нюансы.\n"
            f"ЦЕЛЕВОЙ РАЗМЕР ДИАЛОГА: {token_min:,}-{token_max:,} токенов. Ориентируйся на этот объём."
        ),
    }
    v_instr = verbosity_map.get(chat_size,
        f"Отвечай в естественном стиле. ЦЕЛЕВОЙ РАЗМЕР ДИАЛОГА: {token_min:,}-{token_max:,} токенов."
    )

    with open(plans_address, 'rb') as f:
        plans = pickle.load(f)
    with open(input_address, 'rb') as f:
        user_batches = pickle.load(f)

    chat_dir = os.path.dirname(output_address)
    profile_context_text = ""
    profile_path = os.path.join(chat_dir, "user_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        main_spec = profile_data.get("main_spec", {})
        relationships = profile_data.get("relationships", {})
        rel_labels = {
            "parent": "Родитель", "partner": "Супруг/Супруга",
            "children": "Ребёнок", "friends": "Друг", "acquaintances": "Знакомый"
        }
        lines = ["═══ УЧАСТНИКИ ДИАЛОГА (используй эти имена, не придумывай новых) ═══"]
        lines.append(
            f"Клиент: {main_spec.get('name', '?')} "
            f"({main_spec.get('age', '?')} лет, {main_spec.get('job_title', '?')})"
        )
        for rel_type, people in relationships.items():
            label = rel_labels.get(rel_type, rel_type)
            for p in people:
                lines.append(f"{label}: {p['name']} ({p['age']} лет)")
        lines.append(
            "ВАЖНО: когда клиент упоминает перевод, подарок или операцию с кем-то — "
            "используй имена из этого списка. Не придумывай новых персонажей."
        )
        profile_context_text = "\n".join(lines)

    final_chat_history = []
    global_id = 0
    assistant_memory = ConversationSummaryBuffer(llm, recent_messages_count=10)

    for batch_idx, batch_data in enumerate(user_batches):
        print(f"📦 Обработка этапа {batch_idx + 1}/{len(user_batches)}...")

        current_plan_text = plans[batch_idx] if batch_idx < len(plans) else ""
        msgs = batch_data.get('messages') or []
        time_anchor = batch_data.get('time_anchor', 'N/A')
        batch_turns = []
        last_assistant_response = ""  # отслеживание повторов для hard

        noise_instruction = ""
        if chat_size in ("medium", "hard"):
            noise_parts = []
            if re.search(r'\[SEMANTIC[\s_]NOISE\]', current_plan_text, re.IGNORECASE):
                noise_parts.append(
                    "⚠️ СЕМАНТИЧЕСКИЙ ШУМ: В этом батче клиент сообщает противоречивую или ошибочную "
                    "информацию. Оператор фиксирует это в ответе (уточняет, запрашивает подтверждение), "
                    "но не исправляет самовольно."
                )
            if re.search(r'\[DISTRACTOR\]', current_plan_text, re.IGNORECASE):
                noise_parts.append(
                    "⚠️ ОТВЛЕКАЮЩИЙ ФАКТ: В этом батче клиент отвлекается на постороннюю тему. "
                    "Оператор кратко реагирует и мягко возвращает разговор к основному вопросу."
                )
            if re.search(r'\[DECOY\]', current_plan_text, re.IGNORECASE):
                noise_parts.append(
                    "⚠️ ФАКТ-ДВОЙНИК: В этом батче есть похожие факты (близкие суммы, даты или имена). "
                    "Оператор чётко называет именно тот факт, который указан в плане — не путает и не смешивает."
                )
            if noise_parts:
                noise_instruction = (
                    "\n═══ СПЕЦИАЛЬНЫЕ ИНСТРУКЦИИ ПО ШУМУ ═══\n"
                    + "\n".join(noise_parts)
                    + "\n═══════════════════════════════════════"
                )

        for raw_user_msg in msgs:
            mapping_match = re.search(r'\[ID:(.+?)\]', raw_user_msg)
            bullet_id = mapping_match.group(1) if mapping_match else "unknown"
            clean_user_msg = re.sub(r'\[ID:.+?\]', '', raw_user_msg).strip()

            assistant_memory.add_message("user", clean_user_msg)

            system_prompt = ai_assistant_llm_template \
                .replace("<topic>", topic) \
                .replace("<current_plan>", current_plan_text) \
                .replace("<verbosity_instruction>", v_instr) \
                .replace("<summary_context>", assistant_memory.summary or "Нет предыдущего контекста") \
                .replace("<profile_context>", profile_context_text) \
                .replace("<noise_instruction>", noise_instruction)

            full_context = assistant_memory.get_messages_for_llm(system_prompt)

            MAX_ANSWER_RETRIES = 4 if chat_size == "hard" else 2
            is_valid = False
            response = None

            for attempt in range(MAX_ANSWER_RETRIES):
                try:
                    _raw_resp = llm.invoke(full_context)
                    response = _raw_resp.content if (_raw_resp and _raw_resp.content) else ""
                    if not response.strip():
                        print(f"⚠️ Пустой ответ LLM на буллет {bullet_id} (попытка {attempt+1})")
                        continue
                except RuntimeError as e:
                    print(f"⚠️ API недоступен на буллет {bullet_id}: {e}")
                    break
                is_valid = check_answer_quality(
                    response, clean_user_msg,
                    current_plan_text=current_plan_text,
                    has_noise=bool(re.search(r'\[SEMANTIC[\s_]NOISE\]', current_plan_text, re.IGNORECASE)),
                    difficulty=chat_size,
                    last_response=last_assistant_response,
                    recent_turns=assistant_memory.messages[-6:] if assistant_memory.messages else None,
                )
                if is_valid:
                    break
                print(f"⚠️ Низкое качество ответа на буллет {bullet_id}, перегенерация ({attempt + 1}/{MAX_ANSWER_RETRIES})...")

            if not is_valid:
                msg = f"❌ Буллет {bullet_id} не прошёл валидацию после {MAX_ANSWER_RETRIES} попыток — пропускаем turn"
                print(msg)
                skipped_log = os.path.join(chat_dir, "skipped_turns.log")
                with open(skipped_log, "a", encoding="utf-8") as _sl:
                    _sl.write(msg + "\n")
                if assistant_memory.messages and assistant_memory.messages[-1]["role"] == "user":
                    assistant_memory.messages.pop()
                continue

            assistant_memory.add_message("assistant", response)
            last_assistant_response = response or ""  # обновляем fingerprint для следующего turn

            turn = [
                {
                    "role": "user",
                    "content": clean_user_msg,
                    "id": global_id,
                    "target_bullet_id": bullet_id,
                    "time_anchor": time_anchor
                },
                {
                    "role": "assistant",
                    "content": response,
                    "id": global_id + 1,
                    "metadata": {"quality_ok": is_valid}
                }
            ]
            global_id += 2
            batch_turns.append(turn)

        final_chat_history.append(batch_turns)

    total_turns = sum(len(batch) for batch in final_chat_history)
    if total_turns == 0:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Ни один turn не прошёл валидацию. Файл НЕ сохранён.")
        return None

    os.makedirs(os.path.dirname(output_address), exist_ok=True)
    with open(output_address, 'wb') as f:
        pickle.dump(final_chat_history, f)

    with open(output_address.replace('.pickle', '.json'), 'w', encoding='utf-8') as f:
        json.dump(final_chat_history, f, indent=2, ensure_ascii=False)

    all_chat_text = json.dumps(final_chat_history, ensure_ascii=False)
    return all_chat_text

MAX_TASKS_PER_TYPE = 1  # ровно 1 задача на тип

def _candidate_in_chat(candidate, chat_text: str) -> bool:
    """Проверяет, что ключевые факты кандидата есть в тексте чата.

    Извлекает значимые слова (длиннее 4 символов) из всех строковых полей
    кандидата и проверяет, что хотя бы 30% из них встречаются в чате.
    Числа проверяются отдельно как приоритетные факты.
    """
    if isinstance(candidate, dict):
        text = " ".join(str(v) for v in candidate.values() if isinstance(v, (str, list)))
    else:
        text = str(candidate)

    chat_lower = chat_text.lower()

    numbers = re.findall(r'\d[\d\s]*\d|\d+', text)
    if numbers:
        found_numbers = sum(1 for n in numbers if n.replace(" ", "") in chat_lower.replace(" ", ""))
        if found_numbers / len(numbers) >= 0.4:
            return True

    words = [w.lower() for w in re.findall(r'[а-яёa-z]{5,}', text, re.IGNORECASE)]
    if not words:
        return True

    matches = sum(1 for w in words if w in chat_lower)
    return matches / len(words) >= 0.3

def _check_hard_batch_position(candidate) -> bool:
    """Fix #4: Для hard проверяет, что кандидат из батчей 3-6 (анти-primacy/recency bias).

    Если batch_numbers не найден — пропускаем проверку (soft enforcement).
    Поддерживает форматы: прямое поле, вложенное в source_bullet, batch_a/batch_b для interference.
    """
    if not isinstance(candidate, dict):
        return True

    batch_num = None
    if "batch_numbers" in candidate:
        batch_num = candidate["batch_numbers"]
    elif "source_bullet" in candidate and isinstance(candidate["source_bullet"], dict):
        batch_num = candidate["source_bullet"].get("batch_numbers")
    elif "batch_a" in candidate:
        try:
            ba = int(candidate["batch_a"])
            bb = int(candidate.get("batch_b", ba))
            bc_raw = candidate.get("batch_c")
            bc = int(bc_raw) if bc_raw is not None else None
            return 3 <= ba <= 6 or 3 <= bb <= 6 or (bc is not None and 3 <= bc <= 6)
        except (TypeError, ValueError):
            return True

    if batch_num is None:
        return True

    try:
        return 3 <= int(batch_num) <= 6
    except (TypeError, ValueError):
        return True

def _token_overlap_ratio(answer, source):
    """Доля токенов ответа, найденных в источнике."""
    answer_tokens = set(answer.lower().split())
    source_tokens = set(source.lower().split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & source_tokens) / len(answer_tokens)

def _structural_check(qa_json, difficulty="easy"):
    """Этап 1: Структурная проверка — нужные поля и непустой ответ."""
    question = str(qa_json.get("question", "")).strip()
    answer = str(qa_json.get("answer", "")).strip()

    if not question or len(question) < 15:
        return False, "Вопрос слишком короткий или пустой"
    if not answer or len(answer) < 2:
        return False, "Ответ пустой"
    if answer.lower() in ("нет данных", "неизвестно", "n/a", "", "да", "нет"):
        return False, "Ответ — заглушка"
    if len(question.split()) < 4:
        return False, "Вопрос менее 4 слов"

    path = qa_json.get("reasoning_path", [])
    if not isinstance(path, list) or len(path) < 1:
        return False, "отсутствует reasoning_path (нужен список минимум из 1 шага)"
    min_steps = {"easy": 1, "medium": 2, "hard": 3}
    required = min_steps.get(difficulty, 1)
    if len(path) < required:
        return False, f"reasoning_path требует ≥{required} шагов для {difficulty}, найдено {len(path)}"
    return True, "OK"

def _decoy_grounding_check(qa_json: dict, chat_text: str, difficulty: str = "easy"):
    """Проверяет, что каждый decoy_answer приближённо присутствует в тексте чата.

    Числовые decoy: хотя бы одно число ≥3 цифр из decoy должно встречаться в чате.
    Строковые decoy: ≥30% значимых слов (≥4 символов) для easy/medium,
                     ≥50% для hard (Баг B: жёстче требования для сложных задач).

    Баг A исправлен: пустые/None decoy теперь считаются незаземлёнными.
    """
    decoys = qa_json.get("decoy_answers", [])
    if not isinstance(decoys, list) or not any(decoys):
        return True, "нет decoy_answers"

    word_overlap_threshold = 0.5 if difficulty == "hard" else 0.3

    chat_no_spaces = chat_text.replace(" ", "").lower()
    chat_lower = chat_text.lower()
    ungrounded = []

    for decoy in decoys:
        if decoy is None or str(decoy).strip() == "":
            ungrounded.append(str(decoy))
            continue
        decoy_str = str(decoy).strip()
        decoy_normalized = re.sub(r'[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?$', '', decoy_str).strip()
        date_matches = re.findall(r'\d{2}\.\d{2}\.\d{4}', decoy_normalized)
        if date_matches and any(d in chat_lower for d in date_matches):
            continue  # дата найдена в диалоге — считается заземлённой
        numbers = re.findall(r'\d{3,}', decoy_normalized.replace(" ", ""))
        if numbers:
            if not any(n in chat_no_spaces for n in numbers):
                ungrounded.append(decoy_str)
        else:
            words = [w.lower() for w in re.findall(r'[а-яёa-z]{4,}', decoy_str, re.IGNORECASE)]
            if words:
                hits = sum(1 for w in words if w in chat_lower)
                if hits / len(words) < word_overlap_threshold:
                    ungrounded.append(decoy_str)

    if ungrounded:
        return False, f"Decoy не найдены в диалоге: {ungrounded[:2]}"

    if difficulty in ("medium", "hard"):
        answer_str = str(qa_json.get("answer", ""))
        ans_nums = re.findall(r'\d+(?:[.,]\d+)?', answer_str.replace(' ', ''))
        if ans_nums:
            try:
                ans_val = float(ans_nums[0].replace(',', '.'))
                if ans_val > 0:
                    max_ratio = 4.0 if difficulty == "hard" else 7.0
                    too_far = []
                    for decoy in decoys:
                        d_str = str(decoy).strip()
                        d_nums = re.findall(r'\d+(?:[.,]\d+)?', d_str.replace(' ', ''))
                        if not d_nums:
                            continue
                        try:
                            d_val = float(d_nums[0].replace(',', '.'))
                        except ValueError:
                            continue
                        if d_val <= 0:
                            continue
                        ratio = max(d_val / ans_val, ans_val / d_val)
                        if ratio > max_ratio:
                            too_far.append(f"{d_str} (x{ratio:.1f})")
                    if too_far:
                        return False, f"Decoy слишком далеко от ответа ({difficulty}, max ratio {max_ratio}x): {too_far[:2]}"
            except ValueError:
                pass

    return True, "все decoy заземлены"

def _grounding_check(answer, source_text, t_type):
    """Этап 2: Проверка обоснованности ответа в источнике.

    Для типов, где ответ вычисляется (temporal, composite, knowledge_update),
    используем мягкий порог. Для information_extraction и interference — строже.
    """
    if t_type == "composite":
        return True, "composite: логический вывод, overlap не применяется — передаём LLM"

    overlap = _token_overlap_ratio(answer, source_text)

    computed_types = {"temporal_reasoning", "knowledge_update"}
    if t_type in computed_types:
        if overlap >= 0.1:
            return True, f"overlap {overlap:.0%} (computed type, OK)"
        if t_type == "temporal_reasoning" and re.search(r'\d', answer):
            return True, "temporal answer contains numeric result"
        src_lower = source_text.lower()
        if answer.lower() in src_lower:
            return True, "substring match (computed)"
        nums_in_answer = re.findall(r'\d{3,}', answer)
        if nums_in_answer:
            nums_src_set = set(re.findall(r'\d+', src_lower))
            if all(n.replace(' ', '') in nums_src_set for n in nums_in_answer):
                return True, "numeric match (computed)"
        nums_short = re.findall(r'\d[\d.,]+', answer)
        if nums_short:
            src_compact = src_lower.replace(',', '.').replace(' ', '')
            if any(n.replace(',', '.').replace(' ', '') in src_compact for n in nums_short):
                return True, "short numeric match (computed)"
        answer_words = [re.sub(r'[^\w]', '', w) for w in answer.lower().split() if len(w) > 3]
        if answer_words:
            matched = sum(1 for stem in answer_words if stem and stem[:4] in src_lower)
            if matched >= max(1, len(answer_words) * 0.6):
                return True, f"stem match ({matched}/{len(answer_words)} слов, computed)"
        return False, f"overlap {overlap:.0%} < 10% для вычисляемого типа"

    if overlap >= 0.25:
        return True, f"overlap {overlap:.0%} >= 25%"

    if answer.lower() in source_text.lower():
        return True, "substring match"

    source_lower = source_text.lower()
    nums_in_answer = re.findall(r'\d{3,}', answer)
    if nums_in_answer:
        nums_src = re.findall(r'\d+', source_lower)
        nums_src_set = set(nums_src)
        compact = [n.replace(' ', '') for n in nums_in_answer]
        if all(n in nums_src_set or n.replace(' ', '') in nums_src_set for n in compact):
            return True, "numeric match"

    answer_words = [re.sub(r'[^\w]', '', w) for w in answer.lower().split() if len(w) > 3]
    if answer_words:
        matched = sum(1 for stem in answer_words if stem and stem[:4] in source_lower)
        if matched >= max(1, len(answer_words) * 0.6):
            return True, f"stem match ({matched}/{len(answer_words)} слов)"

    return False, f"overlap {overlap:.0%} < 25%, нет substring match"

def _difficulty_check(question, answer, source_text, t_type, difficulty):
    """Этап 2.5: Проверяет, что задача не тривиальна (бесплатно)."""
    q_lower = question.lower()
    a_lower = answer.lower()

    if a_lower in q_lower:
        return False, "Ответ содержится в вопросе"

    answer_words = set(a_lower.split())
    question_words = set(q_lower.split())
    if answer_words and answer_words.issubset(question_words):
        return False, "Все слова ответа есть в вопросе"

    min_q_len = {"easy": 10, "medium": 15, "hard": 15}
    if len(question) < min_q_len.get(difficulty, 10):
        return False, f"Вопрос слишком короткий для {difficulty}"

    max_a_words = {"easy": 20, "medium": 15, "hard": 12}
    composite_max = {"easy": 30, "medium": 25, "hard": 20}
    word_limit = composite_max.get(difficulty, 30) if t_type == "composite" else max_a_words.get(difficulty, 20)
    answer_words = re.findall(r'\w+', answer)
    if len(answer_words) > word_limit:
        return False, f"Ответ слишком длинный ({len(answer_words)} слов) для {difficulty}/{t_type}"

    if difficulty in ("medium", "hard"):
        trivial_starts = ["какая сумма", "когда было", "назови", "какой номер",
                          "сколько стоит", "какая дата", "какое имя"]
        for ts in trivial_starts:
            if q_lower.startswith(ts):
                return False, f"Шаблонное начало вопроса: '{ts}'"

    return True, "OK"

def _llm_validate(question, answer, source_text, t_type, difficulty):
    """Этап 3: LLM-валидация через ОТДЕЛЬНУЮ модель (validator_llm).

    Проверяет: grounding, answerability, non_trivial, format.
    Используется другая модель (например GPT-4o-mini) для независимой проверки.
    """
    from prompts import task_validation_prompt

    prompt = task_validation_prompt \
        .replace("<task_type>", t_type) \
        .replace("<difficulty>", difficulty) \
        .replace("<question>", question) \
        .replace("<answer>", answer) \
        .replace("<source_bullet>", source_text[:8000])

    if validator_llm is None:
        return True, "validator_llm не инициализирован — LLM-валидация пропущена (PASS)"

    try:
        raw = validator_llm.invoke(prompt).content
        clean = re.sub(r'```json|```', '', raw).strip()
        result = json.loads(repair_json(clean))

        verdict = str(result.get("verdict", "FAIL")).upper()
        reason = result.get("reason", "")

        if verdict == "PASS":
            return True, reason
        return False, reason
    except Exception as e:
        return True, f"LLM validation error (fallback PASS): {e}"

def _adversarial_validate(question, answer, t_type, difficulty):
    """Этап 5: Adversarial no-context check.
    Проверяет, угадывается ли ответ БЕЗ диалога по общим знаниям."""
    prompt = f"""Ты — участник теста. Тебе дан вопрос из банковского диалога.
Ответь на вопрос, используя ТОЛЬКО свои общие знания. У тебя НЕТ доступа к диалогу.

Вопрос: {question}
Тип задачи: {t_type}

Верни JSON:
{{
  "can_answer": true/false,
  "guessed_answer": "твой лучший вариант или null",
  "confidence": "high/low"
}}"""
    if validator_llm is None:
        return True, "validator_llm не инициализирован — adversarial check пропущен (PASS)"

    try:
        raw = validator_llm.invoke(prompt).content
        result = json.loads(repair_json(re.sub(r'```json|```', '', raw).strip()))
        can_answer = result.get("can_answer", False)
        guessed = str(result.get("guessed_answer", "")).strip().lower()
        confidence = result.get("confidence", "low")
        if can_answer and guessed and guessed != "null" and confidence == "high":
            gold_norm = answer.lower().strip()
            guessed_tokens = set(guessed.split())
            gold_tokens = set(gold_norm.split())
            if guessed_tokens and gold_tokens:
                intersection = guessed_tokens & gold_tokens
                precision = len(intersection) / len(guessed_tokens)
                recall = len(intersection) / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                if f1 >= 0.7:
                    return False, f"Задача угадывается без диалога (F1={f1:.2f}): модель ответила '{guessed}'"
        return True, "Задача не угадывается без контекста"
    except Exception as e:
        return True, f"Adversarial check пропущен: {e}"

def validate_task(qa_json, source_text, t_type, difficulty="easy"):
    question = str(qa_json.get("question", "")).strip()
    answer = str(qa_json.get("answer", "")).strip()

    ok, reason = _structural_check(qa_json, difficulty)
    if not ok:
        return False, f"[STRUCT] {reason}"

    ok, reason = _grounding_check(answer, source_text, t_type)
    if not ok:
        return False, f"[GROUND] {reason}"

    ok, reason = _difficulty_check(question, answer, source_text, t_type, difficulty)
    if not ok:
        return False, f"[DIFFICULTY] {reason}"

    if t_type == "interference" and difficulty == "hard":
        events = qa_json.get("source_bullet", {}).get("interfering_events", {})
        if not events.get("event_3"):
            return False, "[TRIPLE] interference/hard требует event_3 (тройная интерференция не задокументирована)"

    overlap = _token_overlap_ratio(answer, source_text)
    if overlap >= 0.6 and difficulty != "hard":
        return True, f"high-confidence grounding ({overlap:.0%}), skip LLM"

    ok, reason = _llm_validate(question, answer, source_text, t_type, difficulty)
    if not ok:
        return False, f"[LLM] {reason}"

    if difficulty in ("medium", "hard"):
        ok, adv_reason = _adversarial_validate(question, answer, t_type, difficulty)
        if not ok:
            return False, f"[Stage 5 Adversarial] {adv_reason}"

    return True, "PASS"

def generate_single_task_type(t_type, difficulty, selection_template, gen_prompt_template, full_plan_text, chat_text):
    try:
        print(f"🔍 [START] Анализ типа: {t_type} | сложность: {difficulty}")

        selection_prompt = selection_template.replace("<plan>", full_plan_text)
        MAX_SELECTION_RETRIES = 4
        candidates = []

        for sel_attempt in range(MAX_SELECTION_RETRIES):
            try:
                raw_response = llm.invoke(selection_prompt).content
                clean_json = re.sub(r'```json|```', '', raw_response).strip()
                if not clean_json:
                    print(f"⚠️ [{t_type}/{difficulty}] Selection вернул пустой ответ (попытка {sel_attempt+1})")
                    continue

                parsed = json.loads(repair_json(clean_json))
                if isinstance(parsed, dict):
                    for key in ("candidates", "events", "items", "results"):
                        if key in parsed and isinstance(parsed[key], list):
                            parsed = parsed[key]
                            break
                    else:
                        parsed = [parsed]
                if parsed and isinstance(parsed[0], list):
                    flat = []
                    for item in parsed:
                        if isinstance(item, list):
                            flat.extend(item)
                        else:
                            flat.append(item)
                    parsed = flat
                if parsed:
                    grounded = [c for c in parsed if _candidate_in_chat(c, chat_text)]
                    if difficulty == "hard":
                        batch_ok = [c for c in grounded if _check_hard_batch_position(c)]
                        if batch_ok:
                            grounded = batch_ok
                        elif grounded:
                            print(f"⚠️ [{t_type}/{difficulty}] Batch-фильтр убрал все кандидаты (нет батчей 3-6), повторяем попытку {sel_attempt+1}/{MAX_SELECTION_RETRIES}")
                            grounded = []  # очищаем, уходим на retry

                    if grounded:
                        candidates = grounded
                        break
                    elif difficulty == "hard":
                        print(f"⚠️ [{t_type}/{difficulty}] Все {len(parsed)} кандидатов не найдены в чате (hard), повторяем (попытка {sel_attempt+1})")
                        continue
                    else:
                        print(f"⚠️ [{t_type}/{difficulty}] Все {len(parsed)} кандидатов не найдены в чате, используем без фильтрации")
                        candidates = parsed
                        break
            except Exception as e:
                print(f"⚠️ [{t_type}/{difficulty}] Ошибка Selection (попытка {sel_attempt+1}): {e}")

        if not candidates:
            print(f"❌ [{t_type}/{difficulty}] Selection не вернул валидных кандидатов после {MAX_SELECTION_RETRIES} попыток")
            return t_type, difficulty, []

        tasks_for_type = []
        skipped_no_intermediate = 0

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            if len(tasks_for_type) >= MAX_TASKS_PER_TYPE:
                break

            bullet_text_for_validation = ""  # Текст, в котором будем искать ответ
            final_prompt = gen_prompt_template

            if t_type == "interference":
                fact_1 = cand.get('event_a') or cand.get('event_1', '') or str(cand)
                fact_2 = cand.get('event_b') or cand.get('event_2', '') or ''
                fact_3 = (cand.get('event_c') or cand.get('event_3', '')) if difficulty == "hard" else ''

                final_prompt = final_prompt \
                    .replace("<bullet_point_1>", str(fact_1)) \
                    .replace("<bullet_point_2>", str(fact_2)) \
                    .replace("<bullet_point_3>", str(fact_3))  # только в interference_gen_hard

                bullet_text_for_validation = f"{fact_1} {fact_2} {fact_3} {chat_text}".strip() or str(cand)

            elif t_type == "knowledge_update":
                old_f = cand.get('old_fact', '') or cand.get('old_value', '')
                new_f = cand.get('new_fact', '') or cand.get('new_value', '') or cand.get('bullet_points', '')
                intermediate = cand.get('intermediate_facts', [])
                if not isinstance(intermediate, list):
                    intermediate = []
                if difficulty == "hard" and not intermediate:
                    print(f"  ⏭️ [{t_type}/{difficulty}] Кандидат без intermediate_facts пропущен (hard)")
                    skipped_no_intermediate += 1
                    continue
                if intermediate:
                    intermediate_text = "\n".join(f"- {f}" for f in intermediate)
                else:
                    intermediate_text = "(промежуточных значений нет)"

                final_prompt = final_prompt \
                    .replace("<old_bullet_point>", str(old_f)) \
                    .replace("<intermediate_facts>", intermediate_text) \
                    .replace("<new_bullet_point>", str(new_f))

                bullet_text_for_validation = f"{old_f} {intermediate_text} {new_f}".strip() or str(cand)
            elif t_type == "composite":
                steps = cand.get('chain_steps', [])
                if isinstance(steps, str): steps = [steps]  # Защита от ошибок

                chain_text_str = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(steps)])

                final_prompt = final_prompt.replace("<chain_text>", chain_text_str)

                chat_context_limit = 15000 if difficulty == "hard" else 3000
                bullet_text_for_validation = chain_text_str + "\n\n--- КОНТЕКСТ ДИАЛОГА ---\n" + chat_text[:chat_context_limit]
            else:
                if isinstance(cand, dict):
                    bullet_text = cand.get('bullet_points') or cand.get('fact') or cand.get('text') or str(cand)
                else:
                    bullet_text = str(cand)

                final_prompt = final_prompt.replace("<bullet_point>", str(bullet_text))
                bullet_text_for_validation = str(bullet_text) + " " + chat_text

            final_prompt = final_prompt.replace("<conversation_turns>", chat_text)

            MAX_RETRIES = 4
            task_accepted = False

            for attempt in range(MAX_RETRIES):
                try:
                    qa_raw = llm.invoke(final_prompt).content
                    qa_clean = re.sub(r'```json|```', '', qa_raw).strip()
                    qa_json = json.loads(repair_json(qa_clean))
                    if isinstance(qa_json, list):
                        qa_json = qa_json[0] if qa_json else {}
                    if not isinstance(qa_json, dict) or "question" not in qa_json or "answer" not in qa_json:
                        raise ValueError(f"Нет обязательных полей question/answer: {list(qa_json.keys()) if isinstance(qa_json, dict) else type(qa_json)}")
                except Exception as e:
                    print(f"⚠️ [{t_type}/{difficulty}] Ошибка генерации (попытка {attempt+1}): {e}")
                    continue

                decoy_ok, decoy_reason = _decoy_grounding_check(qa_json, chat_text, difficulty)
                if not decoy_ok:
                    if attempt < MAX_RETRIES - 1:
                        print(f"  🔄 [{t_type}/{difficulty}] Decoy не заземлены: {decoy_reason}, перегенерация...")
                    else:
                        print(f"  👻 [{t_type}/{difficulty}] DROP: {decoy_reason}")
                    continue

                is_valid, val_reason = validate_task(qa_json, bullet_text_for_validation, t_type, difficulty)
                if is_valid:
                    qa_json['difficulty'] = difficulty
                    qa_json['capability'] = t_type
                    qa_json['source_bullet'] = cand
                    tasks_for_type.append(qa_json)
                    print(f"  ✅ [{t_type}/{difficulty}] Задача принята ({len(tasks_for_type)}/{MAX_TASKS_PER_TYPE}).")
                    task_accepted = True
                    break
                else:
                    if attempt < MAX_RETRIES - 1:
                        print(f"  🔄 [{t_type}/{difficulty}] Не прошла валидацию: {val_reason}, перегенерация...")
                    else:
                        print(f"  👻 [{t_type}/{difficulty}] DROP после {MAX_RETRIES} попыток: {val_reason}")

        if (not tasks_for_type and t_type == "knowledge_update" and difficulty == "hard"
                and skipped_no_intermediate > 0 and skipped_no_intermediate == len(candidates)):
            print(f"⚠️ [{t_type}/{difficulty}] Все {skipped_no_intermediate} кандидатов без intermediate_facts — повторяем selection (до 2 доп. попыток)")
            ku_retry_prompt = selection_prompt + (
                "\n\nВАЖНО: В предыдущем ответе поле intermediate_facts было пустым у всех кандидатов. "
                "Найди ТОЛЬКО параметры с явным паттерном A→B→A→C (3+ изменения). "
                "Поле intermediate_facts ОБЯЗАТЕЛЬНО должно содержать минимум 1 промежуточное значение."
            )
            for ku_attempt in range(2):
                try:
                    raw = llm.invoke(ku_retry_prompt).content
                    clean = re.sub(r'```json|```', '', raw).strip()
                    extra_parsed = json.loads(repair_json(clean))
                    if isinstance(extra_parsed, dict):
                        extra_parsed = [extra_parsed]
                    extra_with_intermediate = [
                        c for c in extra_parsed
                        if isinstance(c.get('intermediate_facts'), list) and len(c['intermediate_facts']) > 0
                    ]
                    if extra_with_intermediate:
                        print(f"  ✅ [{t_type}/{difficulty}] Доп. selection вернул {len(extra_with_intermediate)} кандидатов с intermediate_facts")
                        candidates = extra_with_intermediate
                        for cand in candidates:
                            if len(tasks_for_type) >= MAX_TASKS_PER_TYPE:
                                break
                            old_f = cand.get('old_fact', '') or cand.get('old_value', '')
                            new_f = cand.get('new_fact', '') or cand.get('new_value', '') or cand.get('bullet_points', '')
                            intermediate = cand.get('intermediate_facts', [])
                            intermediate_text = "\n".join(f"- {f}" for f in intermediate)
                            retry_gen_prompt = gen_prompt_template \
                                .replace("<old_bullet_point>", str(old_f)) \
                                .replace("<intermediate_facts>", intermediate_text) \
                                .replace("<new_bullet_point>", str(new_f)) \
                                .replace("<conversation_turns>", chat_text)
                            bullet_text_retry = f"{old_f} {intermediate_text} {new_f}".strip() or str(cand)
                            for attempt in range(MAX_RETRIES):
                                try:
                                    qa_raw = llm.invoke(retry_gen_prompt).content
                                    qa_clean = re.sub(r'```json|```', '', qa_raw).strip()
                                    qa_json = json.loads(repair_json(qa_clean))
                                    if isinstance(qa_json, list):
                                        qa_json = qa_json[0] if qa_json else {}
                                    if not isinstance(qa_json, dict) or "question" not in qa_json or "answer" not in qa_json:
                                        raise ValueError("Нет полей question/answer")
                                except Exception as e:
                                    print(f"⚠️ [{t_type}/{difficulty}] Доп. генерация (попытка {attempt+1}): {e}")
                                    continue
                                decoy_ok, decoy_reason = _decoy_grounding_check(qa_json, chat_text, difficulty)
                                if not decoy_ok:
                                    continue
                                is_valid, val_reason = validate_task(qa_json, bullet_text_retry, t_type, difficulty)
                                if is_valid:
                                    qa_json['difficulty'] = difficulty
                                    qa_json['capability'] = t_type
                                    qa_json['source_bullet'] = cand
                                    tasks_for_type.append(qa_json)
                                    print(f"  ✅ [{t_type}/{difficulty}] Задача принята (доп. retry).")
                                    break
                            if tasks_for_type:
                                break
                        break
                    else:
                        print(f"  ⚠️ [{t_type}/{difficulty}] Доп. selection (попытка {ku_attempt+1}): снова нет intermediate_facts")
                except Exception as e:
                    print(f"  ⚠️ [{t_type}/{difficulty}] Ошибка доп. selection (попытка {ku_attempt+1}): {e}")

        return t_type, difficulty, tasks_for_type

    except Exception as e:
        print(f"❌ Критическая ошибка потока {t_type}/{difficulty}: {e}")
        return t_type, difficulty, []

def generate_probing_questions(plan_address: str, chat_address: str, save_dir: str,
                               difficulty: str = "easy",
                               task_type: str = "information_extraction",
                               max_workers: int = 4):
    print(f"--- Генерация задач | difficulty={difficulty} | task_type={task_type} ---")

    with open(plan_address, 'rb') as f:
        plans = pickle.load(f)
    real_plans = [p for p in plans if p != "[SYNTHETIC PADDING BATCH]"]
    if not real_plans:
        print(f"❌ generate_probing_questions: все батчи плана синтетические, нет реальных фактов для задач")
        return []
    full_plan_text = "\n\n".join(real_plans)

    with open(chat_address, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)

    real_batch_indices = {i for i, p in enumerate(plans) if p != "[SYNTHETIC PADDING BATCH]"}
    chat_text = ""
    for batch_idx, batch in enumerate(chat_data):
        if batch_idx not in real_batch_indices:
            continue
        for turn in batch:
            for msg in turn:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                chat_text += f"{role}: {content}\n"

    selection_templates = {
        "information_extraction": {
            "easy": information_extraction_selection_easy,
            "medium": information_extraction_selection_medium,
            "hard": information_extraction_selection_hard,
        },
        "knowledge_update": {
            "easy": knowledge_update_selection_easy,
            "medium": knowledge_update_selection_medium,
            "hard": knowledge_update_selection_hard,
        },
        "temporal_reasoning": {
            "easy": temporal_reasoning_selection_easy,
            "medium": temporal_reasoning_selection_medium,
            "hard": temporal_reasoning_selection_hard,
        },
        "interference": {
            "easy": interference_selection_easy,
            "medium": interference_selection_medium,
            "hard": interference_selection_hard,
        },
        "composite": {
            "easy": composite_selection_easy,
            "medium": composite_selection_medium,
            "hard": composite_selection_hard,
        },
    }

    generation_templates = {
        "information_extraction": {
            "easy": information_extraction_gen_easy,
            "medium": information_extraction_gen_medium,
            "hard": information_extraction_gen_hard,
        },
        "knowledge_update": {
            "easy": knowledge_update_gen_easy,
            "medium": knowledge_update_gen_medium,
            "hard": knowledge_update_gen_hard,
        },
        "temporal_reasoning": {
            "easy": temporal_reasoning_gen_easy,
            "medium": temporal_reasoning_gen_medium,
            "hard": temporal_reasoning_gen_hard,
        },
        "interference": {
            "easy": interference_gen_easy,
            "medium": interference_gen_medium,
            "hard": interference_gen_hard,
        },
        "composite": {
            "easy": composite_gen_easy,
            "medium": composite_gen_medium,
            "hard": composite_gen_hard,
        },
    }

    valid_task_types = list(selection_templates.keys())
    if task_type not in selection_templates:
        raise ValueError(f"Неизвестный task_type={task_type!r}. Допустимые: {valid_task_types}")
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"Неизвестный difficulty={difficulty!r}. Допустимые: ['easy', 'medium', 'hard']")
    sel_template = selection_templates[task_type][difficulty]
    gen_template = generation_templates[task_type][difficulty]

    t_type_res, diff_res, tasks = generate_single_task_type(
        task_type, difficulty, sel_template, gen_template,
        full_plan_text, chat_text
    )

    all_tasks = {task_type: tasks}

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tasks.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)

    total_count = len(tasks)
    print(f"Задачи сохранены: {save_path} ({total_count} задач типа {task_type})")
    return all_tasks
