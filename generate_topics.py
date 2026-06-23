import json
import os
import argparse
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from dotenv import load_dotenv
from llm import gemini_base as llm
from profile_creation import create_profile

load_dotenv()

TASK_TYPES = [
    "information_extraction",
    "knowledge_update",
    "temporal_reasoning",
    "interference",
    "composite",
]

DIFFICULTY_PARAMS: dict[str, dict[str, Any]] = {
    "easy": {
        "token_range": [20_000, 25_000],
        "num_events": 10,
        "noise_level": 0.0,
    },
    "medium": {
        "token_range": [50_000, 60_000],
        "num_events": 20,
        "noise_level": 0.20,
    },
    "hard": {
        "token_range": [90_000, 100_000],
        "num_events": 30,
        "noise_level": 0.40,
    },
}

CATEGORY_CATALOG = {
    "lending": {
        "name": "Кредитование",
        "description": "Ипотека, потребительские кредиты, автокредиты, рефинансирование, реструктуризация, досрочное погашение",
    },
    "deposits": {
        "name": "Вклады и сбережения",
        "description": "Срочные вклады, накопительные счета, капитализация процентов, пролонгация, АСВ",
    },
    "cards": {
        "name": "Карты и счета",
        "description": "Дебетовые/кредитные карты, кэшбэк, перевыпуск, блокировка, лимиты, грейс-период",
    },
    "daily_banking": {
        "name": "Ежедневные операции",
        "description": "Переводы (СБП, SWIFT), оплата ЖКХ, мобильная связь, штрафы ГИБДД, выписки",
    },
    "security": {
        "name": "Безопасность и мошенничество",
        "description": "115-ФЗ, фишинг, чарджбэк, блокировка счёта, социальная инженерия, оспаривание транзакций",
    },
    "investments": {
        "name": "Инвестиции",
        "description": "ИИС, брокерский счёт, акции, облигации, дивиденды, налоги на инвестиции",
    },
    "insurance": {
        "name": "Страхование",
        "description": "Страхование жизни, имущества, КАСКО, ОСАГО, страховые случаи, выплаты",
    },
    "documents": {
        "name": "Документы и справки",
        "description": "2-НДФЛ, справки по форме банка, обновление паспортных данных, доверенности",
    },
    "tech_support": {
        "name": "Техподдержка",
        "description": "Мобильное приложение, интернет-банк, push-уведомления, сбои, биометрия",
    },
    "business": {
        "name": "Бизнес-банкинг",
        "description": "РКО, эквайринг, зарплатный проект, валютный контроль, кредитование МСП",
    },
    "forex": {
        "name": "Валютные операции",
        "description": "Обмен валюты, конвертация по курсу ЦБ, SWIFT/SEPA переводы, валютные счета, курсовая разница, ограничения 2022–2024 гг.",
    },
    "taxes": {
        "name": "Налоги и вычеты",
        "description": "Возврат НДФЛ (имущественный, социальный, инвестиционный вычет), налог на вклады и инвестиции, уведомления от ФНС, роль банка как налогового агента",
    },
    "pension": {
        "name": "Пенсионные накопления",
        "description": "Перевод накоплений в НПФ, обязательное пенсионное страхование, программа долгосрочных сбережений (ПДС), выплаты и правопреемники",
    },
    "credit_history": {
        "name": "Кредитная история",
        "description": "Запрос отчёта из БКИ, кредитный рейтинг, оспаривание ошибок, влияние просрочек, исправление истории, самозапрет на кредиты",
    },
    "leasing": {
        "name": "Лизинг",
        "description": "Автолизинг, лизинг оборудования для ИП/МСП, выкупная стоимость, график платежей, досрочный выкуп, расторжение договора",
    },
    "bankruptcy": {
        "name": "Банкротство физических лиц",
        "description": "Процедура банкротства через МФЦ и суд, реструктуризация долгов, мировое соглашение, последствия для счетов и кредитной истории",
    },
    "bnpl": {
        "name": "Рассрочка и BNPL",
        "description": "Сплит, Долями, 0-0-24, рассрочка от магазина через банк, досрочное закрытие, пересчёт, штрафы за просрочку, споры об условиях",
    },
    "selfemployed": {
        "name": "Самозанятые и НПД",
        "description": "Налог на профессиональный доход, формирование чеков через «Мой налог», ограничения банков для самозанятых, смена статуса ИП/самозанятый, привязка счёта",
    },
    "escrow": {
        "name": "Эскроу-счета",
        "description": "Обязательный эскроу при ДДУ (с 2019 г.), раскрытие средств застройщику после сдачи, задержки, банкротство застройщика, возврат средств покупателю",
    },
    "nsi_isi": {
        "name": "НСЖ и ИСЖ",
        "description": "Накопительное и инвестиционное страхование жизни, взносы и доходность, мисселлинг, досрочное расторжение, страховые случаи, выплаты и налогообложение",
    },
    "loyalty": {
        "name": "Программы лояльности",
        "description": "Мили, баллы (Спасибо, Кешбэк, Бонусы), партнёрские категории, сгорание бонусов, конвертация в рубли, споры о начислении и списании",
    },
    "digital_ruble": {
        "name": "Цифровой рубль",
        "description": "Цифровой рубль ЦБ РФ, открытие кошелька, пополнение и переводы, ограничения, смарт-контракты, работа с цифровым рублём через банковское приложение",
    },
    "pif": {
        "name": "ПИФы и БПИФ",
        "description": "Паевые инвестиционные фонды, БПИФ, покупка и погашение паёв, СЧА, управляющая компания, комиссии, налогообложение при погашении, льгота долгосрочного владения",
    },
    "mortgage_holidays": {
        "name": "Ипотечные и кредитные каникулы",
        "description": "Каникулы по закону 76-ФЗ (с 2019 г.) и по 106-ФЗ, условия: снижение дохода на 30%+, оформление, льготный период, возобновление графика, единственное жильё",
    },
    "mfo": {
        "name": "МФО и микрозаймы",
        "description": "Микрозаймы, рефинансирование займа МФО через банк, предельная ставка ЦБ, реестр МФО, взыскание, проблемы с коллекторами, жалобы в ЦБ",
    },
    "intl_transfers": {
        "name": "Международные переводы (санкционный контекст)",
        "description": "Переводы за рубеж в 2022–2025 гг.: ограничения SWIFT, альтернативы (СПФС, Юнистрим, переводы через третьи страны), лимиты ЦБ, документарное подтверждение, возвраты",
    },
}

TASK_TYPE_EVENT_INSTRUCTIONS = {
    "information_extraction": (
        "Сгенерируй события, содержащие КОНКРЕТНЫЕ ФАКТЫ с точными данными:\n"
        "- Суммы (например: 5 000 000 руб., комиссия 1 500 руб.)\n"
        "- Даты (например: 15.03.2024, до 25 числа каждого месяца)\n"
        "- Процентные ставки (например: 12.9% годовых)\n"
        "- Имена и адреса (например: Иванов Пётр Сергеевич, ул. Ленина 42)\n"
        "Каждое событие — отдельный атомарный факт, который можно проверить."
    ),
    "knowledge_update": (
        "Сгенерируй события, где ФАКТЫ МЕНЯЮТСЯ со временем:\n"
        "- Статусы: 'на рассмотрении' → 'одобрено' → 'подписано'\n"
        "- Ставки: 15% → 14.5% → 14.2%\n"
        "- Лимиты: 50 000 → 100 000 → 75 000 → 120 000\n"
        "- Суммы платежей: 25 000 → 23 500 (после перерасчёта)\n"
        "Каждая пара событий должна описывать OLD → NEW значение одного параметра."
    ),
    "temporal_reasoning": (
        "Сгенерируй события с ЧЁТКИМИ ДАТАМИ и временными зависимостями:\n"
        "- Явные даты: 'Заявка подана 01.03.2024', 'Одобрена 08.03.2024'\n"
        "- Относительные сроки: 'через 5 рабочих дней после подачи', 'в следующем месяце'\n"
        "- Дедлайны: 'документы нужно предоставить до 15 апреля'\n"
        "- Последовательности: событие A → через N дней → событие B → через M дней → событие C\n"
        "Между событиями должны быть вычислимые временные интервалы."
    ),
    "interference": (
        "Сгенерируй ПОХОЖИЕ события с мелкими различиями, которые легко перепутать:\n"
        "- Два перевода разным людям с близкими суммами (5 000 и 5 200 руб.)\n"
        "- Два платежа одного типа в разные даты (ЖКХ 25.01 и ЖКХ 25.02)\n"
        "- Два похожих имени (Иванов А.С. и Иванов А.П.)\n"
        "- Два кредитных предложения (14.5% на 15 лет и 14.8% на 15 лет)\n"
        "Цель: создать МАКСИМАЛЬНУЮ путаницу между похожими фактами."
    ),
    "composite": (
        "Сгенерируй ЦЕПОЧКИ причинно-следственных событий (A → B → C → D):\n"
        "- Запросил справку → Обнаружен долг → Оплатил долг → Справка выдана → Подал на ипотеку\n"
        "- Открыл вклад → Получил проценты → Перевёл на карту → Оплатил покупку\n"
        "- Заблокировали карту → Позвонил в банк → Прошёл верификацию → Карта разблокирована\n"
        "Каждая цепочка — логически связанная последовательность, где каждый шаг вытекает из предыдущего."
    ),
}

EVENTS_GENERATION_PROMPT = """
Ты — эксперт по банковскому обслуживанию в России. Сгенерируй банковский сценарий с событиями для тестирования эпизодической памяти LLM.

═══════════════ ПАРАМЕТРЫ ═══════════════
- КАТЕГОРИЯ: {category_name} ({category_desc})
- ТИП ЗАДАЧИ: {task_type}
- КОЛИЧЕСТВО СОБЫТИЙ: {num_events}

═══════════════ КЛИЕНТ ═══════════════
Главного клиента зовут {client_name}. Используй это имя (Фамилия Имя Отчество) во всех событиях, где упоминается главный клиент. Не придумывай другое имя для клиента.
Его близкие (если упоминаются в событиях — используй ТОЛЬКО эти имена, не придумывай других):
{relatives_text}

═══════════════ ТРЕБОВАНИЯ К СОБЫТИЯМ ═══════════════
{event_instructions}

═══════════════ ТРЕБОВАНИЯ К СЦЕНАРИЮ ═══════════════
1. Сценарий должен быть реалистичным для клиента российского банка.
2. `topic` — описание ситуации клиента (1 предложение).
3. `theme` — осложнение или главный конфликт (1 предложение).
4. `subtopics` — 4-6 ключевых слов.
5. `events` — ровно {num_events} событий, каждое — конкретный факт для диалога.
6. Используй российские реалии: рубли, ЦБ, НДФЛ, СБП, 115-ФЗ.

═══════════════ ФОРМАТ ВЫВОДА ═══════════════
Верни ТОЛЬКО валидный JSON-объект. Без Markdown, без комментариев.

{{
  "category": "{category_key}",
  "title": "Краткое название (3-5 слов)",
  "topic": "Описание ситуации клиента (1 предложение)",
  "theme": "Осложнение или главный конфликт (1 предложение)",
  "subtopics": ["слово1", "слово2", "слово3", "слово4"],
  "events": [
    "Событие 1 с конкретными данными",
    "Событие 2 с конкретными данными",
    "..."
  ]
}}
"""

def _parse_llm_json(raw: str):
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        return json.loads(repair_json(clean))
    except Exception:
        return None

def _events_overlap(events_a: list, events_b: list, threshold: float = 0.4) -> bool:
    COMMON_WORDS = {
        'карта', 'карту', 'карте', 'счёт', 'счёта', 'счёту', 'банк', 'банка', 'банке',
        'кредит', 'кредита', 'кредите', 'клиент', 'клиента', 'оператор', 'сумма',
        'сумму', 'платёж', 'платежа', 'лимит', 'лимита', 'ставка', 'ставки',
        'заявка', 'заявки', 'документ', 'договор', 'выписка', 'перевод', 'счет',
        'рублей', 'рубль', 'процент', 'годовых', 'месяц', 'срок', 'банке',
        'одобрено', 'отклонено', 'рассмотрении', 'подписано', 'открыт',
    }

    def specific_keywords(events):
        text = " ".join(str(e) for e in events).lower()
        words = set(re.findall(r'\b\w{4,}\b', text))
        return words - COMMON_WORDS

    kw_a = specific_keywords(events_a)
    kw_b = specific_keywords(events_b)
    if not kw_a or not kw_b:
        return False
    overlap = len(kw_a & kw_b) / min(len(kw_a), len(kw_b))
    return overlap > threshold

def _format_existing_context(existing_topics: list) -> str:
    if not existing_topics:
        return ""
    lines = []
    for t in existing_topics:
        title = t.get("title", "?")
        topic_desc = t.get("topic", "")
        events = t.get("events", [])
        events_preview = " | ".join(str(e) for e in events[:4])
        lines.append(f"• {title}: {topic_desc}\n  События: {events_preview}")
    return "\n".join(lines)

def _load_existing_topics(results_dir: str, difficulty: str) -> list:
    existing: list[dict] = []
    base = os.path.join(results_dir, "dialogue", difficulty)
    if not os.path.isdir(base):
        return existing
    for tt in TASK_TYPES:
        type_dir = os.path.join(base, tt)
        if not os.path.isdir(type_dir):
            continue
        for d in os.listdir(type_dir):
            if not d.isdigit():
                continue
            topic_path = os.path.join(type_dir, d, "topic.json")
            if os.path.exists(topic_path):
                try:
                    with open(topic_path, 'r', encoding='utf-8') as f:
                        existing.append(json.load(f))
                except Exception:
                    pass
    return existing

def _generate_single_topic(task_type: str, category_key: str, category_info: dict,
                           num_events: int, difficulty: str = "easy",
                           existing_context: str = "", existing_topics: Optional[list] = None,
                           chat_dir: Optional[str] = None) -> Optional[dict]:
    nc_prob = 0.3 if difficulty == "hard" else 0.0
    main_spec, relationships = create_profile(name_collision_prob=nc_prob)
    client_name = main_spec["name"]
    profile_data = {"main_spec": main_spec, "relationships": relationships}

    if chat_dir:
        os.makedirs(chat_dir, exist_ok=True)
        profile_path = os.path.join(chat_dir, "user_profile.json")
        if not os.path.exists(profile_path):
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=4, ensure_ascii=False)

    rel_labels = {
        "parent": "Родитель", "partner": "Супруг/Супруга",
        "children": "Ребёнок", "friends": "Друг", "acquaintances": "Знакомый",
    }
    relatives_lines = []
    for rel_type, people in relationships.items():
        label = rel_labels.get(rel_type, rel_type)
        for p in people:
            relatives_lines.append(f"- {label}: {p['name']} ({p['age']} лет)")
    relatives_text = "\n".join(relatives_lines) if relatives_lines else "Не указаны"

    base_prompt = EVENTS_GENERATION_PROMPT.format(
        category_name=category_info["name"],
        category_desc=category_info["description"],
        task_type=task_type,
        num_events=num_events,
        event_instructions=TASK_TYPE_EVENT_INSTRUCTIONS[task_type],
        category_key=category_key,
        client_name=client_name,
        relatives_text=relatives_text,
    )

    if existing_context:
        base_prompt += (
            f"\n\n═══════════════ УЖЕ СОЗДАННЫЕ ТЕМЫ (НЕ ПОВТОРЯТЬ) ═══════════════\n"
            f"{existing_context}\n\n"
            f"ВАЖНО: Твоя тема должна КАРДИНАЛЬНО отличаться от перечисленных выше "
            f"по сценарию, персонажу и событиям. Придумай совершенно другую ситуацию."
        )

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(base_prompt).content
            topic = _parse_llm_json(response)

            if topic is None:
                print(f"  Попытка {attempt}: не удалось распарсить JSON из ответа LLM")
                continue
            if not isinstance(topic, dict):
                print(f"  Попытка {attempt}: ожидался dict, получен {type(topic).__name__}")
                continue

            required_keys = {"category", "title", "topic", "theme", "subtopics", "events"}
            missing = required_keys - set(topic.keys())
            if missing:
                print(f"  Тема {task_type}: пропущены ключи {missing}")
                continue

            events_list = topic.get("events")
            if not isinstance(events_list, list) or len(events_list) == 0:
                print(f"  Тема {task_type}: пустой список событий")
                continue
            min_accepted = max(1, int(num_events * 0.7))  # принимаем ≥70% от требуемого
            if len(events_list) < min_accepted:
                print(f"  Тема {task_type}: слишком мало событий {len(events_list)} < {min_accepted} (нужно {num_events})")
                continue

            if existing_topics:
                new_events = topic["events"]
                duplicate_title = None
                for ex in existing_topics:
                    if _events_overlap(new_events, ex.get("events", []), threshold=0.4):
                        duplicate_title = ex.get("title", "?")
                        break
                if duplicate_title:
                    print(f"  ⚠️ {task_type} (попытка {attempt}): слишком похожа на «{duplicate_title}», повтор...")
                    continue

            topic["task_type"] = task_type
            topic["client_name"] = client_name
            return topic

        except Exception as e:
            print(f"  Ошибка генерации темы {task_type} (попытка {attempt}): {e}")

    print(f"  ✗ {task_type}: не удалось сгенерировать уникальную тему после {max_retries} попыток")
    return None

def generate_topics(count: int = 5, difficulty: str = "easy",
                    categories: Optional[list] = None, output_path: str = "topics.json",
                    results_dir: Optional[str] = None, only_task_types: Optional[list] = None,
                    chat_dir_base: Optional[str] = None):
    params = DIFFICULTY_PARAMS[difficulty]
    num_events: int = params["num_events"]

    if categories:
        selected = {k: v for k, v in CATEGORY_CATALOG.items() if k in categories}
        if not selected:
            print(f"Неизвестные категории: {categories}")
            print(f"Доступные: {list(CATEGORY_CATALOG.keys())}")
            return None
    else:
        selected = CATEGORY_CATALOG

    category_keys = list(selected.keys())

    start_seq = {}
    for tt in TASK_TYPES:
        if chat_dir_base:
            type_dir = os.path.join(chat_dir_base, tt)
        elif results_dir:
            type_dir = os.path.join(results_dir, "dialogue", difficulty, tt)
        else:
            type_dir = None
        if type_dir and os.path.isdir(type_dir):
            existing = [int(d) for d in os.listdir(type_dir) if d.isdigit()]
            start_seq[tt] = max(existing) + 1 if existing else 1
        else:
            start_seq[tt] = 1

    active_types = [tt for tt in TASK_TYPES if not only_task_types or tt in only_task_types]
    if not active_types:
        print(f"Ошибка: none из указанных task_types не найден в {TASK_TYPES}")
        return None

    base_per_type, extra = divmod(count, len(active_types))
    assignments = []
    topic_id = 1
    seq_counters = dict(start_seq)
    for i, task_type in enumerate(active_types):
        n = base_per_type + (1 if i < extra else 0)
        for j in range(n):
            cat_key = category_keys[(topic_id - 1) % len(category_keys)]
            assignments.append((topic_id, task_type, seq_counters[task_type], cat_key, selected[cat_key]))
            seq_counters[task_type] += 1
            topic_id += 1

    print(f"Генерация {count} тем (difficulty={difficulty}, events={num_events})...")
    print(f"  Типы: {active_types}")
    print(f"  Распределение: {', '.join(f'{tt}={base_per_type + (1 if i < extra else 0)}' for i, tt in enumerate(active_types))}")
    if results_dir:
        print(f"  Стартовые seq_id: {start_seq}")

    existing_topics = _load_existing_topics(results_dir, difficulty) if results_dir else []
    existing_context = _format_existing_context(existing_topics)
    if existing_topics:
        print(f"  Найдено существующих тем: {len(existing_topics)} — новые будут отличаться от них")

    valid_topics = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_info = {}
        for (tid, task_type, seq_id, cat_key, cat_info) in assignments:
            if chat_dir_base:
                chat_dir = os.path.join(chat_dir_base, task_type, str(seq_id))
            elif results_dir:
                chat_dir = os.path.join(results_dir, "dialogue", difficulty, task_type, str(seq_id))
            else:
                chat_dir = None
            future = executor.submit(
                _generate_single_topic, task_type, cat_key, cat_info, num_events,
                difficulty, existing_context, existing_topics, chat_dir
            )
            future_to_info[future] = (tid, task_type, seq_id, difficulty)

        for future in as_completed(future_to_info):
            tid, task_type, seq_id, diff = future_to_info[future]
            try:
                topic = future.result()
            except Exception as e:
                print(f"  [{tid}] {task_type}: EXCEPTION — {e}")
                continue
            if topic is not None:
                topic["id"] = tid
                topic["difficulty"] = diff
                topic["meta"] = {
                    "token_range": params["token_range"],
                    "num_events": num_events,
                    "noise_level": params["noise_level"],
                }
                valid_topics.append(topic)
                print(f"  [{tid}] {task_type}: {topic.get('title', '?')}")

                if chat_dir_base or results_dir:
                    if chat_dir_base:
                        save_dir = os.path.join(chat_dir_base, task_type, str(seq_id))
                    else:
                        assert results_dir is not None
                        save_dir = os.path.join(results_dir, "dialogue", difficulty, task_type, str(seq_id))
                    os.makedirs(save_dir, exist_ok=True)
                    topic_path = os.path.join(save_dir, "topic.json")
                    with open(topic_path, "w", encoding="utf-8") as f:
                        json.dump(topic, f, indent=4, ensure_ascii=False)
                    print(f"  💾 {task_type}/{seq_id}/topic.json")
            else:
                print(f"  [{tid}] {task_type}: FAILED")

    valid_topics.sort(key=lambda t: t["id"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_topics, f, indent=2, ensure_ascii=False)
    print(f"Сохранено {len(valid_topics)} тем в {output_path}")

    type_counts: dict[str, int] = {}
    for t in valid_topics:
        tt = t.get("task_type", "?")
        type_counts[tt] = type_counts.get(tt, 0) + 1
    print(f"  По типам: {type_counts}")

    return valid_topics

def parse_args():
    parser = argparse.ArgumentParser(description="Генератор банковских тем для BEAM")
    parser.add_argument("--count", type=int, default=5,
                        help="Количество тем для генерации (по умолчанию 5)")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Уровень сложности (по умолчанию easy)")
    parser.add_argument("--output", type=str, default="topics.json",
                        help="Путь к выходному файлу (по умолчанию topics.json)")
    parser.add_argument("--results_dir", type=str, default="data/results",
                        help="Базовая директория для сохранения topic.json по папкам (по умолчанию data/results)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=list(CATEGORY_CATALOG.keys()),
                        help="Ограничить генерацию выбранными категориями")
    parser.add_argument("--list-categories", action="store_true",
                        help="Показать доступные категории и выйти")
    parser.add_argument("--only_task_types", nargs="+", default=None,
                        choices=TASK_TYPES,
                        help="Генерировать только для указанных task_types (по умолчанию все 5)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.list_categories:
        print("Доступные категории:")
        for key, info in CATEGORY_CATALOG.items():
            print(f"  {key:<15} {info['name']}: {info['description']}")
    else:
        generate_topics(
            count=args.count,
            difficulty=args.difficulty,
            categories=args.categories,
            output_path=args.output,
            results_dir=args.results_dir,
            only_task_types=args.only_task_types,
        )
