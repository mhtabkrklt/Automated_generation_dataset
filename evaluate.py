import os
import re
import json
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from model_config import build_model, get_available_models
from metrics import (
    compute_all_metrics, run_llm_judge, should_call_judge,
    extract_answer_from_json, avg_metrics, final_accuracy,
)
from utils import get_eval_chat_dirs, invoke_with_retries


EVAL_SYSTEM_PROMPT = (
    "Ты — AI-ассистент, отвечающий на вопросы по истории диалога с банком.\n"
    "Твоя задача — ВНИМАТЕЛЬНО прочитать весь диалог и дать точный ответ на поставленный вопрос.\n"
    "ВАЖНО: Не копируй текст механически — убедись, что ты понимаешь контекст и отвечаешь именно на заданный вопрос.\n"
    "Для interference-задач: в диалоге может быть несколько похожих событий — найди именно то, о котором спрашивается.\n"
    "\n"
    "КРИТИЧЕСКИЕ ПРАВИЛА ОТВЕТА:\n"
    "1. Твой ответ должен содержать ТОЛЬКО JSON-объект — ничего до и ничего после.\n"
    "2. Никаких вступлений, объяснений, рассуждений, комментариев.\n"
    "3. Никаких markdown-блоков (```json ... ```).\n"
    "4. Ровно один ключ: \"answer\" со значением-строкой.\n"
    "5. Значение answer — короткое и точное, без лишних слов.\n"
    "\n"
    "ЕДИНСТВЕННО ДОПУСТИМЫЙ ФОРМАТ:\n"
    "{\"answer\": \"<значение>\"}\n"
    "\n"
    "СТРОГИЙ ФОРМАТ ЗНАЧЕНИЙ:\n"
    "  Сумма:    цифры без пробелов + ' руб.' → {\"answer\": \"30300 руб.\"}\n"
    "  Ставка:   X.X% с точкой               → {\"answer\": \"14.5%\"}\n"
    "  Дата:     ДД.ММ.ГГГГ                  → {\"answer\": \"15.01.2024\"}\n"
    "  Интервал: N дней / N часов            → {\"answer\": \"6 дней\"}\n"
    "  Строка:   точно как в диалоге        → {\"answer\": \"одобрено\"}\n"
    "\n"
    "ЗАПРЕЩЕНО: пробелы внутри числа ('30 300'), запятая в дробях ('14,5%'), символ '₽'.\n"
    "\n"
    "ПРИМЕРЫ НЕПРАВИЛЬНЫХ ОТВЕТОВ (ЗАПРЕЩЕНО):\n"
    "Ответ: {\"answer\": \"30300 руб.\"}  ← нельзя добавлять текст до JSON\n"
    "{\"answer\": \"30300 руб.\", \"reason\": \"...\"}  ← нельзя добавлять лишние ключи\n"
    "Клиент перевёл 30300 руб.  ← нельзя отвечать текстом\n"
    "```json\n{\"answer\": \"30300 руб.\"}\n```  ← нельзя оборачивать в markdown"
)

TASK_TYPE_INSTRUCTIONS = {
    "information_extraction": {
        "instruction": (
            "Найди в диалоге ОДИН конкретный факт, о котором спрашивается, и верни его в СТРОГОМ формате.\n"
            "Формат ответа:\n"
            "  - Сумма: цифры без пробелов + ' руб.' → '30300 руб.'\n"
            "  - Ставка: X.X% с точкой → '14.5%'\n"
            "  - Дата: ДД.ММ.ГГГГ → '15.01.2024'\n"
            "  - ID / номер: точно как в диалоге → '20260115SBP001'\n"
            "  - Статус / строка: точно как в диалоге → 'одобрено'\n"
            "ЗАПРЕЩЕНО: пробелы внутри числа, '₽', 'рублей', запятая в дробях, пояснения."
        ),
        "format": '{"answer": "<значение в строгом формате>"}',
    },
    "knowledge_update": {
        "instruction": (
            "В диалоге параметр менялся несколько раз. Определи актуальное значение и верни его в СТРОГОМ формате.\n"
            "Формат ответа:\n"
            "  - Ставка: X.X% с точкой → '12.5%'\n"
            "  - Сумма: цифры без пробелов + ' руб.' → '7000 руб.'\n"
            "  - Статус: точно как в диалоге → 'одобрено'\n"
            "  - Дата: ДД.ММ.ГГГГ → '25.03.2026'\n"
            "ЗАПРЕЩЕНО: указывать старое значение, пробелы внутри числа, '₽', запятая в дробях."
        ),
        "format": '{"answer": "<актуальное значение в строгом формате>"}',
    },
    "temporal_reasoning": {
        "instruction": (
            "Вычисли временной интервал между двумя событиями из вопроса. "
            "Ответ — ТОЛЬКО число и единица, ничего лишнего.\n"
            "Формат: 'N дней' / 'N часов' / 'N минут'\n"
            "Примеры: '3 дня', '14 дней', '48 часов', '30 минут'\n"
            "ЗАПРЕЩЕНО: диапазон ('3-5 дней'), объяснение, указание дат."
        ),
        "format": '{"answer": "<N> <дней/часов/минут>"}',
    },
    "interference": {
        "instruction": (
            "В диалоге есть похожие события. Вопрос указывает на ОДНО конкретное — "
            "верни его параметр в СТРОГОМ формате.\n"
            "Формат ответа:\n"
            "  - Сумма: цифры без пробелов + ' руб.' → '5000 руб.'\n"
            "  - ID: точно как в диалоге → '20260115SBP001'\n"
            "  - Дата: ДД.ММ.ГГГГ → '15.01.2026'\n"
            "  - Строка: точно как в диалоге\n"
            "ЗАПРЕЩЕНО: перечислять несколько событий, пробелы внутри числа, '₽'."
        ),
        "format": '{"answer": "<значение нужного события в строгом формате>"}',
    },
    "composite": {
        "instruction": (
            "Проследи цепочку событий и определи итоговый результат. "
            "Верни его в СТРОГОМ формате.\n"
            "Формат ответа:\n"
            "  - Сумма: цифры без пробелов + ' руб.' → '3200000 руб.'\n"
            "  - Статус: точно как в диалоге → 'одобрено'\n"
            "  - Дата: ДД.ММ.ГГГГ → '20.02.2026'\n"
            "  - Цепочка: 'Событие1 → Событие2 → Событие3'\n"
            "ЗАПРЕЩЕНО: пересказывать цепочку целиком, пробелы внутри числа, '₽'."
        ),
        "format": '{"answer": "<итоговый факт в строгом формате>"}',
    },
}


def load_chat_text(chat_dir):
    for fname in ("chat_truncated.json", "chat.json"):
        path = os.path.join(chat_dir, fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"⚠️ Ошибка чтения {path}: {e}")
                return None
            if not isinstance(chat_data, list) or not chat_data:
                print(f"⚠️ Неожиданная структура {path}: ожидался непустой list")
                return None
            lines = []
            for batch in chat_data:
                if not isinstance(batch, list):
                    continue
                for turn in batch:
                    if not isinstance(turn, list):
                        continue
                    for msg in turn:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        lines.append(f"{role}: {content}")
            if not lines:
                print(f"⚠️ {path}: не извлечено ни одного сообщения — неожиданная структура файла")
                return None
            return "\n".join(lines)
    return None


def load_tasks(chat_dir):
    path = os.path.join(chat_dir, "tasks.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Ошибка чтения {path}: {e}")
        return None


def load_events(chat_dir):
    path = os.path.join(chat_dir, "topic.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            topic = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Ошибка чтения {path}: {e}")
        return None
    events = topic.get("events", [])
    if not events:
        return None
    lines = [f"{i + 1}. {e}" for i, e in enumerate(events)]
    return "\n".join(lines)



# Порог токенов для chat_text: оставляем запас ~70K для промпта + ответа модели
_CHAT_TOKEN_LIMIT = 200_000

_SUMMARIZE_SYSTEM = (
    "Ты — точный аналитик банковских диалогов. Твоя задача — создать подробное структурированное "
    "изложение диалога, сохранив ВСЕ факты без исключения:\n"
    "- все даты (ДД.ММ.ГГГГ)\n"
    "- все суммы (в рублях, точно)\n"
    "- все ставки и проценты\n"
    "- все имена, паспортные данные, ИНН\n"
    "- все ID транзакций, номера договоров, счетов\n"
    "- все статусы и решения банка\n"
    "- все изменения параметров (что было → что стало)\n"
    "- хронологический порядок событий\n"
    "Формат: структурированный текст по батчам/темам. Никаких сокращений фактических данных."
)

_SUMMARIZE_USER = (
    "Создай подробное изложение следующего банковского диалога, сохранив все числовые данные, "
    "даты, имена и факты:\n\n{chat_text}"
)


def summarize_chat(chat_text: str, judge_adapter) -> str:
    # суммари первой половины + дословно вторая: сохраняет точный контекст недавних событий
    from utils import get_token_number
    total_tokens = get_token_number(chat_text)
    print(f"  📝 Диалог {total_tokens:,} токенов — суммаризирую первую половину...")

    lines = chat_text.splitlines()
    mid = len(lines) // 2
    first_half = "\n".join(lines[:mid])
    second_half = "\n".join(lines[mid:])

    try:
        summary = invoke_with_retries(judge_adapter, [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {"role": "user",   "content": _SUMMARIZE_USER.format(chat_text=first_half)},
        ])
        if summary and len(summary) > 100:
            result = f"[СУММАРИ НАЧАЛА ДИАЛОГА]\n{summary}\n\n[ПРОДОЛЖЕНИЕ ДИАЛОГА (дословно)]\n{second_half}"
            print(f"  ✅ Итого: {get_token_number(result):,} токенов (было {total_tokens:,})")
            return result
    except Exception as e:
        print(f"  ⚠️ Ошибка суммаризации: {e}")

    print(f"  ⚠️ Суммаризация не удалась — используем вторую половину диалога")
    return second_half


def evaluate_single_task(model_adapter, chat_text, task, judge_adapter=None, events_text=None):
    # исключаем thought/reasoning_path — цепочка рассуждений генератора не должна попадать в промпт
    task = {k: v for k, v in task.items() if k not in ("thought", "reasoning_path")}

    question = task.get("question", "")
    gold_answer = str(task.get("answer", ""))
    task_type = task.get("capability", "unknown")

    type_info = TASK_TYPE_INSTRUCTIONS.get(task_type, {
        "instruction": "Ответь на вопрос по диалогу.",
        "format": '{"answer": "<ответ>"}',
    })

    gold_answer = re.sub(r'(?<=\d) (?=\d)', '', gold_answer)

    events_section = f"КЛЮЧЕВЫЕ СОБЫТИЯ:\n{events_text}\n\n" if events_text else ""
    open_user_message = (
        f"ИСТОРИЯ ДИАЛОГА:\n{chat_text}\n\n"
        f"{events_section}"
        f"ЗАДАНИЕ: {type_info['instruction']}\n"
        f"ВОПРОС: {question}\n\n"
        f"Формат ответа (строго JSON):\n{type_info['format']}"
    )
    open_response = invoke_with_retries(model_adapter, [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user",   "content": open_user_message},
    ])
    predicted_answer = extract_answer_from_json(open_response)
    predicted_answer = re.sub(r'(?<=\d) (?=\d)', '', predicted_answer)
    metrics = compute_all_metrics(predicted_answer, gold_answer)

    if judge_adapter and should_call_judge(metrics):
        try:
            judge_score = run_llm_judge(judge_adapter, question, gold_answer, predicted_answer, task_type)
            metrics["llm_judge"] = judge_score
        except Exception as e:
            print(f"  ⚠️ LLM Judge ошибка: {e}")
            metrics["llm_judge"] = -1.0

    metrics["final_accuracy"] = final_accuracy(metrics)

    return {
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "raw_response": open_response,
        "task_type": task_type,
        "difficulty": task.get("difficulty", "unknown"),
        "metrics": metrics,
    }


def evaluate_chat(model_adapter, chat_dir, chat_id, judge_adapter=None):
    chat_text = load_chat_text(chat_dir)
    tasks_data = load_tasks(chat_dir)

    if chat_text is None or tasks_data is None:
        return []

    from utils import get_token_number
    if get_token_number(chat_text) > _CHAT_TOKEN_LIMIT:
        if judge_adapter is not None:
            chat_text = summarize_chat(chat_text, judge_adapter)
        else:
            print(f"  ⚠️ {chat_id}: диалог превышает {_CHAT_TOKEN_LIMIT:,} токенов, judge недоступен — отправляем как есть")

    events_text = load_events(chat_dir)

    results = []
    for task_type, task_list in tasks_data.items():
        if not isinstance(task_list, list):
            continue
        for task in task_list:
            if not isinstance(task, dict):
                print(f"⚠️ Пропуск некорректной задачи в {task_type}: ожидался dict, получен {type(task).__name__}")
                continue
            task_with_type = dict(task)
            if "capability" not in task_with_type:
                task_with_type["capability"] = task_type
            result = evaluate_single_task(
                model_adapter, chat_text, task_with_type,
                judge_adapter=judge_adapter,
                events_text=events_text,
            )
            result["chat_id"] = chat_id
            results.append(result)

    return results


def aggregate_results(all_results):
    report = {
        "per_model": {},
        "per_model_difficulty": {},
        "per_model_task_type": {},
        "per_model_task_type_difficulty": {},
    }

    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    for model, results in by_model.items():
        report["per_model"][model] = avg_metrics(results)

        by_diff_all = defaultdict(list)
        for r in results:
            by_diff_all[r["difficulty"]].append(r)
        report["per_model_difficulty"][model] = {}
        for diff in ["easy", "medium", "hard"]:
            if diff in by_diff_all:
                report["per_model_difficulty"][model][diff] = avg_metrics(by_diff_all[diff])

        by_type = defaultdict(list)
        for r in results:
            by_type[r["task_type"]].append(r)

        report["per_model_task_type"][model] = {}
        for t_type, t_results in by_type.items():
            report["per_model_task_type"][model][t_type] = avg_metrics(t_results)

        report["per_model_task_type_difficulty"][model] = {}
        for t_type, t_results in by_type.items():
            by_diff = defaultdict(list)
            for r in t_results:
                by_diff[r["difficulty"]].append(r)

            report["per_model_task_type_difficulty"][model][t_type] = {}
            for diff, d_results in by_diff.items():
                report["per_model_task_type_difficulty"][model][t_type][diff] = avg_metrics(d_results)

    return report


def print_report(report):
    print("\n" + "=" * 100)
    print("EVALUATION REPORT")
    print("=" * 100)

    def _s(m, key):
        v = m.get(key, -1.0)
        return f"{v:.4f}" if v >= 0 else "N/A"

    print(f"\n{'Модель':<30} {'N':>5} {'Acc':>8} {'EM':>8} {'F1':>8} {'BERT':>8} {'Judge':>8}")
    print("-" * 82)
    for model, m in report["per_model"].items():
        print(f"{model:<30} {m['count']:>5} {_s(m,'final_accuracy'):>8} {_s(m,'exact_match'):>8} "
              f"{_s(m,'token_f1'):>8} {_s(m,'bert_score'):>8} {_s(m,'llm_judge'):>8}")

    print(f"\n{'Модель':<30} {'Сложность':<10} {'N':>5} {'Acc':>8} {'EM':>8} {'F1':>8} {'BERT':>8} {'Judge':>8}")
    print("-" * 92)
    for model, diffs in report.get("per_model_difficulty", {}).items():
        for diff in ["easy", "medium", "hard"]:
            if diff in diffs:
                m = diffs[diff]
                print(f"{model:<30} {diff:<10} {m['count']:>5} {_s(m,'final_accuracy'):>8} {_s(m,'exact_match'):>8} "
                      f"{_s(m,'token_f1'):>8} {_s(m,'bert_score'):>8} {_s(m,'llm_judge'):>8}")

    print(f"\n{'Модель':<25} {'Тип':<25} {'N':>5} {'Acc':>8} {'EM':>8} {'F1':>8} {'Judge':>8}")
    print("-" * 95)
    for model, types in report["per_model_task_type"].items():
        for t_type, m in sorted(types.items()):
            print(f"{model:<25} {t_type:<25} {m['count']:>5} {_s(m,'final_accuracy'):>8} {_s(m,'exact_match'):>8} "
                  f"{_s(m,'token_f1'):>8} {_s(m,'llm_judge'):>8}")

    print(f"\n{'Модель':<20} {'Тип':<22} {'Сложность':<10} {'N':>5} {'Acc':>8} {'EM':>8} {'F1':>8} {'Judge':>8}")
    print("-" * 105)
    for model, types in report["per_model_task_type_difficulty"].items():
        for t_type, diffs in sorted(types.items()):
            for diff in ["easy", "medium", "hard"]:
                if diff in diffs:
                    m = diffs[diff]
                    print(f"{model:<20} {t_type:<22} {diff:<10} {m['count']:>5} {_s(m,'final_accuracy'):>8} "
                          f"{_s(m,'exact_match'):>8} {_s(m,'token_f1'):>8} {_s(m,'llm_judge'):>8}")

    print("=" * 100)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def run_evaluation(chats_dir, model_names, max_threads=2, judge_model="google/gemini-3.1-flash-lite-preview",
                   overwrite=False):
    chat_entries = get_eval_chat_dirs(chats_dir)

    # фолбэк: если нет вложенной структуры {task_type}/{seq_id}/ — ищем числовые директории напрямую
    if not chat_entries and os.path.isdir(chats_dir):
        flat = sorted(
            [d for d in os.listdir(chats_dir) if d.isdigit() and os.path.isdir(os.path.join(chats_dir, d))],
            key=lambda x: int(x)
        )
        chat_entries = [(d, os.path.join(chats_dir, d)) for d in flat]

    if not chat_entries:
        print("❌ Не найдено чатов для оценки.")
        return

    if "all" in model_names:
        model_names = get_available_models()

    if judge_model in model_names:
        warnings.warn(
            f"Judge-модель '{judge_model}' входит в список оцениваемых моделей. "
            f"Это создаёт self-evaluation bias — модель оценивает сама себя. "
            f"Используйте другую judge-модель или исключите её из --models.",
            UserWarning, stacklevel=2,
        )
        print(f"⚠️  WARNING: Judge-модель '{judge_model}' совпадает с одной из оцениваемых. "
              f"Результаты LLM-judge для этой модели будут смещены.")

    judge_adapter = None
    try:
        judge_adapter = build_model(judge_model)
        print(f"✅ Judge-модель: {judge_model}")
    except Exception:
        print(f"⚠️ Judge-модель {judge_model} недоступна, LLM-as-judge отключен")

    results_dir = os.path.join(chats_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for model_name in model_names:
        print(f"\n--- Оценка моделью: {model_name} ---")

        # идемпотентность: пропускаем модель если результаты уже есть
        model_results_path = os.path.join(results_dir, f"results_{_safe_model_name(model_name)}.json")
        if not overwrite and os.path.exists(model_results_path):
            print(f"  ⏭️  Результаты уже существуют ({model_results_path}). Пропуск. "
                  f"Используйте --overwrite для принудительного перезапуска.")
            try:
                with open(model_results_path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
            except Exception as e:
                print(f"  ⚠️ Не удалось загрузить существующие результаты: {e}")
            continue

        try:
            model_adapter = build_model(model_name)
        except Exception as e:
            print(f"⚠️ Пропуск модели {model_name}: {e}")
            continue

        model_results = []
        failed_chats = []

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_chat = {}
            for chat_id, chat_dir in chat_entries:
                future = executor.submit(evaluate_chat, model_adapter, chat_dir, chat_id, judge_adapter=judge_adapter)
                future_to_chat[future] = chat_id

            for future in tqdm(as_completed(future_to_chat), total=len(future_to_chat), desc=f"Eval {model_name}"):
                chat_id = future_to_chat[future]
                try:
                    chat_results = future.result()
                    for r in chat_results:
                        r["model"] = model_name
                    model_results.extend(chat_results)
                except Exception as exc:
                    print(f"  ❌ Ошибка в чате {chat_id}: {exc}")
                    failed_chats.append({"chat_id": chat_id, "error": str(exc)})

        if failed_chats:
            print(f"  ⚠️ Пропущено чатов из-за ошибок: {len(failed_chats)} / {len(chat_entries)}")

        with open(model_results_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"  📁 Результаты сохранены: {model_results_path} "
              f"({len(model_results)} задач, {len(failed_chats)} чатов с ошибками)")

        all_results.extend(model_results)
        if failed_chats:
            failed_path = os.path.join(results_dir, f"failed_chats_{_safe_model_name(model_name)}.json")
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_chats, f, indent=2, ensure_ascii=False)

    all_results_path = os.path.join(results_dir, "all_results.json")
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    report = aggregate_results(all_results)

    report_path = os.path.join(results_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Отчёт сохранён: {report_path}")

    print_report(report)

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="BEAM Evaluation Pipeline")
    parser.add_argument("--chats_dir", type=str, default="data/results/dialogue/easy",
                        help="Директория с чатами (например: data/results/dialogue/easy)")
    parser.add_argument("--models", nargs="+", default=["google/gemini-2.5-flash"],
                        help="Список моделей для оценки (или 'all')")
    parser.add_argument("--threads", type=int, default=2, help="Потоки для параллельной обработки чатов")
    parser.add_argument("--judge_model", type=str, default="google/gemini-3.1-flash-lite-preview",
                        help="Модель-судья для LLM-as-judge (по умолчанию google/gemini-3.1-flash-lite-preview)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Перезаписать существующие результаты (по умолчанию: пропускать уже оценённые модели)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        chats_dir=args.chats_dir,
        model_names=args.models,
        max_threads=args.threads,
        judge_model=args.judge_model,
        overwrite=args.overwrite,
    )
