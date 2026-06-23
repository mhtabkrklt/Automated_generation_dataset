import os
import re
import json
import argparse
import warnings
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from model_config import build_model, get_available_models
from metrics import (
    compute_all_metrics, run_llm_judge, should_call_judge,
    extract_answer_from_json, avg_metrics, path_accuracy, final_accuracy,
)
from graph_generator import serialize_graph_for_model
from utils import get_eval_chat_dirs, invoke_with_retries


EVAL_SYSTEM_PROMPT_GRAPH = (
    "Ты — AI-ассистент, отвечающий на вопросы по структурированному графу знаний.\n"
    "Твоя задача — ВНИМАТЕЛЬНО прочитать граф и дать точный ответ на поставленный вопрос.\n"
    "ВАЖНО: Используй только факты из графа. Навигируй по рёбрам последовательно.\n"
    "\n"
    "КРИТИЧЕСКИЕ ПРАВИЛА ОТВЕТА:\n"
    "1. Твой ответ должен содержать ТОЛЬКО JSON-объект — ничего до и ничего после.\n"
    "2. Никаких вступлений, объяснений, рассуждений, комментариев.\n"
    "3. Никаких markdown-блоков (```json ... ```).\n"
    "4. Ровно один ключ: \"answer\" со значением-строкой.\n"
    "5. Значение answer — короткое и точное, без лишних слов.\n"
    "\n"
    "СТРОГИЙ ФОРМАТ ЗНАЧЕНИЙ:\n"
    "  Сумма:    цифры без пробелов + ' руб.' → {\"answer\": \"4500000 руб.\"}\n"
    "  Ставка:   X.X% с точкой               → {\"answer\": \"12.5%\"}\n"
    "  Дата:     ДД.ММ.ГГГГ                  → {\"answer\": \"15.01.2024\"}\n"
    "  Интервал: N дней / N часов            → {\"answer\": \"5 дней\"}\n"
    "  Строка:   точно из атрибутов графа   → {\"answer\": \"одобрено\"}\n"
    "\n"
    "ЗАПРЕЩЕНО: пробелы внутри числа ('4 500 000'), запятая в дробях ('12,5%'), символ '₽'.\n"
    "\n"
    "ЕДИНСТВЕННО ДОПУСТИМЫЙ ФОРМАТ:\n"
    "{\"answer\": \"<значение>\"}\n"
)

TASK_TYPE_INSTRUCTIONS_GRAPH = {
    "information_extraction": {
        "instruction": (
            "Найди в графе конкретный факт, на который указывает вопрос. "
            "Верни его в СТРОГОМ формате: сумма → '4500000 руб.', ставка → '12.5%', "
            "дата → '15.01.2024', строка → точно из атрибутов.\n"
            "ЗАПРЕЩЕНО: пробелы внутри числа, '₽', запятая в дробях."
        ),
        "format": '{"answer": "<значение в строгом формате>"}',
    },
    "knowledge_update": {
        "instruction": (
            "Параметр мог меняться со временем. Проанализируй временну́ю информацию в атрибутах рёбер "
            "и верни актуальное значение в СТРОГОМ формате: ставка → '12.5%', "
            "сумма → '4500000 руб.', дата → '15.01.2024'.\n"
            "ЗАПРЕЩЕНО: пробелы внутри числа, '₽', запятая в дробях."
        ),
        "format": '{"answer": "<актуальное значение в строгом формате>"}',
    },
    "temporal_reasoning": {
        "instruction": (
            "Найди в атрибутах рёбер даты событий, упомянутых в вопросе. "
            "Вычисли временной интервал. Ответ — ТОЛЬКО 'N дней' / 'N часов' / 'N минут'.\n"
            "ЗАПРЕЩЕНО: диапазон ('3-5 дней'), пояснения, указание дат."
        ),
        "format": '{"answer": "<N> <дней/часов/минут>"}',
    },
    "interference": {
        "instruction": (
            "В графе есть похожие события. Вопрос указывает на одно конкретное — "
            "верни его атрибут в СТРОГОМ формате: сумма → '5000 руб.', "
            "дата → '15.01.2024', строка → точно из графа.\n"
            "ЗАПРЕЩЕНО: перечислять несколько событий, пробелы внутри числа."
        ),
        "format": '{"answer": "<значение нужного события в строгом формате>"}',
    },
    "composite": {
        "instruction": (
            "Проследи причинно-следственную цепочку и определи результат. "
            "Верни в СТРОГОМ формате: сумма → '3200000 руб.', статус → 'одобрено', "
            "цепочка → 'Событие1 → Событие2 → Событие3'.\n"
            "ЗАПРЕЩЕНО: пересказывать цепочку целиком, пробелы внутри числа."
        ),
        "format": '{"answer": "<итоговый факт в строгом формате>"}',
    },
}


def load_graph_text(chat_dir: str) -> Optional[str]:
    graph_path = os.path.join(chat_dir, "graph.json")
    if not os.path.exists(graph_path):
        return None

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Ошибка чтения {graph_path}: {e}")
        return None

    difficulty = "easy"
    task_type = ""
    topic_path = os.path.join(chat_dir, "topic.json")
    if os.path.exists(topic_path):
        try:
            with open(topic_path, "r", encoding="utf-8") as f:
                topic_data = json.load(f)
            difficulty = topic_data.get("difficulty", "easy")
            task_type = topic_data.get("task_type", "")
        except Exception:
            pass

    return serialize_graph_for_model(graph, difficulty, task_type=task_type)


def load_graph_tasks(chat_dir: str) -> Optional[dict]:
    path = os.path.join(chat_dir, "tasks_graph.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Ошибка чтения {path}: {e}")
        return None


def evaluate_single_graph_task(model_adapter, graph_text: str, task: dict,
                                judge_adapter=None) -> dict:
    task = {k: v for k, v in task.items() if k not in ("thought", "source_node_ids", "reasoning_path")}

    question = task.get("question", "")
    gold_answer = str(task.get("answer", ""))
    task_type = task.get("capability", "unknown")
    difficulty = task.get("difficulty", "easy")

    type_info = TASK_TYPE_INSTRUCTIONS_GRAPH.get(task_type, {
        "instruction": "Ответь на вопрос по графу.",
        "format": '{"answer": "<ответ>"}',
    })

    gold_answer = re.sub(r'(?<=\d) (?=\d)', '', gold_answer)

    open_user_message = (
        f"ГРАФ ЗНАНИЙ:\n{graph_text}\n\n"
        f"ЗАДАНИЕ: {type_info['instruction']}\n"
        f"ВОПРОС: {question}\n\n"
        f"Формат ответа (строго JSON):\n{type_info['format']}"
    )
    open_response = invoke_with_retries(model_adapter, [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT_GRAPH},
        {"role": "user", "content": open_user_message},
    ])
    predicted_answer = extract_answer_from_json(open_response)
    predicted_answer = re.sub(r'(?<=\d) (?=\d)', '', predicted_answer)
    metrics_result = compute_all_metrics(predicted_answer, gold_answer)

    if task_type == "composite":
        metrics_result["path_accuracy"] = round(path_accuracy(predicted_answer, gold_answer), 4)

    if judge_adapter and should_call_judge(metrics_result):
        try:
            judge_score = run_llm_judge(
                judge_adapter, question, gold_answer, predicted_answer, task_type
            )
            metrics_result["llm_judge"] = judge_score
        except Exception as e:
            print(f"  ⚠️ LLM Judge ошибка: {e}")
            metrics_result["llm_judge"] = -1.0

    metrics_result["final_accuracy"] = final_accuracy(metrics_result)

    return {
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "raw_response": open_response,
        "task_type": task_type,
        "difficulty": difficulty,
        "metrics": metrics_result,
    }


def evaluate_graph_chat(model_adapter, chat_dir: str, chat_id: str,
                        judge_adapter=None) -> list:
    graph_text = load_graph_text(chat_dir)
    tasks_data = load_graph_tasks(chat_dir)

    if graph_text is None or tasks_data is None:
        return []

    results = []
    for task_type, task_list in tasks_data.items():
        if not isinstance(task_list, list):
            continue
        for task in task_list:
            if not isinstance(task, dict):
                continue
            task_with_type = dict(task)
            if "capability" not in task_with_type:
                task_with_type["capability"] = task_type
            result = evaluate_single_graph_task(
                model_adapter, graph_text, task_with_type, judge_adapter=judge_adapter
            )
            result["chat_id"] = chat_id
            results.append(result)

    return results


def aggregate_graph_results(all_results: list) -> dict:
    report: dict[str, dict] = {
        "per_model": {},
        "per_model_difficulty": {},
        "per_model_task_type": {},
        "per_model_task_type_difficulty": {},
    }

    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    for model, results in by_model.items():
        report["per_model"][model] = avg_metrics(results, include_path=True)

        by_diff_all = defaultdict(list)
        for r in results:
            by_diff_all[r["difficulty"]].append(r)
        report["per_model_difficulty"][model] = {}
        for diff in ["easy", "medium", "hard"]:
            if diff in by_diff_all:
                report["per_model_difficulty"][model][diff] = avg_metrics(by_diff_all[diff], include_path=True)

        by_type = defaultdict(list)
        for r in results:
            by_type[r["task_type"]].append(r)

        report["per_model_task_type"][model] = {}
        for t_type, t_results in by_type.items():
            report["per_model_task_type"][model][t_type] = avg_metrics(t_results, include_path=True)

        report["per_model_task_type_difficulty"][model] = {}
        for t_type, t_results in by_type.items():
            by_diff = defaultdict(list)
            for r in t_results:
                by_diff[r["difficulty"]].append(r)
            report["per_model_task_type_difficulty"][model][t_type] = {}
            for diff, d_results in by_diff.items():
                report["per_model_task_type_difficulty"][model][t_type][diff] = avg_metrics(
                    d_results, include_path=True
                )

    return report


def print_graph_report(report: dict):
    print("\n" + "=" * 105)
    print("GRAPH EVALUATION REPORT")
    print("=" * 105)

    def _s(m, key):
        v = m.get(key, -1.0)
        return f"{v:.4f}" if v >= 0 else "N/A"

    print(f"\n{'Модель':<30} {'N':>5} {'Acc':>8} {'EM':>8} {'F1':>8} {'BERT':>8} {'PATH':>8} {'Judge':>8}")
    print("-" * 100)
    for model, m in report["per_model"].items():
        print(f"{model:<30} {m['count']:>5} {_s(m,'final_accuracy'):>8} {_s(m,'exact_match'):>8} "
              f"{_s(m,'token_f1'):>8} {_s(m,'bert_score'):>8} {_s(m,'path_accuracy'):>8} {_s(m,'llm_judge'):>8}")

    print(f"\n{'Модель':<25} {'Тип':<25} {'N':>5} {'Acc':>8} {'EM':>8} {'F1':>8} {'PATH':>8} {'Judge':>8}")
    print("-" * 100)
    for model, types in report["per_model_task_type"].items():
        for t_type, m in sorted(types.items()):
            print(f"{model:<25} {t_type:<25} {m['count']:>5} {_s(m,'final_accuracy'):>8} {_s(m,'exact_match'):>8} "
                  f"{_s(m,'token_f1'):>8} {_s(m,'path_accuracy'):>8} {_s(m,'llm_judge'):>8}")

    print("=" * 95)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def run_graph_evaluation(
    chats_dir: str,
    model_names: list,
    max_threads: int = 2,
    judge_model: str = "google/gemini-3.1-flash-lite-preview",
    overwrite: bool = False,
) -> Optional[dict]:
    chat_entries = get_eval_chat_dirs(chats_dir)

    if not chat_entries:
        print("❌ Не найдено чатов для оценки.")
        return None

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

    results_dir = os.path.join(chats_dir, "eval_results_graph")
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
                future = executor.submit(
                    evaluate_graph_chat, model_adapter, chat_dir, chat_id, judge_adapter
                )
                future_to_chat[future] = chat_id

            for future in tqdm(as_completed(future_to_chat), total=len(future_to_chat),
                               desc=f"Eval {model_name}"):
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
            print(f"  ⚠️ Пропущено чатов: {len(failed_chats)} / {len(chat_entries)}")

        with open(model_results_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"  📁 Результаты сохранены: {model_results_path} ({len(model_results)} задач)")

        all_results.extend(model_results)
        if failed_chats:
            failed_path = os.path.join(results_dir, f"failed_chats_{_safe_model_name(model_name)}.json")
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_chats, f, indent=2, ensure_ascii=False)

    all_results_path = os.path.join(results_dir, "all_results.json")
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    report = aggregate_graph_results(all_results)
    report_path = os.path.join(results_dir, "evaluation_report_graph.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Отчёт сохранён: {report_path}")

    print_graph_report(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="BEAM Graph Evaluation Pipeline")
    parser.add_argument("--chats_dir", type=str, default="data/results/graph/easy",
                        help="Директория графового пайплайна (например, data/results/graph/easy)")
    parser.add_argument("--models", nargs="+", default=["google/gemini-2.5-flash"],
                        help="Список моделей для оценки (или 'all')")
    parser.add_argument("--threads", type=int, default=2, help="Потоки для параллельной обработки")
    parser.add_argument("--judge_model", type=str, default="google/gemini-3.1-flash-lite-preview",
                        help="Модель-судья для LLM-as-judge (по умолчанию google/gemini-3.1-flash-lite-preview)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Перезаписать существующие результаты (по умолчанию: пропускать уже оценённые модели)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_graph_evaluation(
        chats_dir=args.chats_dir,
        model_names=args.models,
        max_threads=args.threads,
        judge_model=args.judge_model,
        overwrite=args.overwrite,
    )