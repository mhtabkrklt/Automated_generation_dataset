import json
import os
import argparse
import time
import pickle
import threading
import tiktoken
from collections import defaultdict
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from adjust_chats_length import run_truncation_pipeline
from generate_topics import (
    generate_topics as _generate_topics_fn,
    CATEGORY_CATALOG as _CATEGORY_CATALOG,
)

from main import (
    generate_plans,
    user_messages_generation,
    answer_generation,
    generate_probing_questions
)
from utils import get_all_chat_dirs

TASK_TYPES = [
    "information_extraction",
    "knowledge_update",
    "temporal_reasoning",
    "interference",
    "composite",
]

DIFFICULTY_CONFIG = {
    "easy": {
        "token_range": (20_000, 25_000),
        "token_limit": 25_000,
        "num_events": 10,
        "num_batches": 3,
        "num_bullets": 10,
        "noise_level": 0.0,
        "distractor_ratio": 0.0,
        "decoy_ratio": 0.0,
        "no_truncate": False,
        "min_ratio": 0.80,
        "family_interference": True,
        "family_level": "easy",
        "name_collision_prob": 0.0,
    },
    "medium": {
        "token_range": (50_000, 60_000),
        "token_limit": 60_000,
        "num_events": 20,
        "num_batches": 5,
        "num_bullets": 15,
        "noise_level": 0.20,
        "distractor_ratio": 0.20,
        "decoy_ratio": 0.15,
        "no_truncate": False,
        "min_ratio": 0.83,
        "family_interference": True,
        "family_level": "medium",
        "name_collision_prob": 0.2,
    },
    "hard": {
        "token_range": (90_000, 100_000),
        "token_limit": 100_000,
        "num_events": 30,
        "num_batches": 8,
        "num_bullets": 15,
        "noise_level": 0.40,
        "distractor_ratio": 0.40,
        "decoy_ratio": 0.35,
        "no_truncate": True,
        "min_ratio": 0.60,
        "family_interference": True,
        "family_level": "hard",
        "name_collision_prob": 0.4,
    },
}

CONFIG = {
    "threads_plans": 3,
    "threads_main": 3,
}

class Metrics:
    def __init__(self):
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = tiktoken.get_encoding("gpt2")

        self.input_tokens = 0
        self.output_tokens = 0
        self.processed_files = 0
        self._lock = threading.Lock()

    def _count(self, text):
        if not text: return 0
        content = str(text) if not isinstance(text, str) else text
        return len(self.encoding.encode(content))

    def update(self, input_data=None, output_data=None):
        in_count = self._count(input_data)
        out_count = self._count(output_data)
        with self._lock:
            self.input_tokens += in_count
            self.output_tokens += out_count

    def mark_success(self):
        with self._lock:
            self.processed_files += 1

    def print_summary(self):
        from llm import get_token_usage
        usage = get_token_usage()
        total = self.input_tokens + self.output_tokens
        cost = (self.input_tokens / 1e6 * 0.15) + (self.output_tokens / 1e6 * 0.60)
        print(f"\n📊 --- TOKEN REPORT ---")
        print(f"📥 Input (Context):  {self.input_tokens:,}".replace(",", " "))
        print(f"📤 Output (Gen):     {self.output_tokens:,}".replace(",", " "))
        print(f"∑  Total Tokens:     {total:,}".replace(",", " "))
        print(f"💵 Примерная стоимость (ставки gpt-4o-mini, только для справки): ~${cost:.4f}")
        if usage.get("retry_calls", 0) > 0:
            print(f"\n⚠️  --- RETRY СТАТИСТИКА ---")
            print(f"🔁 Неудачных попыток:    {usage['retry_calls']}")
            print(f"🚫 Rate-limit (429):     {usage['rate_limit_hits']}  (токены не списываются)")
            other_retries = usage["retry_calls"] - usage["rate_limit_hits"]
            if other_retries > 0:
                print(f"❌ Прочие ошибки:        {other_retries}  (оценка входных токенов: {usage['retry_input']:,})".replace(",", " "))
        print("-" * 30)

metrics = Metrics()

def run_plans_stage(chats_directory: str, difficulty_config: dict, topics_file: Optional[str] = None, max_threads=CONFIG["threads_plans"]):
    print(f"\n--- [STAGE 1] PLAN GENERATION | TOKEN LIMIT: {difficulty_config['token_limit']} ---")

    noise_level = difficulty_config["noise_level"]
    noise_enabled = noise_level > 0
    distractor_ratio = difficulty_config["distractor_ratio"]
    decoy_ratio = difficulty_config["decoy_ratio"]

    noise_settings = {
        "enabled": noise_enabled,
        "level": noise_level,
        "description": "противоречия в суммах, смена цели ипотеки, неверные даты документов",
        "distractor_enabled": distractor_ratio > 0,
        "distractor_ratio": distractor_ratio,
        "decoy_enabled": decoy_ratio > 0,
        "decoy_ratio": decoy_ratio,
        "family_interference": difficulty_config.get("family_interference", False),
        "family_level": difficulty_config.get("family_level", "easy"),
        "name_collision_prob": difficulty_config.get("name_collision_prob", 0.0),
    }

    if topics_file and os.path.exists(topics_file):
        with open(topics_file, "r", encoding="utf-8") as f:
            topics_data = json.load(f)

        groups = defaultdict(list)
        for topic_obj in topics_data:
            tt = topic_obj.get("task_type", "information_extraction")
            groups[tt].append(topic_obj)

        for task_type in TASK_TYPES:
            type_dir = os.path.join(chats_directory, task_type)
            existing_ids = sorted([int(d) for d in os.listdir(type_dir) if d.isdigit()]) \
                if os.path.isdir(type_dir) else []
            next_seq = max(existing_ids) + 1 if existing_ids else 1

            for topic_obj in groups.get(task_type, []):
                found = False
                for eid in existing_ids:
                    edir = os.path.join(chats_directory, task_type, str(eid))
                    tp = os.path.join(edir, "topic.json")
                    if os.path.exists(tp):
                        try:
                            with open(tp, 'r', encoding='utf-8') as f:
                                existing_topic = json.load(f)
                            if existing_topic.get("title") == topic_obj.get("title"):
                                found = True
                                break
                        except Exception:
                            pass
                if not found:
                    while True:
                        chat_dir = os.path.join(chats_directory, task_type, str(next_seq))
                        try:
                            os.makedirs(chat_dir, exist_ok=False)
                            break
                        except FileExistsError:
                            next_seq += 1
                    with open(os.path.join(chat_dir, "topic.json"), "w", encoding="utf-8") as f:
                        json.dump(topic_obj, f, indent=4, ensure_ascii=False)
                    next_seq += 1
    else:
        print(f"  ℹ️  topics_file не указан — используем существующие topic.json из директорий")

    pending = []
    for chat_dir in get_all_chat_dirs(chats_directory):
        topic_path = os.path.join(chat_dir, "topic.json")
        plan_path = os.path.join(chat_dir, "plan.pickle")
        if os.path.exists(topic_path) and not os.path.exists(plan_path):
            try:
                with open(topic_path, 'r', encoding='utf-8') as f:
                    topic_obj = json.load(f)
                pending.append((chat_dir, topic_obj))
            except Exception as e:
                print(f"  ⚠️ Не удалось прочитать {topic_path}: {e}")

    print(f"  Директорий с topic.json без plan.pickle: {len(pending)}")

    def process_topic_plan(args):
        chat_dir, topic_obj = args
        save_path_base = os.path.join(chat_dir, "plan")
        try:
            plans = generate_plans(
                topic=topic_obj.get("topic", "Банкинг"),
                theme=topic_obj.get("theme", "Финансы"),
                num_batches=difficulty_config["num_batches"],
                num_bullets=difficulty_config["num_bullets"],
                save_address=save_path_base,
                noise_settings=noise_settings,
                events=topic_obj.get("events"),
                task_type=topic_obj.get("task_type"),
                client_name=topic_obj.get("client_name"),
            )
            metrics.update(input_data=str(topic_obj), output_data=plans)
            metrics.mark_success()
            return "success"
        except Exception as e:
            return f"error: {e}"

    failed_plans = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_topic_plan, a): a for a in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Plans"):
            try:
                result = future.result()
            except Exception as e:
                chat_dir = futures[future][0]
                print(f"  ⚠️ {chat_dir}: необработанное исключение — {e}")
                failed_plans.append(chat_dir)
                continue
            if result != "success":
                chat_dir = futures[future][0]
                print(f"  ⚠️ {chat_dir}: {result}")
                failed_plans.append(chat_dir)
    if failed_plans:
        dirs_list = "\n".join(f"  - {d}" for d in failed_plans)
        raise RuntimeError(
            f"Стадия plans завершилась с ошибками ({len(failed_plans)} директорий):\n{dirs_list}\n"
            f"Исправьте ошибки и повторите стадию plan."
        )

def run_messages_stage(chats_directory: str, max_threads: int = 3):
    print(f"\n--- [STAGE 2] STARTING PARALLEL USER MESSAGES GENERATION ---")
    chat_dirs = get_all_chat_dirs(chats_directory)

    def process_chat(chat_dir):
        save_path = os.path.join(chat_dir, "user_messages.pickle")
        plan_path = os.path.join(chat_dir, "plan.pickle")
        topic_path = os.path.join(chat_dir, "topic.json")

        if os.path.exists(save_path):
            return "skipped"
        if not os.path.exists(topic_path):
            print(f"  ℹ️ {chat_dir}: нет topic.json, пропуск (неполная директория)")
            return "skipped"
        if not os.path.exists(plan_path):
            return "error: no plan"

        try:
            with open(plan_path, 'rb') as f:
                plans = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            return f"error: повреждён plan.pickle — {e}"

        try:
            user_messages_generation(plans=plans, save_address=save_path, max_workers=2,
                                      chat_dir=chat_dir)
        except Exception as e:
            return f"error: user_messages_generation — {e}"

        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                generated_msgs = pickle.load(f)
            metrics.update(input_data=plans, output_data=generated_msgs)
            metrics.mark_success()
            return "success"

        return "error: generation failed"

    failed_msgs = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_chat, d): d for d in chat_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Messages"):
            try:
                result = future.result()
            except Exception as e:
                chat_dir = futures[future]
                print(f"  ⚠️ {chat_dir}: необработанное исключение — {e}")
                failed_msgs.append(chat_dir)
                continue
            if result not in ("success", "skipped"):
                chat_dir = futures[future]
                print(f"  ⚠️ {chat_dir}: {result}")
                failed_msgs.append(chat_dir)
    if failed_msgs:
        raise RuntimeError(
            f"Стадия messages завершилась с {len(failed_msgs)} ошибками. "
            f"Исправьте проблемы и перезапустите --stage messages."
        )

def run_answers_stage(chats_directory: str, difficulty: str, difficulty_config: dict, max_threads: int = 2):
    print(f"\n--- [STAGE 3] ANSWER GENERATION | DIFFICULTY: {difficulty} ---")
    chat_dirs = get_all_chat_dirs(chats_directory)
    pipeline_token_range = list(difficulty_config["token_range"])

    def process_answers(chat_dir):
        output_path = os.path.join(chat_dir, "chat.pickle")
        plan_path = os.path.join(chat_dir, "plan.pickle")
        msgs_path = os.path.join(chat_dir, "user_messages.pickle")
        topic_path = os.path.join(chat_dir, "topic.json")

        if os.path.exists(output_path):
            return "skipped"
        if not os.path.exists(topic_path):
            print(f"  ℹ️ {chat_dir}: нет topic.json, пропуск (неполная директория)")
            return "skipped"
        if not os.path.exists(msgs_path):
            return "error: no msgs"

        try:
            with open(topic_path, "r", encoding="utf-8") as f:
                topic_data = json.load(f)
            topic_title = topic_data.get("topic", "Банкинг")

            with open(plan_path, 'rb') as f:
                p_data = pickle.load(f)
            with open(msgs_path, 'rb') as f:
                m_data = pickle.load(f)

            res = answer_generation(
                input_address=msgs_path,
                output_address=output_path,
                plans_address=plan_path,
                topic=topic_title,
                chat_size=difficulty,
                token_range=pipeline_token_range
            )

            metrics.update(input_data=str(p_data) + str(m_data), output_data=res)
            metrics.mark_success()
            return "success"
        except (pickle.UnpicklingError, EOFError) as e:
            return f"error: повреждён pickle — {e}"
        except Exception as e:
            return f"error: {e}"

    failed_answers = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_answers, d): d for d in chat_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Answers"):
            try:
                result = future.result()
            except Exception as e:
                chat_dir = futures[future]
                print(f"  ⚠️ {chat_dir}: необработанное исключение — {e}")
                failed_answers.append(chat_dir)
                continue
            if result not in ("success", "skipped"):
                chat_dir = futures[future]
                print(f"  ⚠️ {chat_dir}: {result}")
                failed_answers.append(chat_dir)
    if failed_answers:
        raise RuntimeError(
            f"Стадия answers завершилась с {len(failed_answers)} ошибками. "
            f"Исправьте проблемы и перезапустите --stage answers."
        )

def run_truncate_stage(chats_directory: str, difficulty_config: dict):
    token_limit = difficulty_config["token_limit"]
    no_truncate = difficulty_config.get("no_truncate", False)
    min_ratio = difficulty_config.get("min_ratio", 0.80)
    print(f"\n--- [STAGE 3.5] TRUNCATING CHATS TO {token_limit} TOKENS ---")
    for task_type in sorted(os.listdir(chats_directory)):
        type_dir = os.path.join(chats_directory, task_type)
        if os.path.isdir(type_dir):
            print(f"  📁 {task_type}:")
            run_truncation_pipeline(type_dir, token_limit, no_truncate=no_truncate, min_ratio=min_ratio)

def run_tasks_stage(chats_directory: str, difficulty: str = "easy", max_threads: int = 2):
    print(f"\n--- [STAGE 4] QA TASK GENERATION (difficulty={difficulty}) ---")
    chat_dirs = get_all_chat_dirs(chats_directory)

    def process_tasks(chat_dir):
        save_path = os.path.join(chat_dir, "tasks.json")
        if os.path.exists(save_path):
            return "skipped"

        topic_path = os.path.join(chat_dir, "topic.json")
        if not os.path.exists(topic_path):
            print(f"  ℹ️ {chat_dir}: нет topic.json, пропуск (неполная директория)")
            return "skipped"

        task_type = "information_extraction"
        with open(topic_path, 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
        task_type = topic_data.get("task_type", "information_extraction")

        truncated_chat = os.path.join(chat_dir, "chat_truncated.json")
        original_chat = os.path.join(chat_dir, "chat.json")
        chat_json_path = truncated_chat if os.path.exists(truncated_chat) else original_chat

        plan_path = (
            os.path.join(chat_dir, "plan_truncated.pickle")
            if os.path.exists(truncated_chat)
            else os.path.join(chat_dir, "plan.pickle")
        )

        if not os.path.exists(chat_json_path):
            return "skipped: no chat file yet"
        if not os.path.exists(plan_path):
            return "skipped: no plan file yet"

        try:
            with open(plan_path, 'rb') as f:
                plan_data = pickle.load(f)
            with open(chat_json_path, 'r', encoding='utf-8') as f:
                c_txt = f.read()
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            return f"error: не удалось прочитать файлы чата — {e}"

        try:
            tasks = generate_probing_questions(
                plan_address=plan_path,
                chat_address=chat_json_path,
                save_dir=chat_dir,
                difficulty=difficulty,
                task_type=task_type,
            )
        except Exception as e:
            return f"error: generate_probing_questions упала — {e}"

        total_tasks = sum(len(v) for v in tasks.values() if isinstance(v, list)) if isinstance(tasks, dict) else 0
        if total_tasks == 0:
            print(f"⚠️ WARNING: {chat_dir} — сгенерировано 0 задач!")

        metrics.update(input_data=str(plan_data) + c_txt, output_data=tasks)
        metrics.mark_success()
        return "success"

    failed_tasks = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_tasks, d): d for d in chat_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tasks"):
            res = future.result()
            if res not in ("success", "skipped") and not res.startswith("skipped:"):
                chat_dir = futures[future]
                print(f"  ⚠️ {chat_dir}: {res}")
                failed_tasks.append(chat_dir)
            elif res.startswith("skipped:"):
                chat_dir = futures[future]
                print(f"  ⏭ {chat_dir}: {res}")
    if failed_tasks:
        dirs_list = "\n".join(f"  - {d}" for d in failed_tasks)
        raise RuntimeError(
            f"Стадия tasks завершилась с {len(failed_tasks)} ошибками:\n{dirs_list}\n"
            f"Исправьте проблемы и перезапустите --stage tasks."
        )

def run_evaluate_stage(chats_directory: str, model_names: list, eval_threads: int = 2,
                       judge_model: Optional[str] = None):
    print(f"\n--- [STAGE 5] MODEL EVALUATION ---")
    from evaluate import run_evaluation
    kwargs = dict(chats_dir=chats_directory, model_names=model_names, max_threads=eval_threads)
    if judge_model:
        kwargs["judge_model"] = judge_model
    run_evaluation(**kwargs)

def run_topics_stage(chats_directory: str, output_dir: str, difficulty: str,
                     count: int, categories: Optional[list] = None) -> str:
    print(f"\n--- [STAGE 0] TOPIC GENERATION | count={count}, difficulty={difficulty} ---")
    if categories:
        print(f"  Категории: {categories}")

    topics_output = os.path.join(chats_directory, "topics_auto.json")

    if os.path.exists(topics_output):
        print(f"  ℹ️ topics_auto.json уже существует — пропуск генерации (удалите файл для перегенерации)")
        return topics_output

    topics = _generate_topics_fn(
        count=count,
        difficulty=difficulty,
        categories=categories,
        output_path=topics_output,
        results_dir=output_dir,
    )
    n = len(topics) if topics else 0
    print(f"  ✅ Сгенерировано {n} тем → {topics_output}")
    return topics_output

def parse_args():
    parser = argparse.ArgumentParser(description="BEAM Dataset Generator")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["topics", "plan", "messages", "answers", "truncate", "tasks", "evaluate", "all"],
                        help="Этап пайплайна: topics=только генерация тем, all=полный цикл")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--output_dir", type=str, default="data/results",
                        help="Базовая директория результатов (difficulty добавляется автоматически)")
    parser.add_argument("--topics", type=str, default="topics.json",
                        help="Путь к topics.json (используется если --count не указан)")
    parser.add_argument("--count", type=int, default=0,
                        help="Кол-во тем для авто-генерации (0 = использовать --topics файл, "
                             "рекомендуется кратно 5 для равномерного распределения по task_types)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=list(_CATEGORY_CATALOG.keys()),
                        help="Категории для генерации тем (по умолчанию все категории)")
    parser.add_argument("--models", nargs="+", default=["gemini-2.5-pro-preview-06-05"],
                        help="Модели для оценки")
    parser.add_argument("--eval_threads", type=int, default=2)
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Модель-судья для LLM-as-Judge (переопределяет дефолт в evaluate.py)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    difficulty_config = DIFFICULTY_CONFIG[args.difficulty]

    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables.")

    chats_dir = os.path.join(args.output_dir, "dialogue", args.difficulty)
    os.makedirs(chats_dir, exist_ok=True)

    print(f"\n=== DIFFICULTY: {args.difficulty.upper()} | TOKEN LIMIT: {difficulty_config['token_limit']} ===")
    print(f"📂 Output: {chats_dir}/{{task_type}}/{{id}}/")
    start_time = time.time()

    auto_topics_file = None
    if args.stage in ["topics", "all"]:
        if args.count > 0:
            auto_topics_file = run_topics_stage(
                chats_dir, args.output_dir, args.difficulty, args.count, args.categories
            )
        elif args.stage == "topics":
            print("ОШИБКА: --stage topics требует --count N (например: --count 5)")
            import sys; sys.exit(1)

    topics_file = auto_topics_file or (args.topics if os.path.exists(args.topics) else None)

    if args.stage in ["plan", "all"]:
        run_plans_stage(chats_dir, difficulty_config, topics_file, CONFIG["threads_plans"])

    if args.stage in ["messages", "all"]:
        run_messages_stage(chats_dir, CONFIG["threads_main"])

    if args.stage in ["answers", "all"]:
        run_answers_stage(chats_dir, args.difficulty, difficulty_config, CONFIG["threads_main"])

    if args.stage in ["truncate", "all"]:
        run_truncate_stage(chats_dir, difficulty_config)

    if args.stage in ["tasks", "all"]:
        run_tasks_stage(chats_dir, args.difficulty, CONFIG["threads_main"])

    if args.stage in ["evaluate"]:
        run_evaluate_stage(chats_dir, args.models, args.eval_threads, args.judge_model)

    total_time = time.time() - start_time
    print(f"\n=== PIPELINE FINISHED ===")
    print(f"⏱  Total Time: {total_time:.2f}s")
    print(f"📂 Processed Files: {metrics.processed_files}")
    metrics.print_summary()