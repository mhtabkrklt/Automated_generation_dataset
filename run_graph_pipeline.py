import os
import json
import time
import argparse
import shutil
import pickle
from collections import defaultdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from generate_topics import (
    generate_topics as _generate_topics_fn,
    CATEGORY_CATALOG as _CATEGORY_CATALOG,
)
from run_pipeline import (
    run_plans_stage as _run_plans_stage,
    DIFFICULTY_CONFIG,
    CONFIG as DIALOGUE_CONFIG,
)
from utils import get_all_chat_dirs
from graph_generator import generate_graph
from graph_tasks import generate_graph_probing_task
from evaluate_graph import run_graph_evaluation
from adjust_graphs import adjust_graph

TASK_TYPES = [
    "information_extraction",
    "knowledge_update",
    "temporal_reasoning",
    "interference",
    "composite",
]

CONFIG = {
    "threads_plans": DIALOGUE_CONFIG["threads_plans"],
    "threads_graph": 5,
    "threads_tasks": 5,
}

def run_topics_stage(
    chats_directory: str,
    difficulty: str,
    count: int,
    categories: Optional[list] = None,
    reuse_topics: Optional[str] = None,
    only_task_types: Optional[list] = None,
) -> None:
    print(f"\n--- [STAGE 0] TOPIC SETUP | difficulty={difficulty} ---")

    if reuse_topics:
        _reuse_from_dialogue(chats_directory, reuse_topics)
        return

    if count > 0:
        _generate_new_topics(chats_directory, difficulty, count, categories, only_task_types)
    else:
        print("  ℹ️  --count не указан и --reuse_topics не передан. Используем существующие topic.json.")

def _reuse_from_dialogue(graph_dir: str, dialogue_dir: str):
    print(f"  📋 Переиспользуем из: {dialogue_dir}")
    counts: dict[str, int] = defaultdict(int)
    files_to_copy = ["topic.json", "plan.pickle", "plan.txt",
                     "plan_narrative.txt", "user_profile.json"]

    for task_type in TASK_TYPES:
        src_type_dir = os.path.join(dialogue_dir, task_type)
        if not os.path.isdir(src_type_dir):
            continue
        for seq_id in sorted(
            [d for d in os.listdir(src_type_dir) if d.isdigit()],
            key=lambda x: int(x)
        ):
            src_dir = os.path.join(src_type_dir, seq_id)
            dst_dir = os.path.join(graph_dir, task_type, seq_id)
            os.makedirs(dst_dir, exist_ok=True)

            for fname in files_to_copy:
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dst_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    counts[fname] += 1

    for fname, n in counts.items():
        print(f"    {fname}: {n} файлов скопировано")

def _generate_new_topics(chats_directory: str, difficulty: str,
                         count: int, categories: Optional[list] = None,
                         only_task_types: Optional[list] = None):
    print(f"  🆕 Генерация {count} новых тем (difficulty={difficulty})")
    topics_output = os.path.join(chats_directory, "topics_auto.json")

    topics = _generate_topics_fn(
        count=count,
        difficulty=difficulty,
        categories=categories,
        output_path=topics_output,
        results_dir=None,
        chat_dir_base=chats_directory,
        only_task_types=only_task_types,
    )

    if not topics:
        print("  ⚠️ Темы не сгенерированы")
        return

    print(f"  ✅ Сохранено {len(topics)} тем → {topics_output}")

def run_plan_stage(chats_directory: str, difficulty: str,
                   topics_file: Optional[str] = None,
                   max_threads: int = CONFIG["threads_plans"]):
    print(f"\n--- [STAGE 1] PLAN GENERATION | difficulty={difficulty} ---")
    diff_cfg = DIFFICULTY_CONFIG[difficulty]
    graph_diff_cfg = dict(diff_cfg)
    graph_diff_cfg["noise_level"] = 0.0
    graph_diff_cfg["distractor_ratio"] = 0.0
    graph_diff_cfg["decoy_ratio"] = 0.0

    _run_plans_stage(chats_directory, graph_diff_cfg, topics_file, max_threads)

def run_graph_gen_stage(chats_directory: str, difficulty: str,
                        max_threads: int = CONFIG["threads_graph"]):
    print(f"\n--- [STAGE 2] GRAPH GENERATION | difficulty={difficulty} ---")
    chat_dirs = get_all_chat_dirs(chats_directory)

    pending = []
    for chat_dir in chat_dirs:
        plan_path = os.path.join(chat_dir, "plan.pickle")
        graph_path = os.path.join(chat_dir, "graph.json")
        if os.path.exists(plan_path) and not os.path.exists(graph_path):
            pending.append(chat_dir)

    skipped = len(chat_dirs) - len(pending)
    print(f"  Чатов с plan.pickle: {len(chat_dirs)} | Пропущено: {skipped} | Обрабатываем: {len(pending)}")

    def process_graph(chat_dir):
        topic_path = os.path.join(chat_dir, "topic.json")
        plan_path = os.path.join(chat_dir, "plan.pickle")
        profile_path = os.path.join(chat_dir, "user_profile.json")

        topic = "Банковский сценарий"
        task_type = None
        events = []
        if os.path.exists(topic_path):
            try:
                with open(topic_path, "r", encoding="utf-8") as f:
                    topic_data = json.load(f)
                topic = topic_data.get("topic", topic_data.get("title", topic))
                task_type = topic_data.get("task_type")
                events = topic_data.get("events", [])
            except Exception as e:
                print(f"  ⚠️ {chat_dir}: ошибка чтения topic.json: {e}")
                return "error: не удалось прочитать topic.json"

        if not task_type:
            print(f"  ⚠️ {chat_dir}: поле task_type отсутствует в topic.json")
            return "error: task_type не задан в topic.json"

        try:
            with open(plan_path, "rb") as f:
                plan_batches = pickle.load(f)
        except Exception as e:
            return f"error: не удалось загрузить plan.pickle — {e}"

        profile_data = None
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
            except Exception as e:
                print(f"  ⚠️ {chat_dir}: ошибка чтения user_profile.json: {e}, граф без профиля")

        try:
            generate_graph(
                events=events,
                topic=topic,
                difficulty=difficulty,
                task_type=task_type,
                save_dir=chat_dir,
                profile_data=profile_data,
                plan_batches=plan_batches,
            )
            return "success"
        except Exception as e:
            partial = os.path.join(chat_dir, "graph.json")
            if os.path.exists(partial):
                os.remove(partial)
            return f"error: {e}"

    failed = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_graph, d): d for d in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Graph Gen"):
            result = future.result()
            if result != "success":
                chat_dir = futures[future]
                print(f"  ⚠️ {chat_dir}: {result}")
                failed.append(chat_dir)

    if failed:
        print(f"\n⚠️ Ошибки генерации графов: {len(failed)} чатов")

def run_graph_adjust_stage(chats_directory: str, difficulty: str,
                           max_threads: int = CONFIG["threads_graph"]):
    print(f"\n--- [STAGE 2.5] GRAPH ADJUSTMENT | difficulty={difficulty} ---")
    chat_dirs = get_all_chat_dirs(chats_directory)

    pending = []
    for d in chat_dirs:
        has_graph = os.path.exists(os.path.join(d, "graph.json"))
        already_checked = os.path.exists(os.path.join(d, "graph_check.json"))
        if has_graph and not already_checked:
            pending.append(d)

    skipped = len(chat_dirs) - len(pending)
    print(f"  Чатов с graph.json: {len(chat_dirs)} | Уже проверено: {skipped} | Обрабатываем: {len(pending)}")

    counts = {"OK": 0, "trim": 0, "pad": 0, "regen": 0, "skip": 0}

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(adjust_graph, d, difficulty): d for d in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Graph Adjust"):
            result = future.result()
            action = result.get("action", "skip")
            chat_dir = futures[future]
            counts[action] = counts.get(action, 0) + 1
            sb = result.get("stats_before", {})
            sa = result.get("stats_after", sb)
            if action == "regen":
                print(
                    f"  🔄 REGEN  {chat_dir}: {result.get('reason')} "
                    f"(было: {sb.get('nodes')} узлов, {sb.get('edges')} рёбер)"
                )
            elif action == "pad":
                print(
                    f"  ➕ PAD    {chat_dir}: {result.get('reason')} "
                    f"→ {sa.get('nodes')} узлов, {sa.get('edges')} рёбер"
                )
            elif action == "trim":
                print(
                    f"  ✂️  TRIM   {chat_dir}: {result.get('reason')} "
                    f"→ {sa.get('nodes')} узлов, {sa.get('edges')} рёбер, {sa.get('tokens_test_input')} токенов"
                )

    print(
        f"\n  Итого: ✅ OK={counts['OK']} | ➕ pad={counts['pad']} | "
        f"✂️  trim={counts['trim']} | 🔄 regen={counts['regen']}"
    )
    if counts["regen"] > 0:
        print(f"  ⚠️  {counts['regen']} граф(а) помечены для перегенерации — запустите --stage graph_gen повторно.")

def run_graph_tasks_stage(chats_directory: str, difficulty: str,
                          max_threads: int = CONFIG["threads_tasks"]):
    print(f"\n--- [STAGE 3] GRAPH TASK GENERATION | difficulty={difficulty} ---")
    chat_dirs = get_all_chat_dirs(chats_directory)

    def _graph_path(chat_dir: str) -> str | None:
        p = os.path.join(chat_dir, "graph.json")
        return p if os.path.exists(p) else None

    def _tasks_done(chat_dir: str) -> bool:
        p = os.path.join(chat_dir, "tasks_graph.json")
        if not os.path.exists(p):
            return False
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return not data.get("failed", False)
        except Exception:
            return False

    pending = []
    for chat_dir in chat_dirs:
        if _graph_path(chat_dir) and not _tasks_done(chat_dir):
            pending.append(chat_dir)

    skipped = len(chat_dirs) - len(pending)
    print(f"  Чатов с графом: {len(chat_dirs)} | Пропущено: {skipped} | Обрабатываем: {len(pending)}")

    def process_tasks(chat_dir):
        topic_path = os.path.join(chat_dir, "topic.json")
        graph_path = _graph_path(chat_dir)
        task_type = None
        if os.path.exists(topic_path):
            try:
                with open(topic_path, "r", encoding="utf-8") as f:
                    task_type = json.load(f).get("task_type")
            except Exception as e:
                print(f"  ⚠️ {chat_dir}: ошибка чтения topic.json: {e}")
        if not task_type:
            print(f"  ⚠️ {chat_dir}: task_type не найден в topic.json — пропускаем")
            return "error: task_type не задан в topic.json"
        try:
            tasks = generate_graph_probing_task(
                graph_path=graph_path,
                topic_path=topic_path,
                save_dir=chat_dir,
                difficulty=difficulty,
                task_type=task_type,
            )
            total = sum(len(v) for v in tasks.values() if isinstance(v, list))
            return "warn: 0 задач" if total == 0 else "success"
        except Exception as e:
            return f"error: {e}"

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_tasks, d): d for d in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Graph Tasks"):
            result = future.result()
            if result != "success":
                print(f"  ⚠️ {futures[future]}: {result}")

def run_evaluate_stage(chats_directory: str, model_names: list, eval_threads: int = 2):
    print(f"\n--- [STAGE 4] GRAPH MODEL EVALUATION ---")
    run_graph_evaluation(
        chats_dir=chats_directory,
        model_names=model_names,
        max_threads=eval_threads,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="BEAM Graph Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["topics", "plan", "graph_gen", "graph_adjust", "graph_tasks", "evaluate", "all"],
        help=(
            "Стадия пайплайна:\n"
            "  topics       — Stage 0: генерация / копирование topic.json\n"
            "  plan         — Stage 1: генерация plan.pickle + user_profile.json\n"
            "  graph_gen    — Stage 2: генерация graph.json из плана\n"
            "  graph_adjust — Stage 2.5: проверка и корректировка графов (аналог truncate)\n"
            "  graph_tasks  — Stage 3: генерация tasks_graph.json\n"
            "  evaluate     — Stage 4: оценка моделей\n"
            "  all          — полный цикл (0→1→2→2.5→3)"
        )
    )
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--output_dir", type=str, default="data/results",
                        help="Базовая директория результатов")
    parser.add_argument("--count", type=int, default=0,
                        help="Кол-во тем для авто-генерации (кратно 5)")
    parser.add_argument("--topics", type=str, default="topics.json",
                        help="Путь к topics.json (если --count не указан)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=list(_CATEGORY_CATALOG.keys()),
                        help="Категории для генерации тем")
    parser.add_argument("--only_task_types", nargs="+", default=None,
                        choices=["information_extraction", "knowledge_update", "temporal_reasoning",
                                 "interference", "composite"],
                        help="Генерировать темы только для указанных типов задач")
    parser.add_argument("--reuse_topics", type=str, default=None,
                        help=(
                            "Путь к диалоговому пайплайну для копирования артефактов:\n"
                            "  topic.json, plan.pickle, user_profile.json\n"
                            "  Пример: data/results/dialogue/easy"
                        ))
    parser.add_argument("--models", nargs="+", default=["gemini-2.5-pro-preview-06-05"],
                        help="Модели для оценки")
    parser.add_argument("--eval_threads", type=int, default=2)
    parser.add_argument("--plan_threads", type=int, default=CONFIG["threads_plans"])
    parser.add_argument("--graph_threads", type=int, default=CONFIG["threads_graph"])
    parser.add_argument("--tasks_threads", type=int, default=CONFIG["threads_tasks"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY не найден.")

    chats_dir = os.path.join(args.output_dir, "graph", args.difficulty)
    os.makedirs(chats_dir, exist_ok=True)

    print(f"\n=== GRAPH PIPELINE | DIFFICULTY: {args.difficulty.upper()} ===")
    print(f"📂 Output: {chats_dir}/{{task_type}}/{{id}}/")
    print(f"📋 Поток: topics → plan → graph_gen → graph_adjust → graph_tasks → evaluate")
    start_time = time.time()

    if args.stage in ("topics", "all"):
        if args.count > 0 or args.reuse_topics:
            run_topics_stage(
                chats_directory=chats_dir,
                difficulty=args.difficulty,
                count=args.count,
                categories=args.categories,
                reuse_topics=args.reuse_topics,
                only_task_types=args.only_task_types,
            )
        elif args.stage == "topics":
            print("ОШИБКА: --stage topics требует --count N или --reuse_topics PATH")
            import sys; sys.exit(1)

    auto_topics = os.path.join(chats_dir, "topics_auto.json")
    topics_file = auto_topics if os.path.exists(auto_topics) else \
                  (args.topics if os.path.exists(args.topics) else None)

    if args.stage in ("plan", "all"):
        run_plan_stage(chats_dir, args.difficulty, topics_file, args.plan_threads)
        if os.path.exists(auto_topics):
            os.remove(auto_topics)
            print(f"  🗑️  {auto_topics} удалён (темы сохранены в директориях)")

    if args.stage in ("graph_gen", "all"):
        run_graph_gen_stage(chats_dir, args.difficulty, args.graph_threads)

    if args.stage in ("graph_adjust", "all"):
        run_graph_adjust_stage(chats_dir, args.difficulty, args.graph_threads)
        if args.stage == "all":
            for regen_round in range(1, 3):
                regen_pending = [
                    d for d in get_all_chat_dirs(chats_dir)
                    if not os.path.exists(os.path.join(d, "graph.json"))
                    and os.path.exists(os.path.join(d, "plan.pickle"))
                ]
                if not regen_pending:
                    break
                print(f"\n  🔁 Повторная генерация [{regen_round}/2] для {len(regen_pending)} граф(а) после adjust...")
                run_graph_gen_stage(chats_dir, args.difficulty, args.graph_threads)
                run_graph_adjust_stage(chats_dir, args.difficulty, args.graph_threads)

    if args.stage in ("graph_tasks", "all"):
        run_graph_tasks_stage(chats_dir, args.difficulty, args.tasks_threads)

    if args.stage == "evaluate":
        run_evaluate_stage(chats_dir, args.models, args.eval_threads)

    total_time = time.time() - start_time
    print(f"\n=== GRAPH PIPELINE FINISHED ===")
    print(f"⏱  Total Time: {total_time:.2f}s")
    print(f"📂 Output: {chats_dir}")
