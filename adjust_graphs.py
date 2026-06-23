import os
import re
import json
import pickle

from json_repair import repair_json

from graph_generator import (
    GRAPH_DIFFICULTY_CONFIG, GRAPH_EDGE_LIMITS, serialize_graph_for_model,
    _validate_graph, _inject_episode_backbone,
)
from utils import get_token_number
from graph_prompts import graph_pad_prompt


# верхний предел узлов = max_nodes * CEILING_FACTOR (сверх него — обрезаем)
_NODE_CEILING_FACTOR = 1.30

# пороги рёбер берутся из единого источника GRAPH_EDGE_LIMITS в graph_generator.py

# максимум попыток pad перед fallback на regen
_PAD_MAX_RETRIES = 2


def _compute_stats(graph: dict, difficulty: str, task_type: str = "") -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    serialized = serialize_graph_for_model(
        graph, difficulty, show_ground_truth=False, for_task_generation=False
    )
    return {
        "difficulty": difficulty,
        "task_type": task_type,
        "nodes": len(nodes),
        "nodes_gt": sum(1 for n in nodes if n.get("ground_truth")),
        "edges": len(edges),
        "edges_gt": sum(1 for e in edges if e.get("ground_truth")),
        "edges_stale": sum(1 for e in edges if e.get("stale")),
        "edges_invalid": sum(1 for e in edges if not e.get("valid", True)),
        "tokens_test_input": get_token_number(serialized),
    }


def _trim_nodes(graph: dict, target_max: int) -> int:
    nodes = graph.get("nodes", [])
    if len(nodes) <= target_max:
        return 0

    protected = [n for n in nodes if n.get("ground_truth") or n.get("type") == "episode"]
    removable  = [n for n in nodes if not n.get("ground_truth") and n.get("type") != "episode"]

    slots = max(0, target_max - len(protected))
    to_remove = {n["id"] for n in removable[slots:] if "id" in n}

    graph["nodes"] = [n for n in nodes if n.get("id") not in to_remove]
    graph["edges"] = [
        e for e in graph.get("edges", [])
        if e.get("source") not in to_remove and e.get("target") not in to_remove
    ]
    return len(to_remove)


def _trim_edges(graph: dict, target_max: int) -> int:
    # приоритет удаления: оба узла non-GT > один узел GT; gt/stale/invalid не трогаем
    edges = graph.get("edges", [])
    if len(edges) <= target_max:
        return 0

    gt_node_ids = {n["id"] for n in graph.get("nodes", []) if n.get("ground_truth")}

    protected = [
        e for e in edges
        if e.get("ground_truth") or e.get("stale") or not e.get("valid", True)
    ]
    plain = [
        e for e in edges
        if not e.get("ground_truth") and not e.get("stale") and e.get("valid", True)
    ]

    # разбиваем по важности: оба non-GT vs. один GT (сравниваем через ID, не объекты)
    both_non_gt = [
        e for e in plain
        if e.get("source") not in gt_node_ids and e.get("target") not in gt_node_ids
    ]
    both_non_gt_ids = {e["id"] for e in both_non_gt if "id" in e}
    one_is_gt = [
        e for e in plain
        if e.get("id") not in both_non_gt_ids
    ]

    removable_ordered = both_non_gt + one_is_gt
    n_to_remove = len(edges) - target_max
    to_remove = {e["id"] for e in removable_ordered[:n_to_remove] if "id" in e}

    graph["edges"] = [e for e in edges if e.get("id") not in to_remove]
    return len(to_remove)


def _load_plan_text(chat_dir: str) -> str:
    plan_path = os.path.join(chat_dir, "plan.pickle")
    if not os.path.exists(plan_path):
        return ""
    try:
        with open(plan_path, "rb") as f:
            batches = pickle.load(f)
        if isinstance(batches, list):
            real = [b for b in batches if b != "[SYNTHETIC PADDING BATCH]"]
            return "\n\n".join(
                f"=== ЭПИЗОД {i+1} ===\n{batch}"
                for i, batch in enumerate(real)
            )
        return str(batches)
    except Exception:
        return ""


def _pad_graph(graph: dict, chat_dir: str, difficulty: str,
               need_nodes: int, need_edges: int) -> bool:
    # изменяет graph на месте; возвращает True если успешно
    from llm import gemini_base as llm

    plan_text = _load_plan_text(chat_dir)
    if not plan_text:
        return False

    current_graph_text = serialize_graph_for_model(
        graph, difficulty, show_ground_truth=False, for_task_generation=True
    )

    # для hard планов (8 батчей) берём больше контекста, чтобы не обрезать поздние эпизоды
    plan_limit = 10000 if difficulty == "hard" else 6000
    prompt = (graph_pad_prompt
              .replace("<plan_text>", plan_text[:plan_limit])
              .replace("<current_graph>", current_graph_text[:3000])
              .replace("<need_nodes>", str(need_nodes))
              .replace("<need_edges>", str(need_edges))
              .replace("<difficulty>", difficulty))

    existing_node_ids = {n["id"] for n in graph["nodes"]}
    existing_edge_ids = {e["id"] for e in graph["edges"]}
    num_episodes = len([n for n in graph["nodes"] if n.get("type") == "episode"])

    for attempt in range(_PAD_MAX_RETRIES):
        try:
            raw = llm.invoke(prompt).content
            clean = re.sub(r'```json|```', '', raw).strip()
            parsed = json.loads(repair_json(clean))

            new_nodes = parsed.get("nodes", [])
            new_edges = parsed.get("edges", [])

            new_nodes = [n for n in new_nodes if n.get("id") not in existing_node_ids]
            new_edges = [e for e in new_edges if e.get("id") not in existing_edge_ids]

            if not new_nodes and not new_edges:
                continue

            # pad никогда не должен добавлять GT-элементы: это сломает GT-скелет
            for n in new_nodes:
                n["ground_truth"] = False
            for e in new_edges:
                e["ground_truth"] = False

            graph["nodes"].extend(new_nodes)
            graph["edges"].extend(new_edges)

            # _validate_graph/_inject_episode_backbone возвращают новый dict —
            # копируем содержимое обратно для мутации на месте
            validated = _validate_graph(graph)
            if num_episodes > 0:
                validated = _inject_episode_backbone(validated, num_episodes)
            graph["nodes"] = validated["nodes"]
            graph["edges"] = validated["edges"]

            existing_node_ids = {n["id"] for n in graph["nodes"]}
            existing_edge_ids = {e["id"] for e in graph["edges"]}

            return True

        except Exception as e:
            print(f"    ⚠️ [pad] попытка {attempt+1}: {e}")

    return False


def check_and_adjust_graph(graph: dict, difficulty: str,
                           chat_dir: str = "") -> tuple[str, str]:
    cfg = GRAPH_DIFFICULTY_CONFIG.get(difficulty, GRAPH_DIFFICULTY_CONFIG["easy"])
    min_nodes, max_nodes = cfg["num_nodes_range"]
    elim = GRAPH_EDGE_LIMITS.get(difficulty, GRAPH_EDGE_LIMITS["easy"])
    min_edges = elim["min"]
    max_edges = elim["max"]  # None для hard

    non_ep = [n for n in graph.get("nodes", []) if n.get("type") != "episode"]
    n_nodes = len(non_ep)
    n_edges = len(graph.get("edges", []))

    need_nodes = max(0, min_nodes - n_nodes)
    need_edges = max(0, min_edges - n_edges)

    if need_nodes > 0 or need_edges > 0:
        if chat_dir:
            success = _pad_graph(graph, chat_dir, difficulty, need_nodes, need_edges)
            if success:
                non_ep_after = [n for n in graph.get("nodes", []) if n.get("type") != "episode"]
                n_nodes_after = len(non_ep_after)
                n_edges_after = len(graph.get("edges", []))
                if n_nodes_after >= min_nodes and n_edges_after >= min_edges:
                    return "pad", (
                        f"дополнено: узлов {n_nodes}→{n_nodes_after}, "
                        f"рёбер {n_edges}→{n_edges_after}"
                    )
                print(f"    ⚠️ [pad] недостаточно: {n_nodes_after} узлов, {n_edges_after} рёбер → regen")
        return "regen", (
            f"мало узлов: {n_nodes} < {min_nodes}" if need_nodes > 0
            else f"мало рёбер: {n_edges} < {min_edges}"
        )

    node_ceiling = int(max_nodes * _NODE_CEILING_FACTOR)
    actions_log = []

    if n_nodes > node_ceiling:
        removed = _trim_nodes(graph, max_nodes)
        actions_log.append(f"обрезка узлов: -{removed} ({n_nodes}→{max_nodes})")

    if max_edges is not None:
        n_edges_now = len(graph.get("edges", []))
        if n_edges_now > max_edges:
            removed = _trim_edges(graph, max_edges)
            actions_log.append(f"обрезка рёбер: -{removed} ({n_edges_now}→{max_edges})")

    if actions_log:
        return "trim", "; ".join(actions_log)

    edge_range = f"{min_edges}–{max_edges}" if max_edges else f"≥{min_edges}"
    return "OK", (
        f"{n_nodes} узлов [{min_nodes}–{max_nodes}], "
        f"{n_edges} рёбер [{edge_range}]"
    )


def adjust_graph(chat_dir: str, difficulty: str) -> dict:
    graph_path = os.path.join(chat_dir, "graph.json")
    check_path = os.path.join(chat_dir, "graph_check.json")
    stats_path = os.path.join(chat_dir, "graph_stats.json")

    if os.path.exists(check_path):
        if os.path.exists(graph_path):
            with open(check_path, encoding="utf-8") as f:
                return json.load(f)
        else:
            # graph.json удалён при regen, check_path устарел — удаляем и перепроверяем
            os.remove(check_path)

    if not os.path.exists(graph_path):
        return {"action": "skip", "reason": "graph.json отсутствует"}

    task_type = ""
    topic_path = os.path.join(chat_dir, "topic.json")
    if os.path.exists(topic_path):
        try:
            with open(topic_path, encoding="utf-8") as f:
                task_type = json.load(f).get("task_type", "")
        except Exception as e:
            print(f"  ⚠️ {chat_dir}: ошибка чтения topic.json: {e}")

    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)

    stats_before = _compute_stats(graph, difficulty, task_type)
    action, reason = check_and_adjust_graph(graph, difficulty, chat_dir)

    if action == "regen":
        # удаляем также tasks_graph.json — задачи были для отклонённого графа
        tasks_path = os.path.join(chat_dir, "tasks_graph.json")
        for path in (graph_path, stats_path, check_path, tasks_path):
            if os.path.exists(path):
                os.remove(path)
        return {"action": "regen", "reason": reason, "stats_before": stats_before}

    if action in ("trim", "pad"):
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        stats_after = _compute_stats(graph, difficulty, task_type)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_after, f, indent=2, ensure_ascii=False)
    else:
        stats_after = stats_before

    result = {
        "action": action,
        "reason": reason,
        "stats_before": stats_before,
        "stats_after": stats_after,
    }
    with open(check_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
