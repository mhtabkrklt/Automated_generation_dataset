"""
graph_generator.py — генерация графов знаний для графового пайплайна BenchEpisodic.

Главная функция: generate_graph(events, topic, difficulty, task_type) -> dict
Вспомогательная: serialize_graph_for_model(graph, difficulty) -> str
"""

import os
import re
import json
import copy
import random
from typing import Any, Optional
from json_repair import repair_json

from llm import gemini_base as llm, validator_llm
from graph_prompts import (
    GRAPH_SERIALIZATION_HEADER, graph_grounding_validation_prompt,
    GRAPH_SKELETON_PROMPTS, graph_fill_prompt, GRAPH_FILL_RULES,
)
from utils import get_token_number

GRAPH_DIFFICULTY_CONFIG: dict[str, dict[str, Any]] = {
    "easy": {
        "num_nodes_range": (40, 60),
        "min_edges": 60,
        "max_edges": 175,
        "noise": False,
        "stale": False,
        "duplicates": False,
        "stale_pct": 0.00,
        "invalid_pct": 0.00,
        "dup_pct": 0.00,
        "min_gt_elements": 5,
        "fill_edge_extra": 20,
        "pre_noise_edge_factor": 1.0,
    },
    "medium": {
        "num_nodes_range": (60, 80),
        "min_edges": 100,
        "max_edges": 280,
        "noise": True,
        "stale": True,
        "duplicates": False,
        "stale_pct": 0.15,
        "invalid_pct": 0.15,
        "dup_pct": 0.05,
        "min_gt_elements": 8,
        "fill_edge_extra": 40,
        "pre_noise_edge_factor": 1.15,
    },
    "hard": {
        "num_nodes_range": (80, 120),
        "min_edges": 150,
        "max_edges": None,  # без ограничений — шум намеренно увеличивает
        "noise": True,
        "stale": True,
        "duplicates": True,
        "stale_pct": 0.25,
        "invalid_pct": 0.25,
        "dup_pct": 0.15,
        "min_gt_elements": 12,
        "fill_edge_extra": 90,
        "pre_noise_edge_factor": 1.28,
    },
}

GRAPH_EDGE_LIMITS = {
    diff: {"min": cfg["min_edges"], "max": cfg["max_edges"]}
    for diff, cfg in GRAPH_DIFFICULTY_CONFIG.items()
}

NODE_TYPES = {"person", "account", "product", "event", "organization", "amount", "document", "rate", "episode"}
EDGE_RELATIONS = {
    "owns", "applied_for", "transferred_to", "changed_to", "caused",
    "resulted_in", "had_rate", "current_rate", "submitted", "approved",
    "rejected", "paid", "received", "signed", "cancelled", "contains",
    "occurred_in", "followed_by",
}

_ATTR_FORMAT_BLOCK = """
═══════════════ СТАНДАРТ АТРИБУТОВ (ОБЯЗАТЕЛЬНО) ═══════════════
Используй единый формат атрибутов во всех узлах и рёбрах:
  Денежная сумма : {"amount": "5000000 руб."}   — цифры без пробелов, "руб." в конце
  Ставка/процент : {"rate": "12.5%"}             — точка в дробях, не запятая
  Дата           : {"date": "15.01.2024"}         — ДД.ММ.ГГГГ
  Период         : {"period": "01.01.2024–01.04.2024"}
  Статус         : {"status": "одобрено"}         — точно из текста плана

ЗАПРЕЩЕНО: дублировать один факт и в атрибутах узла, и в атрибутах ребра.
Если сумма есть в атрибуте узла amount_1 — ребро owns не должно её повторять.
"""

def _get_gt_guidance(task_type: str, difficulty: str) -> str:
    """Возвращает точную GT-инструкцию для данного task_type и difficulty."""
    if task_type == "information_extraction":
        return """
═══ GT-РАЗМЕТКА ДЛЯ INFORMATION_EXTRACTION ═══
ground_truth=true: узел или ребро, содержащее РОВНО ОДИН конкретный факт (сумму/дату/ставку),
который будет ответом. Не помечай GT смежные узлы — только тот, где лежит ответный атрибут.
"""
    if task_type == "knowledge_update":
        return """
═══ GT-РАЗМЕТКА ДЛЯ KNOWLEDGE_UPDATE ═══
ground_truth=true: ВСЕ рёбра цепочки обновлений (had_rate → current_rate) + узел с финальным значением.
Финальное значение определяется по наибольшей дате в attributes.date — не по порядку в JSON.
"""
    if task_type == "temporal_reasoning":
        min_dated = 3 if difficulty in ("medium", "hard") else 2
        return f"""
═══ GT-РАЗМЕТКА ДЛЯ TEMPORAL_REASONING ═══
ground_truth=true: рёбра с полем "date" в attributes, необходимые для вычисления интервала.
ОБЯЗАТЕЛЬНО: минимум {min_dated} GT-рёбра с полем "date" — это жёсткое требование для {difficulty}.
Без {min_dated} GT-рёбер с датами задача temporal_reasoning будет отклонена и граф перегенерируется.
"""
    if task_type == "interference":
        return """
═══ GT-РАЗМЕТКА ДЛЯ INTERFERENCE ═══
ground_truth=true: ТОЛЬКО одно целевое событие среди 3+ похожих. Все остальные похожие — ground_truth=false.
Целевое событие — строго среднее (не первое и не последнее) среди похожих.
"""
    if task_type == "composite":
        return """
═══ GT-РАЗМЕТКА ДЛЯ COMPOSITE ═══
ground_truth=true: все узлы и рёбра целевой причинно-следственной цепочки.
Каждое GT-ребро содержит "chain_step": N в attributes (1, 2, 3...).
Параллельная ложная цепочка — ground_truth=false.
Нет «плавающих» GT-узлов: каждый GT-узел соединён хотя бы одним GT-ребром.
"""
    return ""

_INTER_EPISODE_BLOCK = """
═══════════════ МЕЖЭПИЗОДНЫЕ СВЯЗИ (ОБЯЗАТЕЛЬНО) ═══════════════
Создай минимум 3 рёбра, соединяющих узлы из РАЗНЫХ эпизодов:
  caused       : событие эпизода N привело к событию эпизода N+1
  resulted_in  : результат эпизода стал причиной следующего
  changed_to   : параметр изменился между эпизодами
  approved     : решение одного эпизода повлияло на следующий
Без межэпизодных рёбер граф — набор несвязных кластеров, composite-задачи невозможны.
"""

def _remap_node_type(raw_type: str) -> str:
    """Приводит произвольный тип узла LLM к ближайшему допустимому из NODE_TYPES."""
    if raw_type in NODE_TYPES:
        return raw_type
    t = raw_type.lower()
    for valid in NODE_TYPES - {"episode"}:
        if valid in t:
            return valid
    if any(k in t for k in ("person", "client", "customer", "user", "employee", "staff", "worker", "human")):
        return "person"
    if any(k in t for k in ("loan", "credit", "mortgage", "deposit", "card", "insurance",
                             "invest", "iis", "brokerage", "service", "tariff",
                             "software", "app", "channel", "system", "product")):
        return "product"
    if any(k in t for k in ("bank", "org", "company", "employer", "location",
                             "branch", "office", "city", "address")):
        return "organization"
    if any(k in t for k in ("amount", "sum", "money", "fee", "penalty",
                             "income", "balance", "cost", "price")):
        return "amount"
    if any(k in t for k in ("doc", "contract", "policy", "statement",
                             "agreement", "certificate", "passport", "report")):
        return "document"
    if any(k in t for k in ("rate", "percent", "interest", "yield")):
        return "rate"
    return "event"  # дефолт

def _remap_edge_relation(raw_relation: str) -> str:
    """Приводит произвольный тип ребра LLM к ближайшему допустимому из EDGE_RELATIONS."""
    if raw_relation in EDGE_RELATIONS:
        return raw_relation
    r = raw_relation.lower()
    for valid in EDGE_RELATIONS:
        if valid in r:
            return valid
    if any(k in r for k in ("had_rate", "prev_rate", "old_rate", "previous_rate", "former_rate",
                             "has_rate", "mortgage_rate", "loan_rate", "interest_rate")):
        return "had_rate"
    if any(k in r for k in ("curr_rate", "new_rate", "actual_rate", "current_rate")):
        return "current_rate"
    if any(k in r for k in ("owns", "has_card", "has_account", "has_product", "possess",
                             "hold", "issued_to", "belong", "assigned", "владе")):
        return "owns"
    if any(k in r for k in ("apply", "applied", "request", "ask_for", "filed", "petition",
                             "подал", "подала", "заявк")):
        return "applied_for"
    if any(k in r for k in ("transfer", "send", "move", "wire", "remit", "перевод",
                             "отправ", "переслал")):
        return "transferred_to"
    if any(k in r for k in ("change", "update", "modif", "switch", "adjust", "revis",
                             "alter", "rate_changed", "limit_changed")):
        return "changed_to"
    if any(k in r for k in ("cause", "trigger", "initiat", "lead_to", "provok", "entail")):
        return "caused"
    if any(k in r for k in ("result", "yield", "produc", "generat", "led_to", "outcome")):
        return "resulted_in"
    if any(k in r for k in ("submit", "hand", "upload", "deliver", "provide_doc",
                             "предоставил", "сдал")):
        return "submitted"
    if any(k in r for k in ("approv", "confirm", "accept", "greenlit", "одобр")):
        return "approved"
    if any(k in r for k in ("reject", "declin", "refus", "deny", "отказ")):
        return "rejected"
    if any(k in r for k in ("pay", "paid", "repay", "settl", "платёж", "оплат", "погас")):
        return "paid"
    if any(k in r for k in ("receiv", "got", "obtain", "получил", "получила")):
        return "received"
    if any(k in r for k in ("sign", "execut", "conclud", "подпис", "заключ")):
        return "signed"
    if any(k in r for k in ("cancel", "terminat", "clos", "end", "annul", "отмен", "закрыт")):
        return "cancelled"
    if any(k in r for k in ("occur", "happen", "took_place", "in_episode")):
        return "occurred_in"
    if any(k in r for k in ("follow", "next", "after", "then", "succeed", "subsequ")):
        return "followed_by"
    if any(k in r for k in ("contain", "includ", "consist", "compris")):
        return "contains"
    return "resulted_in"

def _mutate_one_attr(attrs: dict) -> None:
    """Изменяет одно числовое или строковое значение в атрибутах (inplace).
    Используется для создания семантически правдоподобных, но неверных данных."""
    for k, v in list(attrs.items()):
        if isinstance(v, (int, float)) and v > 0:
            factor = random.choice([0.87, 0.91, 1.09, 1.13])
            attrs[k] = round(v * factor, 2) if isinstance(v, float) else int(v * factor)
            return
        if isinstance(v, str):
            nums = re.findall(r'\d+(?:[.,]\d+)?', v)
            if nums:
                num_str = nums[0]
                try:
                    num_val = float(num_str.replace(',', '.'))
                    new_val = round(num_val * random.choice([0.9, 1.1]), 2)
                    new_str = str(int(new_val)) if new_val == int(new_val) else str(new_val).replace('.', ',')
                    attrs[k] = v.replace(num_str, new_str, 1)
                    return
                except ValueError:
                    pass

def _make_similar_label(label: str, node_type: str) -> str:
    """Генерирует похожую метку с минимальным отличием для узла-приманки.

    Для счётов: меняет последнюю цифру номера.
    Для персон: меняет одну гласную в имени.
    Для остальных: добавляет «(доп.)».
    """
    if not label:
        return "(доп.)"
    if node_type == "account":
        m = re.search(r'(\d+)(\D*)$', label)
        if m:
            digits = m.group(1)
            new_last = str((int(digits[-1]) + random.randint(1, 3)) % 10)
            return label[:m.start(1)] + digits[:-1] + new_last + m.group(2)
    if node_type == "person":
        vowel_swaps = {"а": "о", "о": "а", "е": "и", "и": "е", "у": "ю", "ю": "у",
                       "А": "О", "О": "А", "Е": "И", "И": "Е"}
        for i, ch in enumerate(label):
            if ch in vowel_swaps and i > 0:
                return label[:i] + vowel_swaps[ch] + label[i + 1:]
    return label + " (доп.)"

def _llm_validate_graph_grounding(graph: dict, plan_text: str) -> tuple[bool, str]:
    """Проверяет через validator_llm, что ground_truth элементы графа взяты из плана.

    Шумовые элементы (ground_truth=false: stale, invalid, дубликаты) намеренно
    добавлены и НЕ проверяются. Валидируются только ground_truth=true узлы и рёбра.

    Если plan_text содержит маркеры эпизодов (=== ЭПИЗОД N ===), GT-элементы
    группируются по эпизодам через occurred_in рёбра и каждая группа проверяется
    только против текста своего батча — устраняет слепую зону при длинных планах.

    Returns:
        (True, reason) — граф заземлён
        (False, reason) — обнаружены галлюцинации → перегенерация
    """
    gt_nodes = [n for n in graph.get("nodes", []) if n.get("ground_truth")]
    gt_edges = [e for e in graph.get("edges", []) if e.get("ground_truth")]

    if not gt_nodes and not gt_edges:
        return True, "нет ground_truth элементов для проверки (pass)"

    def _format_gt_lines(nodes_list, edges_list):
        lines = []
        for n in nodes_list:
            attrs = json.dumps(n.get("attributes", {}), ensure_ascii=False)
            lines.append(f"[NODE] {n.get('label', n['id'])} | attrs: {attrs}")
        for e in edges_list:
            attrs = json.dumps(e.get("attributes", {}), ensure_ascii=False)
            lines.append(f"[EDGE] {e.get('relation', '')} | attrs: {attrs}")
        return lines

    def _call_validator(batch_text: str, gt_lines: list) -> tuple[bool, str]:
        gt_text = "\n".join(gt_lines)
        prompt = (graph_grounding_validation_prompt
                  .replace("<plan_text>", batch_text[:8000])
                  .replace("<gt_elements>", gt_text[:3000]))
        try:
            response = validator_llm.invoke(prompt).content
            clean = re.sub(r'```json|```', '', response).strip()
            result = json.loads(repair_json(clean))
            verdict = result.get("verdict", "FAIL")
            reason = result.get("reason", "")
            hallucinated = result.get("hallucinated", [])
            grounded = result.get("grounded_count", 0)
            total = result.get("total_count", len(gt_lines))
            detail = f"{grounded}/{total} заземлено"
            if hallucinated:
                detail += f" | проблемы: {'; '.join(str(h) for h in hallucinated[:2])}"
            if verdict == "PASS":
                return True, detail
            return False, f"{reason} | {detail}"
        except Exception as e:
            print(f"  ⚠️ [graph_grounding] Судья недоступен, принимаем: {e}")
            return True, f"fallback PASS (судья недоступен: {e})"

    episode_texts: dict[str, str] = {}
    ep_parts = re.split(r'=== ЭПИЗОД (\d+) ===', plan_text)
    if len(ep_parts) > 1:
        for i in range(1, len(ep_parts), 2):
            ep_num = ep_parts[i].strip()
            ep_text = ep_parts[i + 1].strip() if i + 1 < len(ep_parts) else ""
            episode_texts[f"ep_{ep_num}"] = ep_text

    if episode_texts:
        node_to_ep = {
            e["source"]: e["target"]
            for e in graph.get("edges", [])
            if e.get("relation") == "occurred_in"
        }
        for n in gt_nodes:
            if n["id"] not in node_to_ep:
                hint = str(n.get("episode_hint", "") or
                           n.get("attributes", {}).get("episode_hint", "")).strip()
                ep_key = f"ep_{hint}" if hint else None
                if ep_key and ep_key in episode_texts:
                    node_to_ep[n["id"]] = ep_key
        for e in gt_edges:
            src = e.get("source", "")
            if src not in node_to_ep:
                hint = str(e.get("episode_hint", "") or
                           e.get("attributes", {}).get("episode_hint", "")).strip()
                ep_key = f"ep_{hint}" if hint else None
                if ep_key and ep_key in episode_texts:
                    node_to_ep[src] = ep_key

        ep_to_gt_nodes: dict[str, list] = {}
        ep_to_gt_edges: dict[str, list] = {}
        ungrouped_nodes, ungrouped_edges = [], []

        for n in gt_nodes:
            ep = node_to_ep.get(n["id"])
            if ep and ep in episode_texts:
                ep_to_gt_nodes.setdefault(ep, []).append(n)
            else:
                ungrouped_nodes.append(n)

        for e in gt_edges:
            ep = node_to_ep.get(e.get("source", ""))
            if ep and ep in episode_texts:
                ep_to_gt_edges.setdefault(ep, []).append(e)
            else:
                ungrouped_edges.append(e)

        results: list[tuple[bool, str, str]] = []
        for ep_id in set(ep_to_gt_nodes) | set(ep_to_gt_edges):
            gt_lines = _format_gt_lines(ep_to_gt_nodes.get(ep_id, []), ep_to_gt_edges.get(ep_id, []))
            if gt_lines:
                ok, reason = _call_validator(episode_texts[ep_id], gt_lines)
                results.append((ok, reason, ep_id))

        ungrouped_lines = _format_gt_lines(ungrouped_nodes, ungrouped_edges)
        if ungrouped_lines:
            ok, reason = _call_validator(plan_text[:8000], ungrouped_lines)
            results.append((ok, reason, "общий"))

        if not results:
            return True, "нет GT-элементов для проверки"

        failed = [(ep, r) for ok, r, ep in results if not ok]
        passed = sum(1 for ok, _, _ in results if ok)
        if failed:
            reasons = "; ".join(f"[{ep}] {r}" for ep, r in failed[:2])
            return False, f"галлюцинации в {len(failed)}/{len(results)} эпизодах: {reasons}"
        return True, f"{passed}/{len(results)} эпизодов заземлены"

    gt_lines = _format_gt_lines(gt_nodes, gt_edges)
    return _call_validator(plan_text[:4000], gt_lines)

def _fix_date_attr(val: str) -> str | None:
    """Нормализует даты с нестандартными разделителями.

    Поддерживаемые форматы ввода:
      - ДД,ММ.ГГГГ (запятая вместо точки)  → ДД.ММ.ГГГГ
      - ДД/ММ/ГГГГ (слеш)                  → ДД.ММ.ГГГГ
      - ДД ММ ГГГГ (пробел)                → ДД.ММ.ГГГГ

    Возвращает исправленную строку или None если дата невалидна (месяц > 12, день > 31).
    """
    fixed = re.sub(
        r'(\d{1,2})[,/\s](\d{2})[./\s](\d{4})',
        lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)}",
        val,
    )
    if fixed == val:
        return val  # нет паттерна — возвращаем как есть
    m = re.search(r'(\d{1,2})\.(\d{2})\.(\d{4})', fixed)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return fixed
    return None  # невалидная дата — удаляем атрибут

def _validate_node(node: dict) -> dict:
    """Нормализует и валидирует узел. Возвращает исправленный узел."""
    node.setdefault("id", f"node_{random.randint(1000, 9999)}")
    node["type"] = _remap_node_type(node.get("type", "event"))
    node.setdefault("label", node["id"])
    if not isinstance(node.get("attributes"), dict):
        node["attributes"] = {}
    node.setdefault("ground_truth", False)
    node["ground_truth"] = bool(node["ground_truth"])
    cleaned = {}
    for k, v in node["attributes"].items():
        if isinstance(v, str):
            fixed = _fix_date_attr(v)
            if fixed is not None:
                cleaned[k] = fixed
        else:
            cleaned[k] = v
    node["attributes"] = cleaned
    return node

_STRUCTURAL_RELATIONS = {"occurred_in", "followed_by"}

def _validate_edge(edge: dict, node_ids: set) -> dict | None:
    """Нормализует ребро. Возвращает None если source/target не существуют или self-loop."""
    edge.setdefault("id", f"edge_{random.randint(1000, 9999)}")
    src = edge.get("source", "")
    tgt = edge.get("target", "")
    if not src or not tgt or src == tgt:
        return None
    if src not in node_ids or tgt not in node_ids:
        return None
    edge["relation"] = _remap_edge_relation(edge.get("relation", "contains"))
    if not isinstance(edge.get("attributes"), dict):
        edge["attributes"] = {}
    edge.setdefault("valid", True)
    edge.setdefault("stale", False)
    edge.setdefault("ground_truth", False)
    edge["valid"] = bool(edge["valid"])
    edge["stale"] = bool(edge["stale"])
    edge["ground_truth"] = bool(edge["ground_truth"])
    if edge.get("relation") in _STRUCTURAL_RELATIONS:
        edge["ground_truth"] = False
    return edge

def _validate_graph(raw: dict) -> dict:
    """Валидирует и нормализует весь граф из ответа LLM."""
    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])

    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    valid_nodes = [_validate_node(n) for n in nodes if isinstance(n, dict)]
    node_ids = {n["id"] for n in valid_nodes}

    valid_edges = []
    for e in edges:
        if isinstance(e, dict):
            result = _validate_edge(e, node_ids)
            if result is not None:
                valid_edges.append(result)

    connected_ids = set()
    for e in valid_edges:
        connected_ids.add(e["source"])
        connected_ids.add(e["target"])
    valid_nodes = [
        n for n in valid_nodes
        if n.get("type") == "episode" or n.get("ground_truth") or n["id"] in connected_ids
    ]

    return {"nodes": valid_nodes, "edges": valid_edges}

def _check_graph_quality(graph: dict, difficulty: str, task_type: str = "") -> tuple[bool, str]:
    """Проверяет качество финального графа (после inject_noise).

    Вызывается как диагностика — возвращает предупреждение, не блокирует пайплайн.

    Проверки:
    1. Минимальное число узлов и рёбер для уровня сложности
    2. Наличие хотя бы одного ground_truth=true узла или ребра
    3. Узлы имеют осмысленные метки (не только ID)
    4. Узлы имеют непустые атрибуты (ключевые факты должны быть в атрибутах)
    5. Для medium/hard: наличие хотя бы одного stale ребра (добавлено inject_noise)
    6. Для temporal_reasoning: минимум 2/3 GT-рёбер с датами
    7. Минимальное число GT-элементов (из GRAPH_DIFFICULTY_CONFIG)
    8. Для composite: GT-рёбра образуют связный подграф
    """
    cfg = GRAPH_DIFFICULTY_CONFIG.get(difficulty, GRAPH_DIFFICULTY_CONFIG["easy"])
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    min_nodes = cfg["num_nodes_range"][0]
    min_edges = GRAPH_EDGE_LIMITS.get(difficulty, GRAPH_EDGE_LIMITS["easy"])["min"]

    if len(nodes) < min_nodes:
        return False, f"мало узлов: {len(nodes)} < {min_nodes} (min для {difficulty})"
    if len(edges) < min_edges:
        return False, f"мало рёбер: {len(edges)} < {min_edges}"

    has_gt = any(n.get("ground_truth") for n in nodes) or any(e.get("ground_truth") for e in edges)
    if not has_gt:
        return False, "нет ни одного элемента с ground_truth=true"

    trivial_labels = sum(1 for n in nodes if n.get("label", "") == n.get("id", "X"))
    if trivial_labels > len(nodes) * 0.5:
        return False, f"больше половины узлов имеют тривиальные метки (label==id): {trivial_labels}/{len(nodes)}"

    nodes_with_attrs = sum(1 for n in nodes if n.get("attributes"))
    if nodes_with_attrs < max(1, len(nodes) * 0.3):
        return False, f"мало узлов с атрибутами: {nodes_with_attrs}/{len(nodes)}"

    if cfg.get("stale"):
        has_stale = any(e.get("stale") for e in edges)
        if not has_stale:
            return False, f"нет stale рёбер для difficulty={difficulty} (LLM не добавил устаревшие связи)"

    if task_type == "temporal_reasoning":
        date_keys = {"date", "timestamp", "since", "until", "from", "period"}
        gt_dated = [e for e in edges if e.get("ground_truth") and date_keys & set(e.get("attributes", {}))]
        min_dated = 3 if difficulty in ("medium", "hard") else 2
        if len(gt_dated) < min_dated:
            return False, f"temporal_reasoning: мало GT-рёбер с датами: {len(gt_dated)} < {min_dated} (для {difficulty})"

    min_gt = cfg.get("min_gt_elements", 5)
    n_gt = (sum(1 for n in nodes if n.get("ground_truth"))
            + sum(1 for e in edges if e.get("ground_truth")))
    if n_gt < min_gt:
        return False, f"мало GT-элементов: {n_gt} < {min_gt} (min для {difficulty})"

    if task_type == "composite":
        gt_edges = [e for e in edges if e.get("ground_truth")]
        if len(gt_edges) < 3:
            return False, f"composite: недостаточно GT-рёбер: {len(gt_edges)} < 3 (нужна цепочка ≥3 шагов)"
        gt_adj: dict[str, set] = {}
        for e in gt_edges:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src and tgt:
                gt_adj.setdefault(src, set()).add(tgt)
                gt_adj.setdefault(tgt, set()).add(src)
        if gt_adj:
            start = next(iter(gt_adj))
            visited = {start}
            queue = [start]
            while queue:
                node = queue.pop()
                for nb in gt_adj.get(node, set()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(visited) < len(gt_adj) * 0.7:
                return False, (f"composite: GT-подграф несвязен "
                               f"({len(visited)}/{len(gt_adj)} узлов достижимы из одной точки)")

    return True, "OK"

def inject_noise(graph: dict, difficulty: str) -> dict:
    """Добавляет шум в граф согласно GRAPH_DIFFICULTY_CONFIG.

    stale рёбра:     15% (medium) / 25% (hard) от non-GT валидных рёбер — помечаются stale
    invalid рёбра:   15% (medium) / 25% (hard) от всех рёбер — добавляются копии с valid=False
    дубликаты узлов:  5% (medium) / 15% (hard) от non-GT узлов с атрибутами
    """
    cfg = GRAPH_DIFFICULTY_CONFIG.get(difficulty, GRAPH_DIFFICULTY_CONFIG["easy"])

    if cfg["stale"]:
        non_gt_valid_edges = [
            e for e in graph["edges"]
            if not e.get("ground_truth") and e.get("valid", True) and not e.get("stale")
            and e.get("relation") not in _STRUCTURAL_RELATIONS
        ]
        if non_gt_valid_edges:
            stale_target = max(1, int(len(non_gt_valid_edges) * cfg["stale_pct"]))
            for edge in random.sample(non_gt_valid_edges, min(stale_target, len(non_gt_valid_edges))):
                edge["stale"] = True
                edge["ground_truth"] = False  # stale ребро не является ground_truth

    if cfg.get("stale") and difficulty in ("medium", "hard"):
        date_keys = {"date", "timestamp", "since", "until", "from", "period"}
        gt_timed = [
            e for e in graph["edges"]
            if e.get("ground_truth") and e.get("valid", True)
            and date_keys & set(e.get("attributes", {}))
        ]
        contr_count = min(3, max(1, len(gt_timed)))
        existing_ids = {e["id"] for e in graph["edges"]}
        for i, orig in enumerate(random.sample(gt_timed, min(contr_count, len(gt_timed)))):
            contr_id = f"edge_contr{i + 1}"
            if contr_id in existing_ids:
                contr_id += f"_{random.randint(100, 999)}"
            fake_attrs = copy.deepcopy(orig.get("attributes", {}))
            for dk in date_keys:
                if dk in fake_attrs:
                    date_str = str(fake_attrs[dk])
                    m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', date_str)
                    if m:
                        d, mo, y = m.groups()
                        new_mo = int(mo) - random.randint(1, 2)
                        new_y = int(y)
                        if new_mo < 1:
                            new_mo += 12
                            new_y -= 1
                        fake_attrs[dk] = (date_str[:m.start()]
                                          + f"{d}.{new_mo:02d}.{new_y}"
                                          + date_str[m.end():])
                    break
            _mutate_one_attr(fake_attrs)
            graph["edges"].append({
                "id": contr_id,
                "source": orig.get("source", ""),
                "target": orig.get("target", ""),
                "relation": orig.get("relation", "contains"),
                "attributes": fake_attrs,
                "valid": True,   # намеренно valid=True — нет маркера «недостоверно»
                "stale": False,  # нет маркера stale — только даты выдают устарелость
                "ground_truth": False,
            })

    if cfg["noise"] and len(graph["edges"]) >= 2:
        invalid_count = max(1, int(len(graph["edges"]) * cfg["invalid_pct"]))
        existing_ids = {e["id"] for e in graph["edges"]}
        candidates = [e for e in graph["edges"] if e.get("valid", True) and e.get("attributes")]
        if not candidates:
            candidates = [e for e in graph["edges"] if e.get("valid", True)]
        all_node_ids = [n["id"] for n in graph["nodes"]]
        for i in range(invalid_count):
            noise_id = f"edge_inv{i + 1}"
            if noise_id in existing_ids:
                noise_id += f"_{random.randint(100, 999)}"
            if candidates and all_node_ids:
                orig = random.choice(candidates)
                fake_attrs = copy.deepcopy(orig.get("attributes", {}))
                _mutate_one_attr(fake_attrs)
                src_id = orig.get("source", "")
                tgt_id = orig.get("target", "")
                other_ids = [nid for nid in all_node_ids if nid not in (src_id, tgt_id)]
                if other_ids:
                    if random.random() < 0.5:
                        tgt_id = random.choice(other_ids)
                    else:
                        src_id = random.choice(other_ids)
                graph["edges"].append({
                    "id": noise_id,
                    "source": src_id,
                    "target": tgt_id,
                    "relation": orig.get("relation", "contains"),
                    "attributes": fake_attrs,
                    "valid": False,
                    "stale": False,
                    "ground_truth": False,
                })
            else:
                src, tgt = random.sample(graph["nodes"], 2)
                graph["edges"].append({
                    "id": noise_id,
                    "source": src["id"],
                    "target": tgt["id"],
                    "relation": "contains",
                    "attributes": {},
                    "valid": False,
                    "stale": False,
                    "ground_truth": False,
                })

    if cfg.get("duplicates") or cfg["dup_pct"] > 0:
        non_gt_nodes = [
            n for n in graph["nodes"]
            if not n.get("ground_truth") and n.get("type") not in ("episode",)
            and n.get("attributes")
        ]
        dup_count = max(1, int(len(non_gt_nodes) * cfg["dup_pct"]))
        existing_node_ids = {n["id"] for n in graph["nodes"]}
        candidates = random.sample(non_gt_nodes, min(dup_count, len(non_gt_nodes)))
        for orig in candidates:
            dup_id = f"dup_{orig['id']}"
            if dup_id in existing_node_ids:
                continue
            dup_attrs = copy.deepcopy(orig.get("attributes", {}))
            _mutate_one_attr(dup_attrs)
            graph["nodes"].append({
                "id": dup_id,
                "type": orig.get("type", "event"),
                "label": _make_similar_label(orig.get("label", dup_id), orig.get("type", "event")),
                "attributes": dup_attrs,
                "ground_truth": False,
            })

    return graph

def serialize_graph_for_model(graph: dict, difficulty: str = "easy",
                              show_ground_truth: bool = False,
                              for_task_generation: bool = False,
                              task_type: str = "") -> str:
    """Преобразует граф в читаемый текст для передачи модели.

    Args:
        show_ground_truth: если True — добавляет маркер [★ ключевой факт] к ground_truth
            элементам. Используется ТОЛЬКО для генератора задач.
        for_task_generation: если True — генератор задач видит полную эпизодную структуру
            (occurred_in, followed_by, шкала) на всех уровнях сложности, чтобы создавать
            корректные вопросы. Тестируемая модель всегда получает False.
        task_type: тип задачи. Для "temporal_reasoning" + for_task_generation=False —
            даты в атрибутах non-GT рёбер скрываются, чтобы модель не могла найти ответ
            простым сканированием. Модель должна навигировать к релевантным рёбрам
            из контекста вопроса.

    Видимость эпизодной структуры (для тестируемой модели, for_task_generation=False):
        easy   — followed_by и occurred_in видны; шкала скрыта
        medium — followed_by видно, occurred_in скрыто; шкала скрыта
        hard   — followed_by и occurred_in скрыты; шкала скрыта; рёбра перемешаны
    """
    lines = [GRAPH_SERIALIZATION_HEADER]

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_index = {n["id"]: n for n in nodes}

    if for_task_generation:
        hide_timeline = False
        hide_occurred_in = False
        hide_followed_by = False
        hide_dates_non_gt = False
    else:
        hide_timeline = True
        hide_occurred_in = (difficulty in ("medium", "hard"))
        hide_followed_by = (difficulty == "hard")
        hide_dates_non_gt = (task_type == "temporal_reasoning")

    if not hide_timeline:
        episode_nodes = sorted(
            [n for n in nodes if n.get("type") == "episode"],
            key=lambda n: n.get("attributes", {}).get("order", 0),
        )
        if episode_nodes:
            ep_parts = []
            for ep in episode_nodes:
                label = ep.get("label", ep["id"])
                summary = ep.get("attributes", {}).get("summary", "")
                ep_parts.append(f"{label}" + (f" ({summary})" if summary else ""))
            lines.append("ВРЕМЕННАЯ ШКАЛА ЭПИЗОДОВ:")
            lines.append("  " + " → ".join(ep_parts))
            lines.append("")

    non_episode_nodes = [n for n in nodes if n.get("type") != "episode"]
    for node in non_episode_nodes:
        node_type = node.get("type", "entity")
        label = node.get("label", node.get("id", "?"))
        attrs = node.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        is_gt = node.get("ground_truth", False)

        attr_parts = []
        for k, v in attrs.items():
            if v not in (None, "", []):
                attr_parts.append(f"{k}: {v}")
        attr_str = f" ({', '.join(attr_parts[:4])})" if attr_parts else ""

        gt_marker = " [★ ключевой факт]" if (show_ground_truth and is_gt) else ""
        lines.append(f"• [{node_type}] {label}{attr_str}{gt_marker}")

    lines.append("\nСОБЫТИЯ И СВЯЗИ:")

    display_edges = []
    for e in edges:
        rel = e.get("relation", "")
        if hide_occurred_in and rel == "occurred_in":
            continue
        if hide_followed_by and rel == "followed_by":
            continue
        display_edges.append(e)

    if difficulty == "hard" and len(display_edges) > 1:
        seed = hash(tuple(sorted(n["id"] for n in nodes))) % (2 ** 32)
        random.Random(seed).shuffle(display_edges)

    for edge in display_edges:
        src_id = edge.get("source", "")
        tgt_id = edge.get("target", "")
        relation = edge.get("relation", "связано")
        attrs = edge.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        valid = edge.get("valid", True)
        stale = edge.get("stale", False)
        is_gt_edge = edge.get("ground_truth", False)

        src_label = node_index.get(src_id, {}).get("label", src_id)
        tgt_label = node_index.get(tgt_id, {}).get("label", tgt_id)

        _date_keys = {"date", "timestamp", "period", "from_date", "to_date", "since", "until"}
        _date_pattern = re.compile(r'^\d{2}\.\d{2}\.\d{4}')
        _skip_keys = {"note", "chain_step", "episode_hint"}
        edge_attr_parts = []
        for k, v in attrs.items():
            if v in (None, "", []) or k in _skip_keys:
                continue
            v_str = str(v)
            if k in _date_keys or _date_pattern.match(v_str):
                fixed = _fix_date_attr(v_str)
                if fixed is None:
                    continue  # невалидная дата — пропускаем
                v_str = fixed
            if (hide_dates_non_gt and not is_gt_edge
                    and (k in _date_keys or _date_pattern.match(v_str))):
                edge_attr_parts.append("[дата скрыта]")
            else:
                edge_attr_parts.append(f"{k}: {v_str}")
        attr_suffix = " | " + " | ".join(edge_attr_parts[:4]) if edge_attr_parts else ""

        markers = []
        if difficulty == "easy":
            if stale:
                markers.append("[⚠ устарело]")
            if not valid:
                markers.append("[✗ недостоверно]")
        elif difficulty == "medium":
            if stale:
                markers.append("[⚠ устарело]")

        if show_ground_truth and is_gt_edge:
            markers.append("[★ ключевой факт]")

        marker_str = "  " + " ".join(markers) if markers else ""
        lines.append(f"• {src_label} --[{relation}]--> {tgt_label}{attr_suffix}{marker_str}")

    return "\n".join(lines)

def _inject_episode_backbone(graph: dict, num_episodes: int) -> dict:
    """Программно добавляет узлы-эпизоды, followed_by цепочку и occurred_in рёбра.

    Гарантирует три вещи независимо от того, что сделал LLM:
    1. Узлы ep_1..ep_N типа episode
    2. followed_by цепочка ep_1 → ep_2 → ... → ep_N
    3. occurred_in рёбра: каждый ground_truth узел (не episode) привязан к своему эпизоду.
       Если LLM уже создал occurred_in — новые не дублируются.
       Ground_truth узлы распределяются по эпизодам равномерно по порядку в графе.
    """
    if num_episodes < 1:
        return graph

    existing_node_ids = {n["id"] for n in graph["nodes"]}
    existing_edge_ids = {e["id"] for e in graph["edges"]}
    ep_ids_present = {n["id"] for n in graph["nodes"] if n.get("type") == "episode"}

    for i in range(1, num_episodes + 1):
        ep_id = f"ep_{i}"
        if ep_id not in ep_ids_present and ep_id not in existing_node_ids:
            graph["nodes"].append({
                "id": ep_id,
                "type": "episode",
                "label": f"Эпизод {i}",
                "attributes": {"order": i},
                "ground_truth": False,
            })

    for i in range(1, num_episodes):
        edge_id = f"ep_chain_{i}"
        if edge_id not in existing_edge_ids:
            graph["edges"].append({
                "id": edge_id,
                "source": f"ep_{i}",
                "target": f"ep_{i + 1}",
                "relation": "followed_by",
                "attributes": {},
                "valid": True,
                "stale": False,
                "ground_truth": False,
            })

    nodes_with_occurred_in = {
        e["source"] for e in graph["edges"] if e.get("relation") == "occurred_in"
    }
    gt_nodes = [
        n for n in graph["nodes"]
        if n.get("ground_truth") and n.get("type") != "episode"
        and n["id"] not in nodes_with_occurred_in
    ]
    for idx, node in enumerate(gt_nodes):
        hint = node.get("episode_hint")
        ep_num = (idx % num_episodes) + 1
        if hint is not None:
            try:
                h = int(hint)
                if 1 <= h <= num_episodes:
                    ep_num = h
            except (ValueError, TypeError):
                pass
        edge_id = f"ep_occurs_{node['id']}"
        if edge_id not in existing_edge_ids:
            graph["edges"].append({
                "id": edge_id,
                "source": node["id"],
                "target": f"ep_{ep_num}",
                "relation": "occurred_in",
                "attributes": {},
                "valid": True,
                "stale": False,
                "ground_truth": False,
            })

    return graph

def _validate_skeleton(skeleton: dict, task_type: str, difficulty: str) -> tuple[bool, str]:
    """Валидация GT-скелета: количество, task-type структура, episode_hint."""
    gt_nodes = skeleton.get("gt_nodes", [])
    gt_edges = skeleton.get("gt_edges", [])
    total_gt = len(gt_nodes) + len(gt_edges)

    if task_type == "knowledge_update":
        _min_gt = {"easy": 4, "medium": 7, "hard": 10}
    else:
        _min_gt = {"easy": 5, "medium": 8, "hard": 12}
    min_gt = _min_gt.get(difficulty, 5)
    if total_gt < min_gt:
        return False, f"мало GT-элементов: {total_gt} < {min_gt}"

    missing_hint = [n.get("id", "?") for n in gt_nodes if not n.get("episode_hint")]
    missing_hint += [e.get("id", "?") for e in gt_edges if not e.get("episode_hint")]
    if missing_hint:
        return False, f"нет episode_hint у {len(missing_hint)} элементов: {missing_hint[:3]}"

    if task_type == "temporal_reasoning":
        date_keys = {"date", "timestamp", "since", "until"}
        min_dated = 3 if difficulty in ("medium", "hard") else 2
        gt_dated = [e for e in gt_edges
                    if e.get("ground_truth", False) and date_keys & set(e.get("attributes", {}))]
        if len(gt_dated) < min_dated:
            return False, f"temporal: GT-рёбер с датами {len(gt_dated)} < {min_dated}"

    if task_type == "composite":
        chain_edges = [e for e in gt_edges
                       if "chain_step" in e.get("attributes", {}) and e.get("ground_truth", False)]
        min_chain = {"easy": 3, "medium": 4, "hard": 5}.get(difficulty, 3)
        if len(chain_edges) < min_chain:
            return False, f"composite: цепочка {len(chain_edges)} шагов < {min_chain}"
        edges_sorted = sorted(chain_edges, key=lambda e: e["attributes"]["chain_step"])
        for i in range(len(edges_sorted) - 1):
            if edges_sorted[i].get("target") != edges_sorted[i + 1].get("source"):
                return False, (f"composite: цепочка несвязна: шаг {i+1} target="
                               f"{edges_sorted[i].get('target')} ≠ шаг {i+2} source="
                               f"{edges_sorted[i+1].get('source')}")

    if task_type == "interference":
        from collections import Counter
        all_gt_edges = [e for e in gt_edges if e.get("ground_truth", True)
                        and e.get("relation") not in _STRUCTURAL_RELATIONS]
        rel_counts = Counter(e.get("relation") for e in all_gt_edges)
        max_same_rel = max(rel_counts.values()) if rel_counts else 0
        min_similar = {"easy": 2, "medium": 3, "hard": 3}.get(difficulty, 2)
        if max_same_rel < min_similar:
            return False, f"interference: похожих рёбер {max_same_rel} < {min_similar}"

    if task_type == "knowledge_update":
        update_relations = {"had_rate", "current_rate", "rate_changed", "changed_to"}
        _update_attr_keys = {"value", "from_value", "to_value", "rate", "amount",
                             "percent", "sum", "balance", "limit", "payment"}
        _date_keys = {"date", "timestamp", "since", "until"}
        def _is_update_edge(e: dict) -> bool:
            if e.get("relation") in update_relations:
                return True
            attrs = e.get("attributes", {})
            if bool(_update_attr_keys & set(attrs)):
                return True
            if bool(_date_keys & set(attrs)):
                for v in attrs.values():
                    if isinstance(v, (int, float)):
                        return True
                    if isinstance(v, str) and any(c.isdigit() for c in v):
                        return True
            return False
        update_edges = [e for e in gt_edges if _is_update_edge(e)]
        min_updates = {"easy": 2, "medium": 2, "hard": 3}.get(difficulty, 2)
        if len(update_edges) < min_updates:
            return False, f"knowledge_update: рёбер обновления {len(update_edges)} < {min_updates}"

    return True, f"OK ({total_gt} GT-элементов)"

def _serialize_skeleton_for_fill(skeleton: dict) -> str:
    """Сериализует GT-скелет в текст для fill-промпта."""
    lines = ["GT-УЗЛЫ (не дублировать):"]
    for n in skeleton.get("gt_nodes", []):
        attrs = json.dumps(n.get("attributes", {}), ensure_ascii=False)
        gt_mark = "[GT]" if n.get("ground_truth") else "[ctx]"
        lines.append(f"  {gt_mark} {n['id']} [{n.get('type','')}] {n.get('label','')} {attrs}"
                     f" ep={n.get('episode_hint','?')}")
    lines.append("GT-РЁБРА (не дублировать):")
    for e in skeleton.get("gt_edges", []):
        attrs = json.dumps(e.get("attributes", {}), ensure_ascii=False)
        gt_mark = "[GT]" if e.get("ground_truth") else "[ctx]"
        lines.append(f"  {gt_mark} {e['id']} {e.get('source','')}→{e.get('relation','')}→"
                     f"{e.get('target','')} {attrs} ep={e.get('episode_hint','?')}")
    return "\n".join(lines)

def _merge_skeleton_fill(skeleton: dict, fill: dict) -> dict:
    """Объединяет GT-скелет и non-GT наполнение в единый граф.

    GT-элементы всегда имеют приоритет — если fill содержит элемент с тем же id,
    он отбрасывается (нельзя перезаписать GT-факт non-GT контекстом).
    """
    gt_node_ids = {n["id"] for n in skeleton.get("gt_nodes", []) if "id" in n}
    gt_edge_ids = {e["id"] for e in skeleton.get("gt_edges", []) if "id" in e}

    fill_nodes = [n for n in fill.get("nodes", []) if n.get("id") not in gt_node_ids]
    fill_edges = [e for e in fill.get("edges", []) if e.get("id") not in gt_edge_ids]

    nodes = list(skeleton.get("gt_nodes", [])) + fill_nodes
    edges = list(skeleton.get("gt_edges", [])) + fill_edges
    return {"nodes": nodes, "edges": edges}

def _check_cross_episode_semantics(graph: dict, task_type: str,
                                    num_episodes: int) -> tuple[bool, str]:
    """Финальная проверка: GT-элементы корректно распределены по эпизодам.

    Вызывается после _inject_episode_backbone() — маппинг node→ep уже есть.
    """
    if num_episodes < 2:
        return True, "один эпизод — проверка не нужна"

    node_to_ep_num: dict[str, int] = {}
    for e in graph.get("edges", []):
        if e.get("relation") == "occurred_in":
            ep_id = e.get("target", "")
            try:
                ep_num = int(ep_id.split("_")[1])
                node_to_ep_num[e["source"]] = ep_num
            except (IndexError, ValueError):
                pass

    gt_edges = [e for e in graph.get("edges", []) if e.get("ground_truth")]

    if task_type == "composite":
        chain_edges = sorted(
            [e for e in gt_edges if "chain_step" in e.get("attributes", {})],
            key=lambda e: e["attributes"]["chain_step"]
        )
        ep_nums_raw = [node_to_ep_num.get(e.get("source")) for e in chain_edges]
        ep_nums: list[int] = [n for n in ep_nums_raw if n is not None]
        if len(ep_nums) >= 2:
            if ep_nums != sorted(ep_nums):
                return False, f"composite: chain_step порядок не совпадает с эпизодами {ep_nums}"
            if len(set(ep_nums)) < 2:
                return False, "composite: вся цепочка в одном эпизоде"

    if task_type == "knowledge_update":
        update_eps = {node_to_ep_num.get(e.get("source"))
                      for e in gt_edges if node_to_ep_num.get(e.get("source"))}
        if len(update_eps) < 2:
            return False, "knowledge_update: все обновления в одном эпизоде"

    if task_type == "temporal_reasoning":
        date_keys = {"date", "timestamp", "since"}
        dated_gt = [e for e in gt_edges if date_keys & set(e.get("attributes", {}))]
        dated_eps = {node_to_ep_num.get(e.get("source"))
                     for e in dated_gt if node_to_ep_num.get(e.get("source"))}
        if len(dated_eps) < 2:
            return False, "temporal_reasoning: GT-даты сосредоточены в одном эпизоде"

    if task_type == "interference":
        from collections import Counter
        rel_counts = Counter(e.get("relation") for e in gt_edges)
        if rel_counts:
            top_rel = rel_counts.most_common(1)[0][0]
            similar = [e for e in gt_edges if e.get("relation") == top_rel]
            sim_eps = {node_to_ep_num.get(e.get("source"))
                       for e in similar if node_to_ep_num.get(e.get("source"))}
            if len(sim_eps) < 2:
                return False, "interference: похожие GT-события в одном эпизоде"

    return True, "OK"

def _remap_episode_assignments(graph: dict, task_type: str, num_episodes: int) -> bool:
    """Переназначает occurred_in рёбра для GT-узлов чтобы исправить порядок.

    Для composite: chain_step N → ep_N (сохраняя хронологию).
    Для остальных: GT-узлы распределяются равномерно по эпизодам.
    Возвращает True если ремаппинг выполнен.
    """
    if num_episodes < 2:
        return False

    gt_node_ids = {n["id"] for n in graph.get("nodes", []) if n.get("ground_truth")}
    graph["edges"] = [
        e for e in graph.get("edges", [])
        if not (e.get("relation") == "occurred_in" and e.get("source") in gt_node_ids)
    ]

    existing_edge_ids = {e["id"] for e in graph["edges"]}

    if task_type == "composite":
        gt_edges_by_step = {}
        for e in graph.get("edges", []):
            if e.get("ground_truth") and "chain_step" in e.get("attributes", {}):
                step = e["attributes"]["chain_step"]
                gt_edges_by_step[step] = e

        assigned: set[str] = set()
        for step, edge in sorted(gt_edges_by_step.items()):
            ep_num = min(step, num_episodes)
            for node_id in (edge.get("source"), edge.get("target")):
                if node_id and node_id in gt_node_ids and node_id not in assigned:
                    edge_id = f"ep_remap_{node_id}"
                    if edge_id not in existing_edge_ids:
                        graph["edges"].append({
                            "id": edge_id, "source": node_id, "target": f"ep_{ep_num}",
                            "relation": "occurred_in", "attributes": {},
                            "valid": True, "stale": False, "ground_truth": False,
                        })
                    assigned.add(node_id)
    else:
        gt_nodes = [n for n in graph.get("nodes", []) if n.get("ground_truth")]
        for idx, node in enumerate(gt_nodes):
            ep_num = (idx % num_episodes) + 1
            edge_id = f"ep_remap_{node['id']}"
            if edge_id not in existing_edge_ids:
                graph["edges"].append({
                    "id": edge_id, "source": node["id"], "target": f"ep_{ep_num}",
                    "relation": "occurred_in", "attributes": {},
                    "valid": True, "stale": False, "ground_truth": False,
                })

    return True

def _validate_fill(graph: dict, difficulty: str) -> tuple[bool, str]:
    """Проверяет размер и качество наполненного графа (без LLM).

    ВАЖНО: вызывается ДО inject_noise, поэтому порог рёбер снижается
    на noise-коэффициент. inject_noise добавляет ~25-30% рёбер для hard/medium
    (invalid-копии + temporal-дубли). Без этого fill с 130 рёбрами
    отвергается, хотя после noise даст 166 ≥ 150 — основная причина retry-петель.
    """
    cfg = GRAPH_DIFFICULTY_CONFIG.get(difficulty, GRAPH_DIFFICULTY_CONFIG["easy"])
    min_nodes = cfg["num_nodes_range"][0]
    min_edges = cfg["min_edges"]

    nodes = [n for n in graph.get("nodes", []) if n.get("type") != "episode"]
    edges = graph.get("edges", [])

    if len(nodes) < min_nodes:
        return False, f"мало узлов: {len(nodes)} < {min_nodes}"

    pre_noise_min = int(min_edges / cfg["pre_noise_edge_factor"])
    if len(edges) < pre_noise_min:
        return False, f"мало рёбер: {len(edges)} < {pre_noise_min} (до noise; целевой минимум {min_edges})"

    nodes_with_attrs = sum(1 for n in nodes if n.get("attributes"))
    if nodes_with_attrs < max(1, len(nodes) * 0.3):
        return False, f"мало узлов с атрибутами: {nodes_with_attrs}/{len(nodes)}"

    return True, f"OK ({len(nodes)} узлов, {len(edges)} рёбер)"

def _format_profile_for_prompt(profile_data: dict) -> str:
    """Форматирует user_profile.json в текстовый блок для промпта."""
    if not profile_data:
        return ""
    main = profile_data.get("main_spec", {})
    rels = profile_data.get("relationships", {})

    lines = [
        "\nПРОФИЛЬ КЛИЕНТА:",
        f"  Имя: {main.get('name', 'не указано')}",
        f"  Возраст: {main.get('age', '?')} лет",
        f"  Пол: {main.get('gender', '?')}",
        f"  Работа: {main.get('job_title', '?')}",
        f"  Доход: {main.get('monthly_income', '?')}",
        f"  Город: {main.get('location', 'Россия')}",
    ]
    if main.get("personality_traits"):
        trait_preview = str(main["personality_traits"])[:200].replace("\n", " ")
        lines.append(f"  Портрет: {trait_preview}...")

    if rels:
        lines.append("  Связи:")
        for rel_type, people in rels.items():
            names = ", ".join(p.get("name", "?") for p in people[:3])
            if names:
                lines.append(f"    {rel_type}: {names}")

    return "\n".join(lines)

def generate_graph(
    events: list,
    topic: str,
    difficulty: str,
    task_type: str,
    save_dir: str | None = None,
    profile_data: dict | None = None,
    plan_batches: list | None = None,
) -> dict:
    """Генерирует граф знаний по схеме Skeleton-First (Вариант А).

    Фаза 1 (скелет): LLM генерирует только GT-элементы (~15-20 штук).
                     Валидация GT-структуры + LLM-заземление на малом объёме.
    Фаза 2 (наполнение): LLM добавляет non-GT узлы/рёбра как фоновый контекст.
                          Только механическая валидация (без LLM).
    После сборки: inject_noise() + _inject_episode_backbone() + финальная проверка эпизодов.

    Args:
        plan_batches: список строк батчей из plan.pickle.
        profile_data: содержимое user_profile.json.

    Returns:
        dict с ключами "nodes" и "edges".
    """
    cfg = GRAPH_DIFFICULTY_CONFIG.get(difficulty, GRAPH_DIFFICULTY_CONFIG["easy"])

    num_nodes_target = random.randint(*cfg["num_nodes_range"])

    if plan_batches and isinstance(plan_batches, list) and len(plan_batches) > 0:
        real_batches = [b for b in plan_batches if b != "[SYNTHETIC PADDING BATCH]"]
        events_text = "\n\n".join(
            f"=== ЭПИЗОД {i + 1} ===\n{batch}"
            for i, batch in enumerate(real_batches)
        )
        num_episodes = len(real_batches)
    elif isinstance(events, list):
        events_text = "\n".join(f"- {e}" for e in events)
        num_episodes = len(events)
    else:
        events_text = str(events)
        num_episodes = 1

    profile_block = _format_profile_for_prompt(profile_data) if profile_data else ""

    MAX_SKEL_RETRIES = 3
    MAX_FILL_RETRIES = 5 if difficulty == "hard" else 3
    graph = None

    skel_prompt_template = GRAPH_SKELETON_PROMPTS.get(
        (task_type, difficulty),
        GRAPH_SKELETON_PROMPTS[("information_extraction", "easy")]
    )

    gt_guidance = _get_gt_guidance(task_type, difficulty)

    for skel_attempt in range(MAX_SKEL_RETRIES):
        skel_prompt = (skel_prompt_template
                       .replace("<topic>", topic)
                       .replace("<events>", events_text)
                       .replace("<difficulty>", difficulty))
        if gt_guidance:
            skel_prompt = skel_prompt + "\n" + gt_guidance
        if profile_block:
            skel_prompt = skel_prompt + profile_block

        if skel_attempt > 0:
            skel_prompt += (f"\n\n⚠️ ПОВТОР {skel_attempt+1}: предыдущий скелет не прошёл валидацию. "
                            f"Убедись что все элементы имеют episode_hint и соблюдена структура {task_type}.")

        try:
            raw = llm.invoke(skel_prompt).content
            clean = re.sub(r'```json|```', '', raw).strip()
            skeleton = json.loads(repair_json(clean))
            if not isinstance(skeleton, dict) or ("gt_nodes" not in skeleton and "gt_edges" not in skeleton):
                print(f"  ⚠️ [skel/{skel_attempt+1}] нет gt_nodes/gt_edges")
                continue
        except Exception as e:
            print(f"  ⚠️ [skel/{skel_attempt+1}] ошибка парсинга: {e}")
            continue

        raw_for_val = {
            "nodes": skeleton.get("gt_nodes", []),
            "edges": skeleton.get("gt_edges", []),
        }
        validated_skel_graph = _validate_graph(raw_for_val)
        for n in validated_skel_graph["nodes"]:
            n["ground_truth"] = True
        for edge in validated_skel_graph["edges"]:
            if edge.get("relation") not in _STRUCTURAL_RELATIONS:
                edge["ground_truth"] = True

        _gt_edge_ep = set()
        for _e in validated_skel_graph["edges"]:
            if _e.get("ground_truth") and _e.get("relation") not in _STRUCTURAL_RELATIONS:
                _gt_edge_ep.add(_e.get("source", ""))
                _gt_edge_ep.add(_e.get("target", ""))
        _demoted = []
        for _n in validated_skel_graph["nodes"]:
            if _n.get("ground_truth") and _n.get("id") not in _gt_edge_ep:
                _n["ground_truth"] = False
                _demoted.append(_n.get("id", "?"))
        if _demoted:
            print(f"  ℹ️ [skel/{skel_attempt+1}] GT-узлы без GT-рёбер понижены до non-GT: {_demoted[:5]}")

        skeleton = {
            "gt_nodes": validated_skel_graph["nodes"],
            "gt_edges": validated_skel_graph["edges"],
        }

        ok, reason = _validate_skeleton(skeleton, task_type, difficulty)
        if not ok:
            print(f"  🔄 [skel/{skel_attempt+1}] структура: {reason}")
            continue

        skel_edges_for_grounding = skeleton["gt_edges"]
        if task_type == "composite":
            _date_keys = {"date", "timestamp", "since", "until", "from", "period"}
            skel_edges_for_grounding = [
                {**e, "attributes": {k: v for k, v in e.get("attributes", {}).items()
                                     if k not in _date_keys}}
                if "chain_step" in e.get("attributes", {}) else e
                for e in skeleton["gt_edges"]
            ]
        skel_as_graph = {"nodes": skeleton["gt_nodes"], "edges": skel_edges_for_grounding}
        ok, reason = _llm_validate_graph_grounding(skel_as_graph, events_text)
        if not ok:
            print(f"  🔄 [skel/{skel_attempt+1}] заземление: {reason}")
            if skel_attempt == MAX_SKEL_RETRIES - 1:
                print(f"  ⚠️ [skel] принимаем с предупреждением: {reason}")
            else:
                continue

        print(f"  ✔ [skel/{skel_attempt+1}] скелет принят ({len(skeleton['gt_nodes'])} узлов, "
              f"{len(skeleton['gt_edges'])} рёбер)")

        skeleton_text = _serialize_skeleton_for_fill(skeleton)

        min_nodes = cfg["num_nodes_range"][0]
        min_edges = cfg["min_edges"]
        need_nodes = max(5, num_nodes_target - len(skeleton["gt_nodes"]))
        need_edges = max(10, min_edges - len(skeleton["gt_edges"]) + cfg["fill_edge_extra"])

        fill_prompt = (graph_fill_prompt
                       .replace("<topic>", topic)
                       .replace("<difficulty>", difficulty)
                       .replace("<skeleton_text>", skeleton_text)
                       .replace("<events>", events_text[:8000])
                       .replace("<need_nodes>", str(need_nodes))
                       .replace("<need_edges>", str(need_edges))
                       .replace("<difficulty_rules>", GRAPH_FILL_RULES.get(difficulty, "")))

        best_merged = None  # лучший частичный результат (для кумулятивного fill)

        for fill_attempt in range(MAX_FILL_RETRIES):
            if fill_attempt > 0 and best_merged is not None:
                cur_non_ep = [n for n in best_merged["nodes"] if n.get("type") != "episode"]
                node_deficit = max(1, min_nodes - len(cur_non_ep))
                cur_need_nodes = max(10, node_deficit * 2)
                cur_need_edges = max(15, min_edges - len(best_merged["edges"]) + 40)
                prefix = f"x{fill_attempt+1}_"
                existing_ids_str = ", ".join(
                    n["id"] for n in best_merged["nodes"] if n.get("type") != "episode"
                )[:600]
                fill_prompt_cur = fill_prompt + (
                    f"\n\n⚠️ ПОВТОР {fill_attempt+1}: уже {len(cur_non_ep)} узлов, "
                    f"{len(best_merged['edges'])} рёбер — НЕДОСТАТОЧНО (нужно {min_edges}).\n"
                    f"Добавь ЕЩЁ {cur_need_nodes} АБСОЛЮТНО НОВЫХ узлов и {cur_need_edges} НОВЫХ рёбер.\n"
                    f"ОБЯЗАТЕЛЬНО используй НОВЫЙ ПРЕФИКС для всех id: '{prefix}' "
                    f"(примеры: {prefix}node_1, {prefix}node_2, {prefix}edge_1, {prefix}edge_2).\n"
                    f"ЗАПРЕЩЁННЫЕ id (уже существуют, НЕ повторять): {existing_ids_str}"
                )
            else:
                fill_prompt_cur = fill_prompt

            try:
                raw = llm.invoke(fill_prompt_cur).content
                clean = re.sub(r'```json|```', '', raw).strip()
                fill = json.loads(repair_json(clean))
                if not isinstance(fill, dict) or ("nodes" not in fill and "edges" not in fill):
                    print(f"  ⚠️ [fill/{fill_attempt+1}] нет nodes/edges")
                    continue
                if not isinstance(fill.get("nodes", []), list) or not isinstance(fill.get("edges", []), list):
                    print(f"  ⚠️ [fill/{fill_attempt+1}] nodes или edges не является списком")
                    continue
                if not fill.get("nodes") and not fill.get("edges"):
                    print(f"  ⚠️ [fill/{fill_attempt+1}] пустое наполнение (nodes=[], edges=[])")
                    continue
            except Exception as e:
                print(f"  ⚠️ [fill/{fill_attempt+1}] ошибка парсинга: {e}")
                continue

            if fill_attempt == 0 or best_merged is None:
                fill_safe = {
                    "nodes": [n for n in fill.get("nodes", []) if isinstance(n, dict)],
                    "edges": [edge for edge in fill.get("edges", []) if isinstance(edge, dict)],
                }
                raw_merged = _merge_skeleton_fill(skeleton, fill_safe)
                gt_node_ids = {n["id"] for n in skeleton.get("gt_nodes", [])}
                gt_edge_ids = {edge["id"] for edge in skeleton.get("gt_edges", [])}
                for n in raw_merged["nodes"]:
                    if isinstance(n, dict) and n.get("id") not in gt_node_ids:
                        n["ground_truth"] = False
                for edge in raw_merged["edges"]:
                    if isinstance(edge, dict) and edge.get("id") not in gt_edge_ids:
                        edge["ground_truth"] = False
            else:
                existing_node_ids = {n["id"] for n in best_merged["nodes"] if "id" in n}
                existing_edge_ids = {edge["id"] for edge in best_merged["edges"] if "id" in edge}
                new_nodes = [n for n in fill.get("nodes", [])
                             if isinstance(n, dict) and n.get("id") not in existing_node_ids]
                new_edges = [edge for edge in fill.get("edges", [])
                             if isinstance(edge, dict) and edge.get("id") not in existing_edge_ids]
                for n in new_nodes:
                    n["ground_truth"] = False
                for edge in new_edges:
                    edge["ground_truth"] = False
                raw_merged = {
                    "nodes": best_merged["nodes"] + new_nodes,
                    "edges": best_merged["edges"] + new_edges,
                }

            merged = _validate_graph(raw_merged)  # финальная нормализация на полном множестве узлов
            if merged.get("nodes"):
                best_merged = merged

            ok, reason = _validate_fill(merged, difficulty)
            if not ok:
                print(f"  🔄 [fill/{fill_attempt+1}] {reason}")
                continue

            print(f"  ✔ [fill/{fill_attempt+1}] наполнение принято ({len(merged['nodes'])} узлов, "
                  f"{len(merged['edges'])} рёбер)")
            graph = merged
            break

        if graph is not None:
            break  # скелет + fill приняты

        print(f"  🔄 [skel/{skel_attempt+1}] fill не удался — пробуем новый скелет")

    if graph is None:
        raise RuntimeError(
            f"[graph_gen] Не удалось построить граф за {MAX_SKEL_RETRIES} попыток скелета "
            f"({task_type}/{difficulty})"
        )

    graph = inject_noise(graph, difficulty)
    _connected_after_noise = set()
    for _e in graph["edges"]:
        _connected_after_noise.add(_e["source"])
        _connected_after_noise.add(_e["target"])
    graph["nodes"] = [
        n for n in graph["nodes"]
        if n.get("type") == "episode" or n.get("ground_truth") or n["id"] in _connected_after_noise
    ]
    graph = _inject_episode_backbone(graph, num_episodes)

    ok_q, reason_q = _check_graph_quality(graph, difficulty, task_type)
    if not ok_q:
        print(f"  ⚠️ [quality] {reason_q} (принимаем, но граф может быть низкого качества)")

    ok, reason = _check_cross_episode_semantics(graph, task_type, num_episodes)
    if not ok:
        print(f"  ⚠️ [cross_ep] {reason} — ремаппинг...")
        _remap_episode_assignments(graph, task_type, num_episodes)
        ok2, reason2 = _check_cross_episode_semantics(graph, task_type, num_episodes)
        if not ok2:
            print(f"  ⚠️ [cross_ep] после ремаппинга: {reason2} (принимаем)")
        else:
            print(f"  ✔ [cross_ep] ремаппинг успешен")

    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    n_gt_nodes = sum(1 for n in graph["nodes"] if n.get("ground_truth"))
    n_gt_edges = sum(1 for e in graph["edges"] if e.get("ground_truth"))
    n_stale_edges = sum(1 for e in graph["edges"] if e.get("stale"))
    n_invalid_edges = sum(1 for e in graph["edges"] if not e.get("valid", True))
    serialized = serialize_graph_for_model(graph, difficulty, show_ground_truth=False, for_task_generation=False)
    n_tokens = get_token_number(serialized)

    stats = {
        "difficulty": difficulty,
        "task_type": task_type,
        "nodes": n_nodes,
        "nodes_gt": n_gt_nodes,
        "edges": n_edges,
        "edges_gt": n_gt_edges,
        "edges_stale": n_stale_edges,
        "edges_invalid": n_invalid_edges,
        "tokens_test_input": n_tokens,
    }

    print(f"  📊 Граф [{difficulty}/{task_type}]: "
          f"{n_nodes} узлов (GT={n_gt_nodes}), "
          f"{n_edges} рёбер (GT={n_gt_edges}, stale={n_stale_edges}, invalid={n_invalid_edges}), "
          f"~{n_tokens} токенов")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "graph.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)

        stats_path = os.path.join(save_dir, "graph_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Граф сохранён: {save_path}")

    return graph