import os
import re
import json
from datetime import datetime
from json_repair import repair_json

_TEMPORAL_UNIT = r'(дней|день|дня|часов|час|часа|минут|минута|минуты|недель|неделя|недели|месяцев|месяц|месяца|лет|год|года)'
_TEMPORAL_ANSWER_PATTERN = re.compile(
    r'^\d+\s+' + _TEMPORAL_UNIT + r'(\s+\d+\s+' + _TEMPORAL_UNIT + r')?$',
    re.IGNORECASE,
)

def _extract_gt_dates(graph: dict) -> list[datetime]:
    dates = []
    for e in graph.get("edges", []):
        if not e.get("ground_truth"):
            continue
        for v in e.get("attributes", {}).values():
            if isinstance(v, str):
                m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', v)
                if m:
                    try:
                        d = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                        dates.append(d)
                    except ValueError:
                        pass
    return sorted(set(dates))

def _temporal_answer_matches_gt_dates(answer: str, graph: dict) -> tuple[bool, str]:
    m = re.match(r'^(\d+)\s+(\S+)$', answer.strip(), re.IGNORECASE)
    if not m:
        return False, "не удалось распарсить число из ответа"

    value = int(m.group(1))
    unit = m.group(2).lower()

    dates = _extract_gt_dates(graph)
    if len(dates) < 2:
        return False, f"в GT только {len(dates)} дат для вычисления"

    if unit in ("дней", "день", "дня"):
        target_days = value
        tolerance = 1
    elif unit in ("недель", "неделя", "недели"):
        target_days = value * 7
        tolerance = 2
    elif unit in ("месяцев", "месяц", "месяца"):
        target_days = value * 30
        tolerance = 5  # месяцы неоднозначны (28-31 день)
    elif unit in ("лет", "год", "года"):
        target_days = value * 365
        tolerance = 10
    else:
        return True, f"единица '{unit}' — пропускаем дневную проверку"

    for i, d1 in enumerate(dates):
        for d2 in dates[i+1:]:
            diff = abs((d2 - d1).days)
            if abs(diff - target_days) <= tolerance:
                return True, (f"ответ {value} {unit} ≈ {diff} дн. "
                              f"между {d1.strftime('%d.%m.%Y')} и {d2.strftime('%d.%m.%Y')}")

    diffs = sorted({abs((dates[j] - dates[i]).days)
                    for i in range(len(dates)) for j in range(i+1, len(dates))})
    return False, (f"ответ {value} {unit} (≈{target_days} дн.) не соответствует "
                   f"ни одной разности GT дат (ближайшие: {diffs[:8]})")

from llm import gemini_base as llm, validator_llm
from graph_prompts import GRAPH_TASK_PROMPTS, graph_task_validation_prompt
from graph_generator import serialize_graph_for_model

def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    a_tokens = set(text_a.lower().split())
    b_tokens = set(text_b.lower().split())
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)

_CALC_STEP_PATTERN = re.compile(
    r'вычислени[еяю]\s*[:：]\s*'
    r'(\d{2}\.\d{2}\.\d{4})\s*[−\-–—]\s*(\d{2}\.\d{2}\.\d{4})\s*=\s*(\d+)\s*(\S+)',
    re.IGNORECASE,
)

def _verify_temporal_arithmetic(task: dict) -> tuple[bool, str]:
    path = task.get("reasoning_path", [])
    for step in path:
        m = _CALC_STEP_PATTERN.search(str(step))
        if not m:
            continue
        d1_str, d2_str, n_str, unit = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            d1 = datetime(int(d1_str[6:]), int(d1_str[3:5]), int(d1_str[:2]))
            d2 = datetime(int(d2_str[6:]), int(d2_str[3:5]), int(d2_str[:2]))
        except ValueError:
            continue
        claimed = int(n_str)
        actual = abs((d1 - d2).days)

        unit_lower = unit.lower()
        if unit_lower in ("дней", "день", "дня"):
            if abs(actual - claimed) <= 1:  # допуск ±1 для граничных случаев подсчёта
                return True, f"арифметика верна: |{actual} - {claimed}| ≤ 1"
            return False, (f"арифметическая ошибка: {d1_str} − {d2_str} = "
                           f"{actual} дней, в задаче указано {claimed}")
        elif unit_lower in ("недель", "неделя", "недели"):
            actual_weeks = actual // 7
            if abs(actual_weeks - claimed) <= 1:
                return True, f"арифметика недель верна: {actual_weeks} ≈ {claimed}"
            return False, f"недели: {actual} дней = {actual_weeks} нед., указано {claimed}"
        elif unit_lower in ("месяцев", "месяц", "месяца"):
            actual_months = actual // 30
            if abs(actual_months - claimed) <= 1:
                return True, f"арифметика месяцев верна: {actual_months} ≈ {claimed}"
            return False, f"месяцы: {actual} дней ≈ {actual_months} мес., указано {claimed}"
    return True, "нет шагов с явной датой-арифметикой"

def _answer_in_graph(answer: str, graph: dict, gt_only: bool = False,
                     non_stale_only: bool = False) -> bool:
    answer_lower = answer.lower().strip()
    if not answer_lower or len(answer_lower) < 2:
        return False

    _is_numeric = bool(re.search(r'\d', answer_lower))
    if _is_numeric:
        _pattern = re.compile(r'(?<!\d)' + re.escape(answer_lower) + r'(?!\d)')
    else:
        _pattern = None

    def _match(text: str) -> bool:
        if _pattern:
            return bool(_pattern.search(text))
        return answer_lower in text

    for node in graph.get("nodes", []):
        if gt_only and not node.get("ground_truth"):
            continue
        attrs = node.get("attributes", {})
        label = node.get("label", "")
        node_text = (label + " " + json.dumps(attrs, ensure_ascii=False)).lower()
        if _match(node_text):
            return True

    for edge in graph.get("edges", []):
        if gt_only and not edge.get("ground_truth"):
            continue
        if non_stale_only and edge.get("stale"):
            continue
        attrs = edge.get("attributes", {})
        relation = edge.get("relation", "")
        edge_text = (relation + " " + json.dumps(attrs, ensure_ascii=False)).lower()
        if _match(edge_text):
            return True

    return False

def _structural_check(task: dict, difficulty: str = "easy") -> tuple[bool, str]:
    question = task.get("question", "")
    answer = task.get("answer", "")

    if not question or len(str(question).strip()) < 10:
        return False, f"вопрос слишком короткий ({len(str(question).strip())} симв.)"
    if not answer or len(str(answer).strip()) < 1:
        return False, "ответ пустой"
    if difficulty in ("medium", "hard") and not task.get("reasoning_path"):
        return False, "отсутствует reasoning_path"
    if not task.get("source_node_ids"):
        return False, "отсутствует source_node_ids"
    decoys = task.get("decoy_answers", [])
    if not isinstance(decoys, list) or len(decoys) < 3:
        n = len(decoys) if isinstance(decoys, list) else 0
        return False, f"требуется 3 decoy_answers для MCQ, найдено {n}"
    return True, "OK"

def _grounding_check(answer: str, graph: dict, graph_text: str,
                     task_type: str = "") -> tuple[bool, str]:
    if task_type == "temporal_reasoning" and _TEMPORAL_ANSWER_PATTERN.match(answer.strip()):
        if _answer_in_graph(answer, graph, gt_only=True):
            return True, f"temporal: '{answer}' найден в GT атрибутах"
        if _answer_in_graph(answer, graph, gt_only=False):
            return False, f"temporal: '{answer}' найден только в non-GT данных — использует шум"
        match_ok, match_reason = _temporal_answer_matches_gt_dates(answer, graph)
        if match_ok:
            return True, f"temporal computed interval: {match_reason}"
        return False, f"temporal computed: {match_reason}"

    ku_strict = (task_type == "knowledge_update")

    if _answer_in_graph(answer, graph, gt_only=True, non_stale_only=ku_strict):
        return True, f"found in GT{' non-stale' if ku_strict else ''} graph attributes"

    if ku_strict and _answer_in_graph(answer, graph, gt_only=True, non_stale_only=False):
        return False, f"knowledge_update: ответ '{answer}' найден только в stale GT-рёбрах — устаревшее значение"

    gt_edges_filter = [e for e in graph.get("edges", [])
                       if e.get("ground_truth") and (not ku_strict or not e.get("stale"))]
    gt_raw = json.dumps(
        {"nodes": [n for n in graph.get("nodes", []) if n.get("ground_truth")],
         "edges": gt_edges_filter},
        ensure_ascii=False
    ).replace(" ", "")
    numbers = re.findall(r'\d{2,}', answer.replace(" ", ""))
    if numbers and all(n in gt_raw for n in numbers):
        return True, f"числовая GT-проверка: {numbers} найдены в GT"

    if task_type in ("information_extraction", "composite") and numbers:
        gt_numbers = set(re.findall(r'\d{2,}', gt_raw))
        answer_val = int("".join(numbers)) if len(numbers) == 1 else None
        gt_ints = []
        for gn in gt_numbers:
            try:
                gt_ints.append(int(gn))
            except ValueError:
                pass
        if answer_val is not None and gt_ints:
            for a in gt_ints:
                for b in gt_ints:
                    if a != b and (a - b == answer_val or b - a == answer_val or a + b == answer_val):
                        return True, f"вычисленный ответ: {answer_val} = арифметика из GT ({a}, {b})"

    if len(answer.strip().split()) <= 2:
        return False, f"ответ '{answer}' не найден в GT-атрибутах графа"

    gt_nodes_text = " ".join(
        (n.get("label", "") + " " + json.dumps(n.get("attributes", {}), ensure_ascii=False))
        for n in graph.get("nodes", []) if n.get("ground_truth")
    )
    gt_edges_text = " ".join(
        (e.get("relation", "") + " " + json.dumps(e.get("attributes", {}), ensure_ascii=False))
        for e in gt_edges_filter
    )
    gt_text = gt_nodes_text + " " + gt_edges_text
    overlap = _token_overlap_ratio(answer, gt_text)
    if overlap >= 0.3:
        return True, f"GT token overlap {overlap:.0%}"

    return False, f"ответ '{answer}' не найден в атрибутах графа"

def _decoy_grounding_check_graph(task: dict, graph: dict,
                                 task_type: str = "") -> tuple[bool, str]:
    decoys = task.get("decoy_answers", [])
    if not decoys or not isinstance(decoys, list):
        return True, "нет decoy_answers"

    if not task_type:
        task_type = task.get("capability", "") or task.get("task_type", "")

    if task_type == "temporal_reasoning":
        graph_text = json.dumps(graph, ensure_ascii=False)
        if re.search(r'\d{2}\.\d{2}\.\d{4}', graph_text):
            return True, "temporal_reasoning: даты в графе есть, decoy-интервалы валидны"
        return False, "temporal_reasoning: в графе нет дат ДД.ММ.ГГГГ для вычисления интервалов"

    graph_raw = json.dumps(graph, ensure_ascii=False).lower()
    graph_no_spaces = graph_raw.replace(" ", "")

    if task_type == "composite":
        ungrounded = []
        all_labels = set()
        for node in graph.get("nodes", []):
            lbl = str(node.get("label", "")).lower()
            if lbl:
                all_labels.add(lbl)
        for edge in graph.get("edges", []):
            for attr_val in edge.get("attributes", {}).values():
                s = str(attr_val).lower()
                if len(s) > 4:
                    all_labels.add(s)

        for decoy in decoys:
            decoy_str = str(decoy).strip()
            if not decoy_str:
                continue
            segments = [s.strip().lower() for s in re.split(r'→|->|—>|\|', decoy_str) if s.strip()]
            matched = sum(
                1 for seg in segments
                if any(seg in lbl or lbl in seg for lbl in all_labels if len(lbl) > 4)
                or seg in graph_raw
            )
            if matched >= max(1, len(segments) // 2):
                continue
            numbers = re.findall(r'\d{3,}', decoy_str.replace(" ", ""))
            if numbers and any(
                re.search(r'(?<!\d)' + n + r'(?!\d)', graph_no_spaces)
                for n in numbers
            ):
                continue
            ungrounded.append(decoy_str[:40])

        if ungrounded:
            return False, f"Decoy не найдены в графе: {ungrounded[:2]}"

        decoy_strs = [str(d).strip().lower() for d in decoys if str(d).strip()]
        if len(decoy_strs) != len(set(decoy_strs)):
            dupes = [d for d in decoy_strs if decoy_strs.count(d) > 1]
            return False, f"Дублирующиеся decoy_answers: {list(set(dupes))[:2]}"
        return True, "все decoy заземлены (composite path check)"

    ungrounded = []

    for decoy in decoys:
        decoy_str = str(decoy).strip()
        if not decoy_str:
            continue
        if _answer_in_graph(decoy_str, graph):
            continue
        numbers = re.findall(r'\d{3,}', decoy_str.replace(" ", ""))
        if numbers and any(
            re.search(r'(?<!\d)' + n + r'(?!\d)', graph_no_spaces)
            for n in numbers
        ):
            continue
        ungrounded.append(decoy_str[:40])

    if ungrounded:
        return False, f"Decoy не найдены в графе: {ungrounded[:2]}"

    decoy_strs = [str(d).strip().lower() for d in decoys if str(d).strip()]
    if len(decoy_strs) != len(set(decoy_strs)):
        dupes = [d for d in decoy_strs if decoy_strs.count(d) > 1]
        return False, f"Дублирующиеся decoy_answers: {list(set(dupes))[:2]}"

    return True, "все decoy заземлены"

def _non_triviality_check(task: dict, task_type: str, difficulty: str = "easy") -> tuple[bool, str]:
    question = str(task.get("question", ""))
    answer = str(task.get("answer", ""))
    q_lower = question.lower()
    a_lower = answer.lower()

    if len(a_lower) > 3 and a_lower in q_lower:
        return False, "ответ полностью содержится в вопросе"

    if len(question.split()) < 4:
        return False, f"вопрос слишком короткий ({len(question.split())} слов)"

    max_words = (50 if difficulty == "hard" else 30) if task_type == "composite" else 10
    if len(answer.split()) > max_words:
        return False, f"ответ слишком длинный ({len(answer.split())} слов, max={max_words})"

    if difficulty == "hard" and task_type in ("composite", "temporal_reasoning"):
        path = task.get("reasoning_path", [])
        n_steps = len(path) if isinstance(path, list) else 0
        if n_steps < 3:
            return False, f"hard {task_type} требует ≥3 шагов в reasoning_path, найдено {n_steps}"

    return True, "OK"

def _extract_gt_text(graph_text: str, max_chars: int = 8000) -> str:
    if len(graph_text) <= max_chars:
        return graph_text
    lines = graph_text.splitlines(keepends=True)
    gt_lines = [l for l in lines if "[★" in l]
    other_lines = [l for l in lines if "[★" not in l]
    gt_block = "".join(gt_lines)
    other_block = "".join(other_lines)
    remaining = max_chars - len(gt_block)
    if remaining > 0:
        return gt_block + other_block[:remaining]
    return gt_block[:max_chars]

def _llm_validate(question: str, answer: str, graph_text: str,
                  task_type: str, difficulty: str) -> tuple[bool, str]:
    prompt = (graph_task_validation_prompt
              .replace("<graph_text>", _extract_gt_text(graph_text, max_chars=8000))
              .replace("<question>", question)
              .replace("<answer>", answer)
              .replace("<task_type>", task_type)
              .replace("<difficulty>", difficulty))
    try:
        response = validator_llm.invoke(prompt).content
        clean = re.sub(r'```json|```', '', response).strip()
        result = json.loads(repair_json(clean))
        verdict = result.get("verdict", "FAIL")
        reason = result.get("reason", "нет пояснения")
        if verdict == "PASS" and result.get("grounded") and result.get("answerable"):
            if "difficulty_fit" in result and not result.get("difficulty_fit"):  # поле опциональное
                return False, f"difficulty_fit=false — {reason}"
            return True, reason
        return False, f"судья: {reason}"
    except Exception as e:
        return False, f"ошибка вызова судьи (fallback FAIL): {e}"

def validate_graph_task(task: dict, graph: dict, graph_text: str,
                        task_type: str, difficulty: str) -> tuple[bool, str]:
    question = str(task.get("question", "")).strip()
    answer = str(task.get("answer", "")).strip()

    ok, reason = _structural_check(task, difficulty)
    if not ok:
        return False, f"[STRUCT] {reason}"

    ok, reason = _grounding_check(answer, graph, graph_text, task_type=task_type)
    if not ok:
        return False, f"[GROUND] {reason}"

    if task_type == "temporal_reasoning":
        ok, reason = _verify_temporal_arithmetic(task)
        if not ok:
            return False, f"[ARITH] {reason}"

    ok, reason = _decoy_grounding_check_graph(task, graph, task_type=task_type)
    if not ok:
        return False, f"[DECOY] {reason}"

    ok, reason = _non_triviality_check(task, task_type, difficulty)
    if not ok:
        return False, f"[TRIVIAL] {reason}"

    if len(answer.strip()) < 2:
        return False, "[LLM-SKIP] ответ слишком короткий"

    gt_text = " ".join(
        json.dumps(n.get("attributes", {}), ensure_ascii=False)
        for n in graph.get("nodes", []) if n.get("ground_truth")
    ) + " ".join(
        json.dumps(e.get("attributes", {}), ensure_ascii=False)
        for e in graph.get("edges", []) if e.get("ground_truth")
    )
    skip_llm = answer.lower() in gt_text.lower() and len(question.split()) >= 5

    if not skip_llm:
        ok, reason = _llm_validate(question, answer, graph_text, task_type, difficulty)
        if not ok:
            return False, f"[LLM] {reason}"

    return True, "PASS"

def generate_graph_probing_task(
    graph_path: str,
    topic_path: str,
    save_dir: str,
    difficulty: str,
    task_type: str,
) -> dict:
    print(f"--- Генерация задачи по графу | difficulty={difficulty} | task_type={task_type} ---")

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Ошибка чтения {graph_path}: {e}")
        return {task_type: []}

    if not graph.get("nodes") and not graph.get("edges"):
        print(f"⚠️ Граф пустой: {graph_path}")
        empty_result: dict[str, list] = {task_type: []}
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "tasks_graph.json"), "w", encoding="utf-8") as f:
            json.dump(empty_result, f, indent=2, ensure_ascii=False)
        return empty_result

    topic = "Банковский сценарий"
    if os.path.exists(topic_path):
        try:
            with open(topic_path, "r", encoding="utf-8") as f:
                topic_data = json.load(f)
            topic = topic_data.get("topic", topic_data.get("title", topic))
        except Exception as e:
            print(f"⚠️ Ошибка чтения {topic_path}: {e} — используем дефолтный топик")

    graph_text = serialize_graph_for_model(
        graph, difficulty, show_ground_truth=True, for_task_generation=True
    )

    prompt_key = (task_type, difficulty)
    prompt_template = GRAPH_TASK_PROMPTS.get(prompt_key)
    if not prompt_template:
        print(f"⚠️ Промпт не найден для ({task_type}, {difficulty}), используем IE/easy")
        from graph_prompts import graph_task_information_extraction_easy
        prompt_template = graph_task_information_extraction_easy

    prompt = (prompt_template
              .replace("<graph_text>", graph_text)
              .replace("<topic>", topic)
              .replace("<difficulty>", difficulty))

    MAX_RETRIES = 6 if difficulty == "hard" else 4  # hard: жёсткие ограничения чаще требуют retry
    accepted_task = None

    for attempt in range(MAX_RETRIES):
        try:
            raw = llm.invoke(prompt).content
            clean = re.sub(r'```json|```', '', raw).strip()
            task_json = json.loads(repair_json(clean))

            if isinstance(task_json, list):
                task_json = next(
                    (t for t in task_json if isinstance(t, dict)
                     and "question" in t and "answer" in t),
                    task_json[0] if task_json else {}
                )
            if not isinstance(task_json, dict):
                raise ValueError(f"Ожидался dict, получен {type(task_json)}")
            if "question" not in task_json or "answer" not in task_json:
                raise ValueError(f"Нет обязательных полей: {list(task_json.keys())}")

        except Exception as e:
            print(f"⚠️ [graph_tasks/{task_type}/{difficulty}] Ошибка генерации (попытка {attempt+1}): {e}")
            continue

        is_valid, reason = validate_graph_task(task_json, graph, graph_text, task_type, difficulty)
        if is_valid:
            task_json["difficulty"] = difficulty
            task_json["capability"] = task_type
            accepted_task = task_json
            print(f"  ✅ [graph_tasks/{task_type}/{difficulty}] Задача принята.")
            break
        else:
            if attempt < MAX_RETRIES - 1:
                print(f"  🔄 [graph_tasks/{task_type}/{difficulty}] Валидация: {reason}, перегенерация...")
            else:
                print(f"  👻 [graph_tasks/{task_type}/{difficulty}] DROP после {MAX_RETRIES} попыток: {reason}")

    result = {task_type: [accepted_task] if accepted_task else []}

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tasks_graph.json")
    n = len(result.get(task_type, []))

    if n == 0:
        failed_result = {task_type: [], "failed": True, "reason": f"0 задач после {MAX_RETRIES} попыток"}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(failed_result, f, indent=2, ensure_ascii=False)
        print(f"⚠️ WARNING: {save_dir} — 0 задач. tasks_graph.json сохранён с failed=true.")
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  📄 tasks_graph.json сохранён: {save_path} ({n} задача)")

    return result
