# BenchEpisodic

Генератор синтетических банковских диалогов для оценки эпизодической памяти LLM. Пайплайн создаёт многоходовые чаты клиент ↔ банк на русском языке, потом генерирует QA-задачи по ним и гоняет на них модели.

Есть два независимых пайплайна:
- **dialogue** — чат + QA-задачи по тексту диалога
- **graph** — граф знаний из того же плана + задачи по навигации в графе

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai tiktoken faker networkx numpy json_repair tqdm python-dotenv
# опционально для метрик:
pip install rouge-score bert-score
```

Создай `.env` рядом с кодом:

```
OPENAI_API_KEY=ваш_ключ
OPENAI_BASE_URL=ваш_endpoint
MODEL_NAME=gpt-4o-mini          # модель генерации, по умолчанию gpt-4o-mini
VALIDATOR_MODEL=gpt-4o          # модель валидации QA-задач, по умолчанию gpt-4o
```

---

## Уровни сложности

| Уровень | Токены | Батчи | Шум | Дистракторы | Приманки |
|---------|--------|-------|-----|-------------|----------|
| easy    | 20–25K | 3     | 0%  | 0%          | 0%       |
| medium  | 50–60K | 5     | 20% | 20%         | 15%      |
| hard    | 90–100K| 8     | 40% | 40%         | 35%      |

---

## Диалоговый пайплайн

### Запустить всё сразу

```bash
# с автогенерацией тем
python run_pipeline.py --stage all --difficulty easy --count 10

# или с готовым файлом тем
python run_pipeline.py --stage all --difficulty easy --topics topics.json
```

Результаты будут в `data/results/dialogue/easy/`.

### Запускать по стадиям

Каждая стадия идемпотентна — пропускает директории, где уже есть выходной файл.

```bash
# Стадия 0: генерация тем (обязательно указывать --difficulty)
python run_pipeline.py --stage topics --difficulty easy --count 10

# или отдельно через generate_topics.py с доп. опциями
python generate_topics.py --count 10 --difficulty easy
python generate_topics.py --count 5 --difficulty medium --categories lending security
python generate_topics.py --list-categories   # посмотреть доступные категории

# Стадия 1: планы батчей
python run_pipeline.py --stage plan --difficulty easy --topics topics.json

# Стадия 2: сообщения пользователя
python run_pipeline.py --stage messages

# Стадия 3: ответы оператора → собирается полный диалог
python run_pipeline.py --stage answers --difficulty easy

# Стадия 3.5: обрезка/добивка до нужного диапазона токенов
python run_pipeline.py --stage truncate --difficulty easy

# Стадия 4: генерация QA-задач
python run_pipeline.py --stage tasks --difficulty easy

# Стадия 5: оценка моделей
python run_pipeline.py --stage evaluate --difficulty easy --models gpt-4o gpt-4o-mini
```

### Как работает пайплайн

```
topics → plan → messages → answers → truncate → tasks → evaluate
```

**topics** — для каждого чата генерируется тема с типом задачи (`information_extraction`, `knowledge_update`, `temporal_reasoning`, `interference`, `composite`). Профиль клиента создаётся сразу — до событий, чтобы имена были консистентны.

**plan** — LLM составляет план диалога по батчам. В каждом батче конкретные суммы, даты, ставки. Для medium/hard план намеренно включает шум (`[SEMANTIC NOISE]`), отвлекающие темы (`[DISTRACTOR]`) и факты-двойники (`[DECOY]`).

**messages** — по плану генерируются реалистичные сообщения клиента. Работает параллельно (до 2 воркеров на батч).

**answers** — банковский оператор отвечает на каждое сообщение. История сжимается через `ConversationSummaryBuffer` когда становится слишком длинной. Каждый ответ проходит rule-based + LLM-валидацию.

**truncate** — диалог обрезается или добивается до нужного диапазона токенов. Чаты hard никогда не обрезаются, только добиваются.

**tasks** — по каждому диалогу генерируется 1 QA-задача. Пятиступенчатая валидация: структура → заземлённость → нетривиальность → LLM-судья → adversarial no-context check (medium/hard).

**evaluate** — модели отвечают в формате `{"answer": "..."}`. Итоговая метрика — Final Accuracy: если EM=1, то 1.0; иначе LLM-Judge (0.0/0.5/1.0). Дополнительно считаются EM, Token F1, BERTScore.

---

## Графовый пайплайн

```bash
# всё сразу
python run_graph_pipeline.py --stage all --difficulty easy --count 10

# по стадиям
python run_graph_pipeline.py --stage plan --difficulty easy
python run_graph_pipeline.py --stage graph_gen --difficulty easy
python run_graph_pipeline.py --stage graph_tasks --difficulty easy
python run_graph_pipeline.py --stage evaluate --difficulty easy --models gpt-4o
```

Темы и планы генерируются с нуля независимо от диалогового пайплайна.

### Как работает

```
topics → plan → graph_gen → graph_tasks → evaluate
```

**graph_gen** — из плана строится граф знаний: узлы (`person`, `account`, `event`, `amount`, `rate`, ...) и рёбра (`owns`, `transferred_to`, `changed_to`, `caused`, ...). Для medium/hard добавляются stale-рёбра, невалидные рёбра и дубликаты узлов.

**graph_tasks** — генерируются задачи по навигации в графе. Модели при оценке получают граф в текстовом виде, но с ограничениями: medium скрывает `occurred_in`, hard скрывает ещё и `followed_by`.

**evaluate** — то же что в диалоговом, плюс для composite-задач считается `path_accuracy`.

---

## Структура файлов

```
data/results/dialogue/{difficulty}/{task_type}/{id}/
    topic.json            — тема + тип задачи + события
    plan.pickle           — план по батчам
    user_profile.json     — профиль клиента
    chat_truncated.json   — финальный диалог
    tasks.json            — QA-задачи

data/results/graph/{difficulty}/{task_type}/{id}/
    graph.json            — граф знаний
    tasks_graph.json      — задачи по графу

data/results/dialogue/{difficulty}/eval_results/
    results_{model}.json
    evaluation_report.json
```

---

## Добавить модель для оценки

В `model_config.py` добавь запись в `MODEL_REGISTRY`:

```python
"your-model-name": {
    "api_key_env": "YOUR_API_KEY",
    "base_url_env": "YOUR_BASE_URL",
    "model_name": "your-model-name",
    "temperature": 0.0,
},
```

И добавь соответствующие переменные в `.env`.

---

## Типы задач

| Тип | Что проверяет |
|-----|---------------|
| `information_extraction` | найти конкретный факт (сумму, дату, ставку) |
| `knowledge_update` | отследить изменение значения (A→B→C) |
| `temporal_reasoning` | вычислить когда / сколько времени прошло |
| `interference` | различить похожие факты (разница ≤200 руб. / 1 день) |
| `composite` | многошаговый вывод через цепочку событий |

---

## Результаты оценки

**Генерация датасета:** Qwen и Gemini 2.5 Flash  
**Валидация QA-задач:** GPT-5-mini и Gemini 2.0 Pro  
**Всего задач: 309** (диалог: 156, граф: 153)  
**Метрика:** Final Accuracy — если EM=1, то 1.0; иначе LLM-судья (0.0/0.5/1.0)

### Диалоговый пайплайн

| Модель     | Easy (59) | Medium (47) | Hard (50) | Среднее |
|------------|-----------|-------------|-----------|---------|
| Kimi K2    | **81.4%** | **76.6%**   | **60.0%** | **72.7%** |
| GLM-5.1    | 81.4%     | 70.2%       | 58.0%     | 69.9%   |
| GPT-5-mini | 75.4%     | 73.4%       | 51.0%     | 66.6%   |

По типам задач (Final Accuracy, все уровни вместе):

| Тип задачи            | Kimi K2 | GLM-5.1 | GPT-5-mini |
|-----------------------|---------|---------|------------|
| information_extraction| 83.3%   | 83.3%   | 75.8%      |
| knowledge_update      | 67.2%   | 67.2%   | 67.2%      |
| temporal_reasoning    | 73.8%   | 72.1%   | 73.8%      |
| interference          | 80.0%   | 75.0%   | 74.2%      |
| composite             | 78.1%   | 62.5%   | 71.9%      |

### Графовый пайплайн

| Модель     | Easy (50) | Medium (50) | Hard (53) | Среднее |
|------------|-----------|-------------|-----------|---------|
| GPT-5-mini | **96.0%** | 88.0%       | 51.9%     | 78.6%   |
| Kimi K2    | 94.0%     | **90.0%**   | **54.7%** | **79.6%** |
| GLM-5.1    | 92.0%     | 86.0%       | 41.5%     | 73.2%   |

### Confidence Intervals (bootstrap, 95%, n=10 000)

Общий итог по всем 309 задачам:

| Модель     | Mean  | 95% CI        |
|------------|-------|---------------|
| Kimi K2    | 76.1% | 71.2% – 80.9% |
| GPT-5-mini | 72.5% | 67.5% – 77.3% |
| GLM-5.1    | 71.5% | 66.3% – 76.4% |

По уровню сложности:

| Уровень | Модель     | Mean  | 95% CI        |
|---------|------------|-------|---------------|
| Easy    | Kimi K2    | 87.2% | 80.7% – 92.7% |
| Easy    | GLM-5.1    | 86.2% | 79.8% – 92.7% |
| Easy    | GPT-5-mini | 84.9% | 78.0% – 90.8% |
| Medium  | Kimi K2    | 83.5% | 76.3% – 90.7% |
| Medium  | GPT-5-mini | 80.9% | 72.7% – 88.7% |
| Medium  | GLM-5.1    | 78.4% | 70.1% – 86.6% |
| Hard    | Kimi K2    | 57.3% | 47.6% – 67.0% |
| Hard    | GPT-5-mini | 51.5% | 42.2% – 61.2% |
| Hard    | GLM-5.1    | 49.5% | 39.8% – 59.2% |

На easy/medium интервалы широко перекрываются — разница незначима. На hard Kimi отрывается заметнее. Подробные CI по пайплайнам, типам задач и каждой ячейке type×difficulty — в [RESULTS.md](RESULTS.md).

### Главные выводы

1. **Kimi K2 — лучший в целом**, особенно на диалоге и hard.
2. **GPT-5-mini** лидирует на лёгком графе, но отстаёт на тяжёлом диалоге.
3. **GLM-5.1** держится на easy, заметно теряет на hard графе (`knowledge_update` hard — 20%).
4. **`knowledge_update`** — самый сложный тип для всех трёх моделей.
5. **Граф легче диалога** на easy/medium (~10–15 п.п.), на hard разрыв закрывается.
6. **EM < Final Accuracy** везде — модели отвечают правильно по смыслу, но формат часто не совпадает дословно.
