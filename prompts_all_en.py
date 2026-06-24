# prompts for dialogue and graph pipelines


# DIALOGUE PIPELINE

# labels generation

label_generation_prompt_template = """
You are a specialist in financial dialogue structure. Your task is to identify the most suitable label categories for sections of a specific banking scenario.

INPUT PARAMETERS:
• TOPIC: <topic>
• THEME: <theme>

════════════════ MAIN GOAL ═════════════════
Analyze the TOPIC and THEME in the context of banking customer service. Generate 15–20 label categories that best fit this financial context.

════════════════ LABEL CATEGORY SELECTION ═════════════════

**UNIVERSAL CATEGORIES (Always include 2–3):**
- **Client Identification Labels** (passport data, code word, biometrics)
- **Emotional State Labels** (client is anxious, in a hurry, satisfied)
- **Decision & Action Labels** (agreement to service, refusal, signing)

**SPECIFIC FINANCIAL CATEGORIES (Choose 9–13 based on relevance):**

**Lending & Credit Labels** — For lending scenarios:
- Mortgage, car loans, personal loans
- Rates, terms, monthly payments, DTI (debt-to-income ratio)
- Approval, rejection, restructuring, early repayment

**Daily Banking Labels** — For everyday operations:
- Transfers (SBP, by card number, SWIFT)
- Bill payments (utilities, mobile, fines)
- Limits, fees, statements, certificates

**Cards & Accounts Labels** — For cards and accounts:
- Debit/credit cards, cashback, loyalty programs
- Block/unblock, reissue, PIN
- Checking accounts, escrow accounts, nominee accounts

**Deposits & Savings Labels** — For deposits and savings:
- Deposits, savings accounts, interest capitalization
- Withdrawal/top-up conditions, rollover
- Deposit insurance (DIA)

**Investments & Brokerage Labels** — For investments:
- IIS, brokerage accounts, buying stocks/bonds
- Investment taxes, dividends, coupons
- Risk profile, strategies, margin trading

**Security & Fraud Labels** — For security:
- Social engineering, phishing, fraudulent charges
- 115-FZ (account freeze by law), suspicious operations
- Transaction disputes (chargeback)

**Document Management Labels** — For documents:
- Income certificates, USRRE extracts, employment records
- Contracts, applications, powers of attorney
- Personal data updates

**Technical Support Labels** — For tech support:
- Mobile app, online banking, login errors
- Outages, push notifications, biometrics

════════════════ SELECTION CRITERIA ═════════════════

**Must include:**
- Categories directly related to the topic (e.g., if topic is "Mortgage", include Lending, Documents, Insurance).
- Categories that create realistic "noise" (Technical Support, Daily Banking).

════════════════ OUTPUT FORMAT ═════════════════

Generate exactly 15–20 categories in the following format (in English for compatibility, description in Russian):

**[Category Name] Labels:**
- Brief explanation of why this is important for the TOPIC.
- 3–5 example labels (can be in English or transliteration).

Example:
**Lending & Credit Labels:**
- Critical for the mortgage scenario, includes discussion of rates and terms.
- Interest Rate Discussion, Document Submission, Approval Status, Early Repayment.

**Security & Fraud Labels:**
- Important for verifying client identity during large transactions.
- Identity Verification, Suspicious Activity Check, Code Word Confirmation.
"""


# plan generation

plan_generation_prompt_profile_and_topic_given_detailed_template = """
You are an architect of detailed financial scenarios. Create a DETAIL-RICH PLAN.

## INPUT DATA
- **TOPIC:** <topic>
- **THEME:** <theme>
- **TIMELINE:** <timeline>
- **STAGES:** <num_batches>
- **LABELS:** <provided_labels>
- **PROFILE:** <user_profile>
- **FAMILY & CONTACTS:** <user_relationships>

═══════════════ DETAIL REQUIREMENTS ═══════════════

**MANDATORY FINANCIAL DETAILS (minimum 5–7 per batch):**
- **Exact amounts:** "12,500 rubles", "1.5M rub", "fee 59 rub".
- **Dates:** "payment by May 25", "deposit opened 10.01.2024".
- **Conditions:** "rate 14.5% per annum", "grace period 120 days".
- **Documents:** "contract number 12345", "passport series 4500".
- **Statuses:** "approved", "under review", "scoring rejection".

**Rules:**
- Instead of "discussed credit" write "discussed credit for 500K rub at 18% for 3 years".
- Instead of "paid services" write "paid utilities 4500 rub via app".

**FAMILY & CONTACTS:**
Use data from the **FAMILY & CONTACTS** section in the dialogue:
- Mention relatives and acquaintances by name (transfers to mom, payments for children, etc.).
- Spouse/partner may appear as co-borrower, transfer recipient, or insurance policy holder.
- Friends and acquaintances may advise the client or be mentioned in context ("like my friend").

═══════════════ STRUCTURE ═══════════════

**1. FORMAT:**
- Section headers `BATCH X PLAN`.
- Exactly <num_bullets> bullet points.
- IMPORTANT: First bullet of each batch — **Time Anchor**: "• **Time Anchor:** [Date]"

**3. PROGRESSION:**
**BATCH 1:**
- Point 1: **Time Anchor**.
- Point 2: **Personal Introduction** (I, [Name], work at [Place], income [Amount]).
- Point 3: Financial goal.

**BATCHES 2+:**
- Event development. Condition changes. Problem resolution.

═══════════════ MANDATORY EVENTS BY BATCH ═══════════════
Events are already distributed across batches. Each BATCH N PLAN MUST include
and expand exactly the events specified for it below.
Batch order = chronological order of events in time.
TASK TYPE: <task_type> — ensure events are suitable for this type.

<events_list>

Start generating in Russian.

IMPORTANT ON LANGUAGE:
- All text in each plan bullet — strictly in Russian.
- English words are allowed ONLY in parentheses as a category tag at the end of the bullet: "(CATEGORY: ...)" — and nowhere else.
- FORBIDDEN to insert English labels directly into the bullet text (e.g., "status Resubmission Requirement", "115-FZ Hold activated"). Instead write in Russian: "status «re-submission required»", "account temporarily blocked under 115-FZ".
"""


# message generation

message_generation_prompt_focused_template = """
You generate CLIENT messages in a bank chat.

## CLIENT PROFILE:
<PROFILE_CONTEXT>

## BULLETS FOR THIS SUB-BATCH (GENERATE BASED ON THESE):
<FOCUSED_BULLETS>

## ALREADY PROCESSED BULLETS IN THIS BATCH (do not repeat, use as context):
<PREVIOUS_SUB_BATCH_PLANS>

## RULES
1. **Use all details:** Mention all amounts, dates and names from the plan bullets.
   If a bullet contains multiple facts (amount + date + rate) — spread them across 2–3 messages.
2. **No repetition:** Messages must not duplicate topics from "Already processed bullets".
3. **Style:** Natural Russian. The client can be polite, anxious, confused.
   Not all messages are the same: alternate direct questions, indirect requests and clarifications.
4. **Typos:** Exactly 1–2 typos per every 10 messages.
   Examples of acceptable typos: "скаэите" instead of "скажите", "рубелй" instead of "рублей", "спасиюо" instead of "спасибо".
5. **Format:** One message per line. Message length: 10–40 words. No monologues.
6. **Count:** <SUB_BATCH_SIZE> messages.

## EXAMPLES (one bullet — three style variants):
Bullet: "Want to open a deposit for 1M rub for 6 months".
• Direct:    "Hello, I want to open a 6-month deposit. I have a million rubles, what's the rate?"
• Indirect:  "I'm interested in 6-month deposit terms, considering around a million."
• Clarifying: "If I put exactly a million for 6 months — is that better than for a year?"
"""


# answer generation

ai_assistant_llm_template = """
You are a bank support operator. Reply to the client in RUSSIAN.
TOPIC: <topic>
PLAN FOR THIS STAGE: <current_plan>

PAST CONTEXT (SUMMARY): <summary_context>

<profile_context>

═══════════════ TERMINOLOGY DICTIONARY (use ONLY Russian versions) ═══════════════
Approved → одобрено | Rejected → отказано | Hold → временная блокировка
Resubmission → повторная подача | Chargeback → возврат средств
Chargeback Initiated → возврат средств инициирован | 115-FZ Hold → блокировка по 115-ФЗ
Pending → в обработке | Completed → завершено | Cancelled → отменено
FORBIDDEN to use English labels from the plan in replies to the client.
═══════════════════════════════════════════════════════════════════════════════════

MEMORY: Reference facts from previous batches — do not ask the client to repeat what was already said.
Number accuracy: use ONLY figures the client mentioned or those specified in the plan.
STRICTLY FORBIDDEN to repeat in the current reply a request or question already asked earlier in this dialogue.
If documents / data / reply were already requested — move to the next topic from the plan, do not duplicate the request.
Each reply must advance the dialogue: a new topic, a new fact, or a next step — never a repetition of the previous.

Your task is to conduct the dialogue according to the plan. Be polite and professional.
STRICTLY FORBIDDEN: output JSON, XML, code blocks (```), markdown tables, or any structured markup inside the reply.
Reply ONLY in plain connected text as a bank support operator — as in a live support chat.
STRICTLY FORBIDDEN: listing previously mentioned facts in any form — sections "Recorded:", "Recorded facts:", "Recorded dialogue facts:", "Summary:", "Result:", "All facts:", "For reference:", "Reminder:", "Summary:", "Recording:", "Recording new facts:", "Updating facts:", "Confirming facts:" and any similar headers followed by lists of facts. FORBIDDEN to use a bold header (**text**) followed by a bulleted or numbered list — ANYWHERE in the reply, not just at the start. FORBIDDEN to accumulate a chronological event list with dates and names. Reply ONLY to the current client question — without repeating conversation history. Maximum 2–3 short paragraphs.
<noise_instruction>
═══════════════ VERBOSITY SETTING ═══════════════
<verbosity_instruction>
═══════════════════════════════════════════════════
"""


# COMMON ANTI-TRIVIALITY BLOCK

ANTI_TRIVIALITY_BLOCK = """
═══════════════ ANTI-TRIVIALITY (MANDATORY) ═══════════════
FORBIDDEN to generate trivial tasks. Each task MUST:
1. **Require searching the context** — the answer CANNOT be given without reading the dialogue.
2. **Be indirect** — the question must NOT contain keywords from the answer.
3. **Avoid templates** — do NOT use phrasings like "What amount...", "When was...", "Name...".
   Use: "What is related to...", "What changed after...", "What was the outcome...".
4. **Answer — atomic fact** (1–5 words): number, date, name, status. NOT a sentence.
5. **Answer language**: Question and answer STRICTLY in Russian. FORBIDDEN to use English technical labels from the plan (e.g., «Approved», «Hold», «Resubmission», «Chargeback Initiated», «115-FZ Hold») as the answer or as part of the question. Use the Russian equivalent from the dialogue text.
6. **Anti-positional bias**: The fact MUST be located in the middle of the dialogue — not in the first 2 and not in the last 2 batches. Facts at the beginning/end are trivial due to primacy/recency bias.
7. **Indirect phrasing**: The question must NOT verbatim reproduce phrases from the dialogue. Use paraphrasing: if the dialogue says "interest rate 14.5%", the question should sound "What is the annual rate under the contract?" — not "What is the interest rate?".

AUTO-CHECK (objective — no subjective judgments):
✓ Question contains ≥ 4 words?
✓ At least 2 key words of the ANSWER are absent from the question?
✓ The question contains NO phrase of ≥ 3 words copied verbatim from the dialogue?
✓ Answer is a number / date / name / status (not a sentence)?
✓ Answer in Russian?
→ If any single checkbox fails — redo the task.
"""

# Specialized anti-triviality blocks by task type

ANTI_TRIVIALITY_IE = """
═══════════════ ANTI-TRIVIALITY: INFORMATION_EXTRACTION ═══════════════
1. Question does NOT contain keywords of the answer (forbidden: "What amount 5000 rub...").
2. Answer — atomic fact: number, date, name or status (1–5 words). NOT a sentence.
3. Answer language: STRICTLY Russian (not «Approved» — only «одобрено»).

AUTO-CHECK (objective):
✓ Question contains ≥ 4 words?
✓ At least 2 key words of the answer are ABSENT from the question?
✓ Answer is a number / date / name / status (not a sentence)?
✓ Answer in Russian?
→ If any single checkbox fails — redo.
"""

ANTI_TRIVIALITY_KU = """
═══════════════ ANTI-TRIVIALITY: KNOWLEDGE_UPDATE ═══════════════
1. Question sounds like a request about the CURRENT state — no hints about the change.
2. FORBIDDEN words and all their synonyms:
   «теперь», «новый», «изменился», «после обновления»,
   «сейчас», «на данный момент», «в текущий момент», «пересмотрели», «скорректировали».
3. Answer — the NEW (final) value of the parameter. NOT the old one, NOT both variants.
4. Answer — atomic fact in Russian.

AUTO-CHECK (objective):
✓ Question starts with «Каков», «Какова», «Сколько», «Какой», «Какая», «Когда» or «Кто»?
✓ The question contains NONE of the forbidden words from point 2?
✓ Answer matches the LAST (final) value of the parameter?
✓ Answer in Russian?
→ If any single checkbox fails — redo.
"""

ANTI_TRIVIALITY_TR = """
═══════════════ ANTI-TRIVIALITY: TEMPORAL_REASONING ═══════════════
1. Answer — COMPUTED result, not a date directly from the question.
2. The question does NOT give both dates explicitly — at least one must be determined from context.
3. The "thought" field MUST contain step-by-step calculation:
   «ДД.ММ.ГГГГ → ДД.ММ.ГГГГ = N дней» (including skipping weekends for working days).
4. Answer contains number + unit («5 дней», «3 часа») OR a specific date DD.MM.YYYY.

AUTO-CHECK (objective):
✓ The thought field contains an intermediate calculation?
✓ Answer — number+unit OR specific date (not a description)?
✓ The question does not give both dates explicitly?
→ If any single checkbox fails — redo.
"""

ANTI_TRIVIALITY_INT = """
═══════════════ ANTI-TRIVIALITY: INTERFERENCE ═══════════════
1. FORBIDDEN in the question any hints about the existence of similar events:
   «первый», «второй», «последний», «предыдущий», «из двух», «оба раза»,
   explicit date of the target event, ordinal number of the event.
2. FORBIDDEN to identify the target event through its DIRECT attributes — product type,
   counterparty name, operation name. For example: «по полису на загородную собственность»,
   «за коммунальные услуги по шаблону», «кредит на автомобиль» — this is direct identification,
   not interference. The question must rely ONLY on the consequence, context or role of the event:
   «тот платёж, после которого счёт был заблокирован», «операция, повлёкшая претензию».
3. Interference criterion: if you remove all similar events from the dialogue and leave only the target,
   the question must become UNANSWERABLE — otherwise this is not interference, but information_extraction.
4. Answer — atomic fact in Russian.

AUTO-CHECK (objective):
✓ The question contains none of the forbidden words from point 1?
✓ The question contains no direct attribute uniquely naming the type/product/counterparty of the target event?
✓ Without knowing the dialogue context (only from the question text), one cannot determine which event is meant?
✓ Answer unambiguously distinguishes the target event from similar ones?
✓ Answer in Russian?
→ If any single checkbox fails — redo.
"""

ANTI_TRIVIALITY_COMP = """
═══════════════ ANTI-TRIVIALITY: COMPOSITE ═══════════════
1. Question asks about CAUSE-EFFECT relationship or result — not about listing steps.
   FORBIDDEN: «Перечисли шаги», «Что произошло после X?», «Назови этапы».
   ALLOWED: «Почему...», «Что позволило...», «Что стало причиной...», «Вопреки чему удалось...».
2. Answer — CONCRETE VERIFIABLE FACT: amount, date, name, status, specific bank decision.
   FORBIDDEN vague action-description answers: «сбор документов», «предпринятые меры»,
   «обращение в банк», «выполнение требований» — this is NOT an answer, it is a retelling.
   CORRECT answer: «14 500 руб.», «одобрено», «разблокировка карты», «15.03.2024».
3. FORBIDDEN: the question contains words or phrases that directly hint at the answer or its
   type. For example: «...а само действие по сбору необходимых материалов?» — answer already in question.
   Check: can you read the question and name the answer without opening the dialogue? If yes — REDO.
4. Answer — short concrete fact (1–5 words) from the dialogue text in Russian.

AUTO-CHECK (objective):
✓ Question starts with «Почему», «Что позволило», «Что стало», «Вопреки», «Какой», «Сколько» or equivalent?
✓ Answer — concrete fact (amount/date/name/status), not a description of an action?
✓ Reading only the question (without the dialogue), one cannot name the answer?
✓ Answer is taken from dialogue text (not invented) and is in Russian?
→ If any single checkbox fails — redo.
"""

STRICT_ANSWER_FORMAT = """
═══════════════ STRICT FORMAT OF "answer" FIELD (MANDATORY) ═══════════════
The "answer" field must strictly follow the format — otherwise the EM metric will not count a correct answer:
  Monetary amount  : digits WITHOUT spaces inside number + " руб."  →  "4500000 руб."
  Rate/percentage  : X.X% with DOT as separator                     →  "12.5%"
  Date             : DD.MM.YYYY                                      →  "15.01.2024"
  Date + time      : DD.MM.YYYY HH:MM:SS                            →  "15.01.2024 10:30:00"
  Interval         : N дней / N часов / N минут                     →  "5 дней"
  String/status    : exactly as in the dialogue text                 →  "одобрено"

FORBIDDEN in "answer":
  ✗ spaces inside number: "4 500 000 руб." → write "4500000 руб."
  ✗ comma in decimals:    "12,5%"           → write "12.5%"
  ✗ symbol "₽" or word "рублей"            → write "руб."
  ✗ units in parentheses, explanations, ranges

═══════════════ MANDATORY "reasoning_path" FIELD ═══════════════
Add to JSON the field "reasoning_path" — a list of strings describing step by step the logic of arriving at the answer.
  Each string = one logical step in Russian.
  easy: 1–2 steps  |  medium: 2–3 steps  |  hard: 3–5 steps
  Example (information_extraction): ["Find the event matching the question condition", "Extract the attribute from the found event"]
  Example (composite hard): ["Find event A by feature X", "A triggered event B", "B led to C", "C determines the final answer"]
"""

DIFFICULTY_EXTRA_EASY = """
ADDITIONALLY: Even at the EASY level the question must require EXACT
recall. Do not ask questions answerable by general banking knowledge.

ADDITIONALLY (MANDATORY): Add to JSON the field "decoy_answers" with EXACTLY 3 wrong answers.
Each decoy belongs to a separate type:
1. Value from another event of the same type (different object, different date or different participant).
2. Similar number but clearly wrong: ±20–50% from the correct one, different order or different currency.
3. Correct concept, wrong unit or context (days ↔ hours, rub. ↔ thousand rub.).
All 3 decoys must be present in the dialogue text. At EASY level decoys are sufficiently obvious.
"""

DIFFICULTY_EXTRA_MEDIUM = """
MANDATORY MEDIUM QUESTION STRUCTURE (exactly 2 steps of logical inference):
• Step 1: Identify the needed event/object by its ROLE, CONDITION or
  CONSEQUENCE — NOT by explicit name, date or amount.
• Step 2: Extract the target attribute from the identified event/object.

TEMPLATES FOR ACCEPTABLE MEDIUM QUESTIONS (choose the suitable type):
  ✓ "What is the [attribute] of the [object] that [contextual role]?"
    Example: "What is the rate on the product the client chose instead of the originally requested one?"
  ✓ "What [attribute] was set after [consequence event]?"
    Example: "What limit was in effect after the operation was recognized as fraudulent?"
  ✓ "What did the client receive as a result of [condition/role], not [false context]?"
    Example: "How much was returned to the client under the claim that was resolved successfully?"

EXAMPLE OF UNACCEPTABLE QUESTION: "What is the credit rate?" — direct search,
this is EASY. MEDIUM requires first identifying the needed object by its role.

VERBATIM BAN (MEDIUM): The question must NOT contain ≥ 2 consecutive words
exactly copied from the dialogue text. Paraphrase:
- "ежемесячный платёж" → "регулярный взнос"
- "подача заявки" → "обращение за кредитом"
- "процентная ставка" → "годовой процент по договору"

ADDITIONALLY (MANDATORY): Add to JSON the field "decoy_answers" with EXACTLY 3 wrong answers.
Each decoy belongs to a separate type:
1. Previous value of the same parameter (before the latest change in the dialogue).
2. Value from an analogous event with a different participant or different period.
3. Correct number in wrong unit or time period.
All 3 decoys must be present in the dialogue text. Decoys are plausible but noticeably different.
"""

DIFFICULTY_EXTRA_HARD = """
MANDATORY HARD QUESTION STRUCTURE (minimum 3 steps of logical inference):
• Step 1: Identify the trigger event/condition using INDIRECT clues from the dialogue.
• Step 2: Find the intermediate fact linked to the trigger.
• Step 3: Apply a condition, exception, aggregation or comparison to get
  the final answer.
The question must be formulated so that the answer CANNOT be obtained in 1–2 steps.

MANDATORY TYPES OF HARD QUESTIONS (TEMPLATES):
  1. CROSS-ATTRIBUTION: "[attribute of object] used for [role]?"
     Example: "What is the limit of the card used for the payment that triggered the block?"
     → step 1: find the cause of the block, step 2: find the card, step 3: find its limit.
  2. CONDITIONAL AGGREGATION: "How much in total [events of this type], except [exclusion]?"
     Example: "How much in total was returned to the client across all claims, not counting the partially rejected one?"
     → step 1: find all claims, step 2: exclude the partially rejected one, step 3: sum.
  3. CAUSAL CHAIN: "What was the immediate cause of [result], given that [condition]?"
     → step 1: find the result, step 2: find the event chain leading to it, step 3: apply the condition.

COMPUTED ANSWER (MANDATORY for HARD): gold_answer MUST NEVER be a direct quote from the dialogue text.
The answer must require arithmetic or synthesis from 2+ dialogue facts:
- Amounts: total of several payments, difference between two values, sum over a period
- Rates: difference between two rates or result of a conditional recalculation
- Days: computed interval not explicitly stated in the dialogue
FORBIDDEN: the answer is a phrase of 2+ words or a number that literally appears in one place in the text without calculation.

SELF-REVEALING QUESTION BAN (MANDATORY): the question must NOT contain words, phrases or qualifications from which one can derive the type or content of the answer without reading the dialogue. Examples of VIOLATIONS:
  ✗ «...а само действие по сбору необходимых материалов?» — "document collection" already in question
  ✗ «...не факт обращения, а именно решение банка о снятии лимита?» — narrows to one answer type
  ✗ «...по тому продукту, который давал наибольшую доходность?» — directly points to the needed object
The question must be formulated so that the answer can ONLY be determined by reading the dialogue in full.

ADDITIONALLY (MANDATORY for HARD): Add to JSON the field "decoy_answers" with EXACTLY 3 wrong answers.

DECOY REQUIREMENTS — each of the 3 options MUST belong to a different trap type:
1. **Outdated value** — the correct answer before the last update/change.
   Example: correct answer «102 000 руб.» → decoy «100 000 руб.» (previous limit value).
2. **Unit/format substitution** — correct number but wrong unit or period.
   Example: correct answer «5 200 руб.» → decoy «5 200 долларов» or «5 200 руб. в месяц».
   Correct answer «14.5% годовых» → decoy «14.5% в месяц».
3. **Neighboring event** — value from an analogous event of the same type (another transfer, another payment, another participant).
   Example: correct answer «5 200 руб.» → decoy «5 000 руб.» (amount of a similar transfer to the same recipient).

DECOY RANGE (MANDATORY — maximum proximity to gold):
- Monetary amounts: each decoy differs from gold by NO MORE than 1,000 rub.
- Percentage rates: each decoy differs from gold by NO MORE than 0.2%
- Days/dates: each decoy differs from gold by NO MORE than 1 day
FORBIDDEN: decoy whose difference from gold exceeds 1,000 rub. / 0.2% / 1 day (by the corresponding unit).

ADDITIONAL RULES:
- All 3 options must be REAL facts from the dialogue (not invented) — each decoy must be present in the text.
- Decoy_answers STRICTLY in Russian — forbidden to use English labels
  (e.g., «Chargeback Initiated», «115-FZ Hold», «Approved»).
  Use Russian equivalents from the dialogue text.
- Forbidden to make a decoy obviously absurd (not «5 200 000 руб.» when the correct answer is «5 200 руб.»).

Example: correct answer "5 200 руб." → decoy: ["4 700 руб. (прошлое значение)", "5 200 долларов (неверная единица)", "4 900 руб. (соседний перевод)"]
"""


FAMILY_INTERFERENCE_EASY_INSTRUCTION = """

═══════════════ FAMILY INVOLVEMENT IN DIALOGUE (easy) ═══════════════
Use 1–2 family members from the FAMILY & CONTACTS section:
- Mention them by name in 1–2 batches (transfers to mom, payments for children, co-borrower partner).
- Transaction amounts with relatives must differ from the client's own transactions.
- Do not duplicate names: each family member has a unique name.
"""

FAMILY_INTERFERENCE_MEDIUM_INSTRUCTION = """

═══════════════ FAMILY INTERFERENCE (medium) ═══════════════
GOAL: add confusion through similar financial events of family members.

REQUIREMENT 1 — ONE PARALLEL DOUBLE (mandatory):
For one key client operation add an analogous operation by a family member:
- Client transfers 47,500 rub. → wife/partner transfers 45,000 rub. to the same or similar recipient
- Client takes credit 500,000 rub. → parent or partner discusses credit 480,000 rub.
Amounts differ by 5–15%. Operations — in DIFFERENT batches (difference ≥ 1 batch).

REQUIREMENT 2 — DELEGATED ACTION (in 1 batch):
The client mentions that a relative performed an action on their behalf:
- «Жена изменила лимит карты до 80 000»
- «Сын оформил страховку на мой счёт»
In the next batch the client references this without knowing the exact current value.

FORBIDDEN: give two family members the same name (this technique is for the hard level).
"""

FAMILY_INTERFERENCE_HARD_INSTRUCTION = """

═══════════════ FAMILY INTERFERENCE (MANDATORY FOR HARD) ═══════════════
GOAL: create maximum confusion through financial events of FAMILY MEMBERS from the FAMILY & CONTACTS section.

REQUIREMENT 1 — PARALLEL FINANCIAL DOUBLES (minimum 2 pairs):
For each key financial operation of the client add an ANALOGOUS operation by a family member:
- Client transfers 47,500 rub. → wife transfers 47,800 rub. to the same or similar recipient
- Client takes credit 500,000 rub. at 14.5% → partner discusses credit 480,000 rub. at 14.9%
- Client opens deposit 100,000 rub. for 6 months → parent opens deposit 95,000 rub. for 6 months
Amounts MUST differ by 3–10%. Operation type — identical. Operations — in DIFFERENT batches (difference ≥ 2 batches).

REQUIREMENT 2 — DELEGATED ACTIONS (minimum 1–2 batches):
The client mentions that a RELATIVE performed an action on their behalf or instead of them:
- «Жена вчера звонила и изменила лимит карты до 120 000»
- «Сын оформил страховку на мой счёт»
- «Мама приходила и подавала документы»
In the next batch the client states a DIFFERENT value (doesn't know the actual — only knows from the relative).

REQUIREMENT 3 — CLIENT ROLE CHANGE (in 1 batch):
In one of the batches the client appears in a different financial role:
- Batches 1–3: client is a borrower (credit 500,000 rub., rate 14.5%)
- Batch 5–6: client is a guarantor for wife's/partner's credit (close amount: 480,000 rub., rate 14.9%)
QA task may ask: whose exact amount, at what rate the client is a borrower (not guarantor).

REQUIREMENT 4 — NAME INTERFERENCE:
If the profile contains family members with the SAME name (e.g., wife and daughter both named Марина) —
both MUST appear as transfer recipients or operation participants in DIFFERENT batches.
Transfer amounts are similar (5–15% difference). QA question — about a specific person by context, not by name.

FORBIDDEN: concentrating all family events in a single batch. Distribute across the entire dialogue.
"""


# Probing Questions Generation

# BASE prompts for selection

information_extraction_prompt = """
### ROLE:
You are a leading ML researcher and benchmark architect for evaluating cognitive abilities of LLMs. Your task is to scan a financial plan and extract "Precise Facts" that will form the basis for probing questions.

### INPUT:
Plan text: <plan>

### TASK:
Find all bullets in the text containing **specific, measurable data**. We look for facts verifiable at 100% (True/False).

### SELECTION CRITERIA (Look for sentences with these entities):
1. **Monetary amounts** (Transfer amounts, limits, salaries, balances).
2. **Dates and Time** (Payment deadlines, meeting dates, deadlines).
3. **Percentage rates** (Mortgage, deposits, taxes).
4. **Named entities** (Bank names, tariff plan names, people names, addresses).

### IMPORTANT:
- STEP 0 (mandatory): Before selecting candidates, list ALL found facts
  (amounts, dates, names, rates, statuses) as a numbered list in the "found_facts" field.
  Only then choose from this list. This ensures no fact is missed.
- Choose only bullets where the fact is clearly formulated.
- Ignore general reasoning (e.g., "Need to save money").
- Preserve the original bullet text in full.

### OUTPUT FORMAT (JSON List):
Return a list of JSON objects. Strictly follow key names!

[
    {
        "capability": "information_extraction",
        "category": "e.g. (Money, Date, Rate or Name)",
        "bullet_points": "Full plan bullet text (e.g.: 'Mortgage approved for 5M rub at 12%.')",
        "batch_numbers": 3
    }
]

Field "batch_numbers" — batch number (integer) from which the fact is taken.
Return ONLY the JSON list. No Markdown markup.
"""

knowledge_update_prompt = """
### ROLE:
You are a leading ML researcher and benchmark architect for evaluating cognitive abilities of LLMs. Your specialty is designing "Knowledge Update/Revision" tests within the Fin AI project.

### TASK:
Analyze the financial plan <plan> and find pairs of bullets describing a change in the same parameter (limit, status, address, amount or date) over time.

### SELECTION CRITERIA:
1. **Parameter identification**: Find an object whose value changed (e.g., application status from "Under review" to "Approved" or limit from "100K" to "150K").
2. **Chronological link**: Bullets must represent successive versions of the same fact.
3. **Presence of metadata**: Bullets must contain specific values (dates, amounts, names) that can be extracted without distortion.

### REASONING TECHNIQUE (CoT):
- Step 0 (mandatory): List ALL parameters mentioned more than once
  (limits, rates, statuses, addresses, amounts) as a list in the "found_parameters" field.
  Only then choose pairs from this list.
- Step 1: Look for patterns "was X, became Y" or "previously X, then X changed to Y".
  Including implicit: two bullets with the same parameter but different values.
- Step 2: Identify the specific parameter that was revised.
- Step 3: Match the old version (outdated fact) and the new version (current fact).

POSITION for knowledge_update: old fact can be from any batch,
new fact — MUST be from a later batch. Pair A→B in chronological order.

### OUTPUT FORMAT (JSON list):
[
  {
    "parameter_name": "Name of the changing parameter (e.g., Credit Limit)",
    "old_fact": "Full text of the original (outdated) plan bullet",
    "new_fact": "Full text of the updated (current) plan bullet",
    "subcategory": "Date / Name / String"
  }
]
"""

temporal_reasoning_prompt = """
### ROLE:
You are an expert in designing cognitive tests for LLM agents. Your task is to analyze a financial plan and find pairs of related events for creating time interval calculation tasks.
### TASK:
Find in text <plan> pairs of bullets describing the start and end of one process (e.g.: application submission and approval, transfer sending and receipt).

### SELECTION CRITERIA:
1. **Presence of dates**: Both events in the pair MUST contain explicit dates or timestamps.
2. **Computability**: There must be an extractable interval between events (in minutes, hours or days).

### REASONING TECHNIQUE (CoT):
- Step 0 (mandatory): First list ALL found dates and timestamps
  as a numbered list. Only then choose pairs from this list.
- Step 1: Group dates by meaning (everything related to "Mortgage", "Limit", etc.).
- Step 2: Select pairs "Event A (start) → Event B (finish)".
  Interval between events: MINIMUM 1 day (or 1 hour for intraday events).

POSITION for temporal_reasoning: both dates can be anywhere in the dialogue,
but the pair must describe one connected process (submission → approval, transfer → receipt).

### OUTPUT FORMAT (JSON list):
[
  {
    "event_start": "Full text of the first event (with date)",
    "event_end": "Full text of the second event (with date)",
    "process_name": "Brief process name (e.g., Credit Processing)",
    "expected_interval_type": "days / hours / minutes",
    "batch_numbers": 3
  }
]

Field "batch_numbers" — batch number (integer) where the key of the two events is located.
"""

interference_selection_prompt = """
### ROLE:
You are a data analysis and benchmark design expert for LLMs. Your task is to find in the provided financial plan pairs of events that can cause "context interference" (confusion of similar memories).

### TASK:
Analyze <plan> and select PAIRS of bullets that have a high degree of similarity but different parameters.

### SEARCH CRITERIA FOR PAIRS:
1. **Similar participants**: Events with the same person, bank or project name (e.g., "EcoPlan" and "EcoPlanet").
2. **Different time frames**: Same actions (transfers, payments) performed at different times or dates.
3. **Different parameters**: Same operation types but with different amounts or statuses.

STEP 0 (mandatory): First list ALL found pairs of similar events
(same participant / operation type / bank) as a numbered list.
Only then choose pairs from this list.

POSITION for interference: two similar events MUST be separated by ≥ 2 batches,
so there is enough "masking" information between them.

### OUTPUT FORMAT (JSON list):
Return a list of objects where each object contains both conflicting facts.

[
  {
    "event_a": "Full text of the first event",
    "event_b": "Full text of the second (similar) event",
    "interference_reason": "Brief description of why they may be confused (e.g.: same recipient, but different dates)"
  }
]
"""

composite_task_selection_prompt = """
### ROLE:
You are a data analyst specializing in identifying cause-effect relationships. Your task is to find "event chains" (Composite Event Chains) in the financial plan.

### INPUT:
Plan text: <plan>

### TASK:
Find sequences of 2–3 bullets that are logically connected through shared participants or projects.
Chain logic: A -> B -> C (where B is the linking element).

STEP 0 (mandatory): First list ALL found cause-effect relationships
(«A led to B», «due to X, Y happened») as a numbered list.
Only then choose chains from this list.

POSITION for composite: chain steps can be scattered across the entire dialogue —
this is normal and even desirable. What matters is that the link between steps is logical,
not just chronological.

### EXAMPLE RELATIONSHIPS:
1. **Investments**: Company A invested in B -> B launched product C. (Link between A and C).
2. **Transactions**: Client transferred money to Mom -> Mom paid utilities. (Link between Client and utilities).
3. **Projects**: Stage 1 complete (Development) -> Stage 2 started (Testing) -> Bug found.

### OUTPUT FORMAT (JSON List):
[
  {
    "chain_steps": [
      "First event text (A)",
      "Second event text (B)",
      "Third event text (C)"
    ],
    "bridging_entity": "Shared participant or project (e.g., Company B)",
    "reasoning": "Event 1 affects 2, and 2 is related to 3"
  }
]

Return ONLY JSON.
"""


# SELECTION prompts by difficulty level

# Information Extraction Selection by difficulty

information_extraction_selection_easy = information_extraction_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Choose bullets containing an **explicit, unambiguous fact** mentioned exactly 1 time in the plan.
The fact must be easily findable — a specific amount, date or name not surrounded by similar data.
Examples: "Salary 85,000 rub.", "Date of birth 15.03.1990".
POSITION (easy, 3 batches): use facts from batch 2 (middle). Batches 1 and 3 — only if no alternative.
"""

information_extraction_selection_medium = information_extraction_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM
Choose bullets where the fact **is surrounded by similar data or noise**.
The fact appears among other similar amounts, dates or names, making extraction harder.
Examples: several transfers to different people with similar amounts, several payment dates in one month.
POSITION (medium, 5 batches): use facts from batches 2, 3 or 4. Batches 1 and 5 — only if no alternative.
"""

information_extraction_selection_hard = information_extraction_prompt + """

### DIFFICULTY REQUIREMENT: HARD
POSITION (hard, 8 batches): the main fact cluster must be in batches 3–6. Batches 1 and 8 — FORBIDDEN. Batches 2 and 7 allowed ONLY for aggregation tasks when 3+ needed facts are objectively distributed across the whole dialogue — but no more than 1 fact from batch 2 and no more than 1 from batch 7. If no suitable facts in batches 3–6 — do not create the task.
Choose groups of **2–3 bullets** satisfying ALL conditions:
1. The answer requires **combining information from multiple places** in the plan (aggregation).
2. The target fact **is surrounded by duplicate facts** — similar amounts, dates or names easily confused.
3. The fact is located **in the middle of the plan** (not the first and not the last batch) — testing memory, not recency bias.
4. The answer requires **calculation or synthesis** (total amount, difference, count).
5. AGGREGATION (priority): Look for groups of 3+ same-type facts in DIFFERENT batches (three payment amounts, three rates, three fees). Such groups are the best material for HARD.

Examples:
- Total of 3 transfers to different people (5000 + 3200 + 12000 = 20200 rub.) with 4 more similar transfers present.
- "How many times did the client contact about mortgage?" with contacts about credits, deposits etc. also present.
"""

# Knowledge Update Selection by difficulty

knowledge_update_selection_easy = knowledge_update_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Find pairs where the parameter changed **exactly 1 time** — from one clear value to another.
The change must be obvious and unambiguous.
Examples: application status "Under review" → "Approved", limit 50,000 → 100,000 rub.
POSITION (easy): old fact can be from any batch, new one — from a later batch. Pair in chronological order.
"""

knowledge_update_selection_medium = knowledge_update_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM
Find a parameter satisfying ALL conditions:
1. Value changed **at least 2 times** (pattern A→B→C): first value → intermediate → final.
2. Intermediate (B) and final (C) values are **similar** (minor difference: rate 14.5% → 14.8% → 14.2%).
3. ≥ 2 parameters updated simultaneously — the agent must track the needed one among several changes.
POSITION (medium): first value A — batches 1–2, intermediate B — batches 2–4, final C — batches 3–5. Gap ≥ 1 batch between A and C.
Examples: card limit 50K → 75K → 70K (slightly reduced after increase). Question — about current 70K.
"""

knowledge_update_selection_hard = knowledge_update_prompt + """

### DIFFICULTY REQUIREMENT: HARD
POSITION (hard, 8 batches): first parameter value — from batches 1–3, final value — STRICTLY from batches 4–7. Batch 8 for final value FORBIDDEN (recency bias).
Find a parameter satisfying ALL conditions:
1. Value changed **3 or more times** (update chain: A → B → C → D).
2. Intermediate values **resemble the final one** (e.g., limit 95K → 100K → 98K → 102K — all close, easy to confuse).
3. Updates occurred **in different batches** (not consecutively), with other events between them.
4. At least one intermediate update looks like a "rollback" (value decreased, then increased again).

ROLLBACK PATTERN (mandatory for HARD): Look for chain A→B→A→C, where B is an intermediate value (rollback/cancellation/correction), and C is the final one. B is a trap for the model. If there are fewer than 3 changes — choose a different parameter.

Examples:
- Limit: 50K → 100K → 75K (decrease!) → 120K. Model may answer "100K" (last increase) instead of "120K".
- Rate: 15% → 14.5% → 14.8% → 14.2%. All values very similar, easy to make mistake.

### EXTENDED OUTPUT FORMAT FOR HARD (JSON list):
Instead of the standard format use THIS format, including the full change chain:
[
  {
    "parameter_name": "Name of the changing parameter",
    "old_fact": "Full text of the first (initial) plan bullet with first value",
    "intermediate_facts": [
      "Full text of bullet with SECOND parameter value",
      "Full text of bullet with THIRD value (rollback or intermediate)"
    ],
    "new_fact": "Full text of bullet with FINAL (current) value",
    "value_chain": "A_value → B_value → A_value → C_value (specific numbers/words)",
    "subcategory": "Date / Name / String"
  }
]
CRITICAL: the intermediate_facts field MUST contain all intermediate values.
If intermediate_facts is empty — the task does not meet HARD requirements and must be discarded.
"""

# Temporal Reasoning Selection by difficulty

temporal_reasoning_selection_easy = temporal_reasoning_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Find a pair of events with **two explicit dates** requiring a simple interval calculation.
Both dates given in DD.MM.YYYY format, trivial calculation. Interval: 1–30 days.
Examples: "Application submitted 01.03.2024, approved 05.03.2024" → 4 days.
POSITION (easy): both events can be in any batch, main requirement — clear dates.
"""

temporal_reasoning_selection_medium = temporal_reasoning_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM
Find **3 events** with dates requiring an intermediate calculation.
Need to calculate the interval not between first and last, but via the intermediate event.
Examples: "Application 01.03 → Review 10.03 → Approval 15.03", question: "How long did the review take?"
POSITION (medium): events can be in different batches; prefer batches 2–4.
"""

temporal_reasoning_selection_hard = temporal_reasoning_prompt + """

### DIFFICULTY REQUIREMENT: HARD
POSITION (hard, 8 batches): at least one of the three key events MUST be in batches 3–6. Batches 1, 2, 7, 8 — only for auxiliary events.

MANDATORY: find **3 events** (A → B → C) forming a chain of **2+ dependent calculations**:
  - Event A: starting point with explicit date or offset (from batches 1–3)
  - Event B: intermediate point whose date IS COMPUTED from A (from batches 3–6)
  - Event C: final point whose date IS COMPUTED from B (from batches 4–7)

Find a chain satisfying ALL conditions:
1. Uses **relative dates** ("in 3 working days", "the day after...").
2. The plan has **similar trap events** (other dates/operations nearby) creating confusion.
3. The answer requires **a chain of 2+ calculations**: find date B → from it calculate date C.
4. FORBIDDEN: choose events all three dates of which are explicitly stated in the text.

IMPORTANT: field "batch_numbers" = batch of intermediate event B.
Add to JSON field "event_mid" — intermediate event B.

Examples:
- "Application 10.01 (A), review in 5 working days (B=17.01), 3 more days for documents after approval (C=20.01)."
  Question: "When will the documents be ready?" — need: 10.01 + 5 wd = 17.01 → + 3 days = 20.01.
- "Salary on 25th (A), deposit opened next day (B=26th), interest in a month (C=26th next month)."
"""

# Interference Selection by difficulty

interference_selection_easy = interference_selection_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Find pairs of events with **a shared participant but in different contexts**.
Events are easily distinguishable by operation type or time period.
Examples: transfer to mom in January and transfer to mom in June — different amounts, different purposes.
POSITION (easy): two similar events separated by ≥ 1 batch.
"""

interference_selection_medium = interference_selection_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM
Find pairs of events with **similar parameters and close amounts**.
The difference is less obvious — amounts differ by less than **500 rub.** or dates by less than **3 days**.
PRIORITY CANDIDATES: family members with the same name (e.g., two people named «Алексей»), two sole proprietors or companies with similar names — best material for interference.
DOUBLE INTERFERENCE (mandatory for MEDIUM): find TWO similar participants or events, not one. If only one candidate — do not create the task.
Examples: utility payment 4,500 rub. 25.01 and utility payment 4,800 rub. 25.02. Or transfer to «Алексей Петров» and transfer to «Алексей Сидоров».
POSITION (medium): two similar events separated by ≥ 2 batches (batches 1+4, 2+5, etc.).
"""

interference_selection_hard = interference_selection_prompt + """

### DIFFICULTY REQUIREMENT: HARD
POSITION (hard, 8 batches): similar events MUST be separated by ≥ 3 batches (e.g., batch 2 and batch 6). Selecting adjacent batches is forbidden.
Find **near-identical events** satisfying ALL conditions:
1. Differ **in only 1 minor detail** (date differs by 1 day, amount by 50–200 rub., one letter in a name).
2. Located **far apart** in the plan — minimum 3 batches between them.
3. Event context is **practically identical** (same operation, same participant, same bank).
4. There is a **THIRD similar event** (triple interference) — without this the task is not created.
5. A **distractor of the same type** is placed between similar events — a similar but non-target operation.

TRIPLE INTERFERENCE (MANDATORY): Find THREE similar events — not two, three.

HOMONYMS (bonus): Two different "ИП Иванов", two "Алексей" — ideal candidate.

IMPORTANT: output field must include `event_c` and `batch_c` for the third event.
IMPORTANT: fields `batch_a`, `batch_b`, `batch_c` — batch numbers of each of the three events.

### REDEFINED OUTPUT FORMAT (for HARD only):
[
  {
    "event_a": "Full text of the first event",
    "event_b": "Full text of the second (similar) event",
    "event_c": "Full text of the third (similar) event",
    "batch_a": <batch A number>,
    "batch_b": <batch B number>,
    "batch_c": <batch C number>,
    "interference_reason": "Why three events may be confused"
  }
]

Examples:
- Three transfers to Ivan: 5,000 rub. 15.01 (batch 2), 5,000 rub. 15.02 (batch 5), 5,200 rub. 15.03 (batch 7).
- Three credit offers: 14.5% (batch 1), 14.8% (batch 4), 14.5% from another bank (batch 6).
"""

# Composite Selection by difficulty

composite_selection_easy = composite_task_selection_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Find **a 2-step chain** (A → B) where the link between events is direct and obvious.
Examples: "Submitted credit application" → "Received approval". Question: "How did the application turn out?"
POSITION (easy): both steps can be in adjacent batches. Connection must be obvious.
"""

composite_selection_medium = composite_task_selection_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM
Find **a 3-step chain** (A → B → C) where the intermediate link (B) is necessary to understand the A–C connection.
Examples: "Opened deposit" → "Received interest" → "Transferred interest to card".
POSITION (medium): steps in batches 1–5, separated by ≥ 1 batch between steps.
COMPETING CHAIN (mandatory for MEDIUM): alongside the main chain there MUST be a parallel similar chain with ≥1 shared element (participant, operation type or amount) that is NOT the answer. If there is no competing chain — choose a different set of events.
Example: "Opened deposit" → "Received interest" → "Transferred interest" (main). Parallel: "Opened savings account" → "Received interest" — same "received interest" operation but different account interferes.
"""

composite_selection_hard = composite_task_selection_prompt + """

### DIFFICULTY REQUIREMENT: HARD
POSITION (hard, 8 batches): first chain step — from batches 1–3, last step — from batches 5–7. Batch 8 for the last step FORBIDDEN. Intermediate steps — strictly in batches 3–6.
Find **a chain of 4+ steps** satisfying ALL conditions:
1. Steps are located **in different batches** of the plan (not consecutive, with other events between them).
2. Intermediate links **do not reference each other directly** — the connection is only visible with the full chain.
3. There is a **parallel (false) chain** — similar events that are NOT connected to the main chain but look connected.
4. To answer one must **filter out the false chain** and trace only the real connection.
5. The false chain has ≥ 1 shared step with the real one (same participant, type or amount) — without this it does not interfere.

FALSE CHAIN (mandatory for HARD): the selected batches MUST contain a parallel similar chain that is NOT the answer. If it is absent — choose a different set of events.
REQUIREMENT FOR FALSE CHAIN: it must share ≥1 element with the real chain (participant, operation type, amount), otherwise it does not interfere.

Examples:
- "Requested certificate" → "Found debt 5,000" → "Paid debt 5,000" → "Certificate issued" → "Applied for mortgage".
  Parallel: "Requested statement" → "Received statement" — similar pattern but not linked to mortgage.
"""


# BASE prompts for generation

information_extraction = """
### ROLE:
You are a leading benchmark architect for evaluating episodic memory of LLM agents. Your specialty is creating "Exact Recall" tasks without distortions.

### TASK:
Create a task that checks whether an agent can extract specific information (name, date, address, amount) from dialogue history.

### INPUT:
- PLAN BULLET: <bullet_point> (THE ONLY source of truth).
- DIALOGUE CONTEXT: <conversation_turns>.

### RULES (Strict):
1. **No hallucinations**: Question and answer must be built ONLY on information from the provided <bullet_point>.
2. **Subcategories**: Choose one subcategory: [Date, Name, String].
3. **Time format**: If a date is extracted, use format DD.MM.YYYY HH:MM:SS.
4. **Brevity example**: Question: "What credit amount did I request?" -> Answer: "500,000 rubles".

### CHAIN OF THOUGHT (CoT):
Step 1: Read <bullet_point> and identify the specific atomic fact (name, amount, date or name).
Step 2: Verify this fact is recorded in <conversation_turns>.
Step 3: Formulate a question requiring extraction of this fact without distortion.

### OUTPUT ONLY IN JSON:
{
  "thought": "Reasoning in Russian: I chose the fact [fact name] from the text [text from bullet_point], because...",
  "question": "Question text in Russian?",
  "answer": "Exact fact in STRICT format: amount→'4500000 руб.', rate→'12.5%', date→'15.01.2024', string→'одобрено'",
  "source_chat_ids": [<message_ids>],
  "source_bullet": {
    "capability": "information_extraction",
    "batch_numbers": <number>,
    "bullet_numbers": <number>,
    "bullet_points": "<full_plan_bullet_text>"
  }
}
""" + STRICT_ANSWER_FORMAT + ANTI_TRIVIALITY_IE

knowledge_update_probing_question_final_prompt = """
### ROLE:
You are a leading LLM agent researcher and benchmark architect. Your specialty is verifying "Knowledge Update/Revision". You create tasks that check whether a model can ignore outdated facts and name only the CURRENT version of data.

### TASK:
Generate a complex KNOWLEDGE UPDATE question. The question must concern a parameter that changed during the dialogue (e.g., limit, application status or address).

### INPUT:
- OLD FACT: <old_bullet_point>
- NEW FACT: <new_bullet_point>
- DIALOGUE CONTEXT: <conversation_turns>.

### STRICT RULES (Anti-Hallucination):
1. **Zero hint**: The question must NOT contain words «теперь», «новый», «изменился» or «после обновления». It must sound like a normal current-state query.
2. **Format accuracy**: If the answer is a date or time, use STRICT format: DD.MM.YYYY HH:MM:SS.
3. **Subcategories**: Determine the type of the changing fact: [Date, Name, String].

### CHAIN OF THOUGHT (CoT):
- Step 1: Compare old fact and new fact. Determine which parameter was changed.
- Step 2: Extract the FINAL value of this parameter from <new_bullet_point>.
- Step 3: Ensure the question contains no hints that information ever changed. The question must test "clean" current knowledge.

### OUTPUT FORMAT (JSON ONLY):
{
  "thought": "Parameter: [name]. Old value: [X]. New value: [Y]. Formulating a question about current state with no hints.",
  "subcategory": "Date / Name / String",
  "question": "What is my current credit card limit?",
  "answer": "Final value in STRICT format: rate→'12.5%', amount→'7000 руб.', date→'25.03.2026', status→'одобрено'",
  "source_chat_ids": [<ids_of_turns_where_update_happened>],
  "source_bullet": {
      "capability": "knowledge_update",
      "revision_history": {
          "old": "<old_bullet_text>",
          "new": "<new_bullet_text>"
      }
  }
}
""" + STRICT_ANSWER_FORMAT + ANTI_TRIVIALITY_KU

temporal_reasoning_probing_question_medium_prompt = """
### ROLE:
You are a specialist in evaluating cognitive abilities of LLM agents. Your task is to create a "Temporal Reasoning" task checking the model's ability to compute intervals between events in dialogue history.

### TASK:
Generate a question requiring calculation of a time interval (days, hours or months) between two financial events.

### INPUT:
- PLAN BULLET: <bullet_point> (Your fact source).
- DIALOGUE HISTORY: <conversation_turns>.

### STRICT RULES (Anti-Hallucination):
1. **Exact dates**: Use only dates explicitly stated in <bullet_point>.
2. **Time format**: Any date/time mention in the answer or reasoning must follow the format: DD.MM.YYYY HH:MM:SS.
3. **Category**: Subcategory of this fact is always "Date".
4. **Mathematical accuracy**: Carefully recompute the difference. If 4 days passed between 01.01.2024 and 05.01.2024, the answer must be "4 days".

### CHAIN OF THOUGHT (CoT):
- Step 1: Find in <bullet_point> two related events with stated dates.
- Step 2: Convert both dates to DD.MM.YYYY format. If a date is only implied
  (e.g., "N days from [date]") — compute it explicitly here.
- Step 3: Calculate the difference STEP BY STEP. Record intermediate steps in thought:
  Example: "01.03 → 05.03: 4 days. 28.02 → 05.03: 5 days (February has 28 days)."
  When counting working days — explicitly list each day and skip Saturday/Sunday.
- Step 4: Formulate a question requiring this calculation. Do NOT give both dates explicitly in the question.

### EXAMPLE:
Question: "How many days passed between the application submission and receiving the card?"
Answer: "5 days".

### OUTPUT FORMAT (JSON ONLY):
{
  "thought": "Event A: [date], Event B: [date]. Difference is [X]. Forming question...",
  "subcategory": "Date",
  "question": "Question text in Russian?",
  "answer": "ONLY number + unit: '3 дня', '12 часов', '30 минут'. Ranges and explanations forbidden.",
  "source_chat_ids": [<ids_of_turns_with_dates>],
  "source_bullet": {
    "capability": "temporal_reasoning",
    "batch_numbers": <number>,
    "bullet_numbers": <number>,
    "bullet_points": "<source_plan_bullet_text>"
  }
}
""" + STRICT_ANSWER_FORMAT + ANTI_TRIVIALITY_TR

context_interference_probing_question_final_prompt = """
### ROLE:
You are a leading ML engineer and researcher in LLM agent episodic memory. Your task is to create a "Context Interference" task.

### TASK:
Generate a complex task that checks whether an agent can separate two similar events in dialogue history, sharing the same participants (people, banks, companies) but with different time frames or conditions.

### INPUT DATA:
- EVENT #1: <bullet_point_1>
- EVENT #2: <bullet_point_2> (Similar to first event in participants).
- DIALOGUE HISTORY: <conversation_turns>.

### STRICT VALIDATION RULES:
1. **Separating similar events**: Choose events with similar names or same participants but different dates or amounts.
2. **Typing**: Specify subcategory: [Date / Name / String].
3. **Time format**: Any date/time mention in the answer must strictly follow: DD.MM.YYYY HH:MM:SS.
4. **No hints**: The question must not contain hints that there were multiple events (e.g., do not use "second time" or "again").

### REASONING TECHNIQUE (Chain of Thought):
- Step 1: Analyze <bullet_point_1> and <bullet_point_2>. Find the shared element and the distinguishing element.
- Step 2: Formulate the question so the model can answer correctly only if it precisely matched the participant with the needed time interval.
- Step 3: Ensure the answer is an atomic fact without distortions.

### EXAMPLE:
- Context: At 10:00 client transferred 5,000 rub. to Ivan I. At 15:00 client transferred 2,000 rub. to the same Ivan I.
- Question: "What amount did the client send to Ivan I. in the first half of the day?"
- Answer: "5,000 rub.".

### OUTPUT FORMAT (JSON ONLY):
{
  "thought": "I found two events with participant [Name]. They differ in time: [Time 1] and [Time 2]. Creating question about event [1].",
  "subcategory": "Date / Name / String",
  "question": "Question text in Russian?",
  "answer": "Exact fact of the needed event in STRICT format: amount→'5000 руб.', date→'15.01.2026', string→as in dialogue",
  "source_chat_ids": [<ids_of_turns_with_both_events>],
  "source_bullet": {
    "capability": "context_interference",
    "interfering_events": {
      "event_1": "<first_bullet_text>",
      "event_2": "<second_bullet_text>"
    }
  }
}
""" + STRICT_ANSWER_FORMAT + ANTI_TRIVIALITY_INT

composite_task_generation_prompt = """
### ROLE:
You are an expert in logic and cognitive testing of LLMs. Your task is to create a "Composite Task" (Multi-hop Reasoning).

### INPUT CHAIN:
Event chain: <chain_text>
- DIALOGUE HISTORY: <conversation_turns>.
### TASK:
Formulate a question that requires the agent to trace the cause-effect link from the FIRST event to the LAST through key steps.

### RULES:
1. **Indirect Question**: Do not ask about facts directly. Ask about cause-effect link or key linking element.
   - *Bad*: "What did company B do?"
   - *Good*: "What allowed company A to achieve result C?"
2. **Grounding**: The answer must be unambiguously identifiable in the dialogue text — not a technical label from the plan.
3. **Answer language**: Answer STRICTLY in Russian. FORBIDDEN to use English technical terms (Resubmission, Hold, Approved, etc.). Use the Russian equivalent from the dialogue.
4. **Answer format**: Answer — CONCRETE VERIFIABLE FACT from the dialogue: amount, date, name, status, specific bank decision (approved/rejected/unblocked/limit removed).
   - *FORBIDDEN* (vague): "measures taken", "document collection", "contacting the bank", "fulfilling requirements"
   - *FORBIDDEN* (label): "Resubmission", "Hold", "Approved"
   - *Good* (concrete fact): "14500 руб.", "одобрено", "разблокировка карты", "15.03.2024", "снятие ограничения по 115-ФЗ"
5. **Question accuracy**: The question must NOT contain words or qualifications from which one can guess the answer without the dialogue.
   - *Bad*: "...not the contact fact, but the document collection action itself?" — answer in question
   - *Good*: "What exactly changed the bank's attitude toward the client after the incident?" — answer unknown without dialogue

### CHAIN OF THOUGHT (CoT):
- Step 1: Identify the starting point (A) and ending point (C).
- Step 2: Find the key linking action (B) — what directly enabled the transition to C.
- Step 3: Find the Russian-language description of B in the dialogue text (not the plan label).
- Step 4: Formulate the question precisely by B's role: "intermediate" if B is the middle of the chain, "immediate" if B is the last step before C.

### OUTPUT JSON:
{
  "thought": "I connect event [A] with event [C]. Key link [B] — [Russian description from dialogue].",
  "question": "Question text in Russian?",
  "answer": "Final fact in STRICT format: amount→'3200000 руб.', status→'одобрено', chain→'А → Б → В'",
  "subcategory": "String"
}
""" + STRICT_ANSWER_FORMAT + ANTI_TRIVIALITY_COMP


# GENERATION prompts by difficulty level

# Information Extraction Generation by difficulty

information_extraction_gen_easy = information_extraction + """

### DIFFICULTY REQUIREMENT: EASY
Create a SIMPLE question extracting **one explicit fact** mentioned exactly 1 time.
The question must be direct, requiring no reasoning.
Example: "What amount was specified in the credit application?" → "500,000 rubles".
""" + DIFFICULTY_EXTRA_EASY

information_extraction_gen_medium = information_extraction + """

### DIFFICULTY REQUIREMENT: MEDIUM (2 inference steps)
Create a question requiring TWO-STEP inference:
  Step 1 — identify the needed object/event by its FUNCTIONAL ROLE or CONSEQUENCE.
  Step 2 — extract the target attribute from the identified object.

MANDATORY TYPES (choose one):
  A) Fact tied to object role: "What is the [attribute] of the [object] that performed [role]?"
     Example: "What amount was in the contract signed after the repeat contact?"
  B) Fact requires excluding similar: from 2 analogous events choose the one tied to the condition.
     Example: "How much was returned to the client under the claim submitted via the app (not at the branch)?"

FORBIDDEN: direct question of type "What is the amount/rate/date of [X]?" — this is EASY.
""" + DIFFICULTY_EXTRA_MEDIUM

information_extraction_gen_hard = information_extraction + """

### DIFFICULTY REQUIREMENT: HARD (minimum 3 inference steps)
Create a question with a MANDATORY 3-step chain — the answer CANNOT be obtained in 1–2 steps.

MANDATORY TYPE (choose one):
  1. CROSS-ATTRIBUTION (3 steps):
     Step 1: determine the trigger event by indirect context (not by direct name).
     Step 2: find the object linked to the trigger.
     Step 3: extract the target attribute of this object.
     Example: "What is the limit of the tool used in the operation that triggered the block?"
       → step 1: find the operation that caused the block;
       → step 2: find the card/account used in it;
       → step 3: find the limit of that card/account.
  2. CONDITIONAL AGGREGATION (3 steps):
     Step 1: find all events of the needed type across the whole dialogue.
     Step 2: exclude events not satisfying the condition.
     Step 3: compute the total for the remaining.
     Example: "How much in total was returned to the client for operations that concluded without a dispute?"
  3. CONDITIONAL CHAIN (3 steps):
     Step 1: determine the condition (from which status/decision to proceed).
     Step 2: find the value for this condition (not for a different variant).
     Step 3: apply an additional filter (date, channel, participant).
     Example: "What amount appeared in the document issued in connection with the first of two insurance contacts?"

FORBIDDEN: questions where the answer is a direct attribute of a single event.
FORBIDDEN: question containing ≥ 3 consecutive words from dialogue text.
FORBIDDEN: question answerable by searching for a single keyword.

"VERBOSE-BUT-EASY" TRAP (most common defect — FORBIDDEN):
  The question is long and formal, but actually requires only 1 search step.
  ✗ "What monthly payment is set for the client's loan according to current service terms at the end of the dialogue?" — with a single loan this is a 1-step search; words "current terms" and "at end" add no inference steps.
  ✗ "What is the client's card limit confirmed during the latest contact?" — if one card — 1 step.
  Check: remove all descriptive words — if the question reduces to "What is [attribute] of [object]?",
  this is a 1-step question, even if wordy. REDO as cross-attribution or aggregation.
""" + DIFFICULTY_EXTRA_HARD

# Knowledge Update Generation by difficulty

knowledge_update_gen_easy = knowledge_update_probing_question_final_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Create a SIMPLE question about the current value of a parameter that changed **exactly 1 time**.
Example: "What is my current credit limit?" (limit changed once).
""" + DIFFICULTY_EXTRA_EASY

knowledge_update_gen_medium = knowledge_update_probing_question_final_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM (2 inference steps)
Create a question requiring TWO-STEP inference:
  Step 1 — determine WHICH update is relevant given the condition (choose among several changes).
  Step 2 — name the final value for the chosen variant.

MANDATORY TYPES (choose one):
  A) Conditional update: parameter changed along different branches (with insurance / without), question asks
     about the realized variant.
     Example: "What monthly payment was fixed for the variant the client approved?"
  B) Competing updates: multiple parameters updated simultaneously, question requires
     identifying the needed one by its FUNCTION.
     Example: "Which indicator was changed to reduce the client's monthly load?"

FORBIDDEN: direct question "What is the current limit/rate/term?" — this is EASY.
""" + DIFFICULTY_EXTRA_MEDIUM

knowledge_update_gen_hard = knowledge_update_probing_question_final_prompt + """

### DIFFICULTY REQUIREMENT: HARD (minimum 3 inference steps)

HARD INPUT DATA:
- INITIAL VALUE: <old_bullet_point>
- INTERMEDIATE VALUES (traps): <intermediate_facts>
- FINAL VALUE: <new_bullet_point>

Create a question with a MANDATORY 3-step chain:
  Step 1 — determine the CONDITION under which the parameter took the final value
    (not the intermediate or alternative one).
  Step 2 — from the change chain establish that EXACTLY this condition was realized
    (filter out intermediate traps from <intermediate_facts>).
  Step 3 — name the final parameter value under this condition.

MANDATORY QUESTION PROPERTIES:
1. Parameter changed **3+ times** — all intermediate values from <intermediate_facts> are traps.
2. Question asks about THE PARAMETER THROUGH ITS FUNCTION or APPLICABILITY CONDITION —
   not through its direct name.
   - Dialogue: "rate recalculated to 14.2%" → question: "What rate is charged on the debt if the client kept the insurance?"
   - Dialogue: "limit 102,000 rub." → question: "What amount can be spent at once with confirmed income?"
3. The dialogue has close values (chain 95K → 100K → 98K → 102K) — model must
   traverse the full chain and verify 102K is final, not 98K or 100K.
4. NEUTRAL PHRASING: FORBIDDEN words "current", "last", "after change",
   "now", "at this moment", "final", "effective".
   Reason for ban: the model must NOT directly search for the "latest value" —
   it must traverse the full chain and verify that exactly this value was realized.
5. VERBATIM BAN: parameter name — only through a synonym.
6. BAN ON TELEGRAPHING THE CHAIN: question must NOT contain words like
   "after all changes", "ultimately", "taking correction into account", "as a result of review" —
   this hints that the value changed, and the model searches for "last", bypassing the chain.
   Bad example: "What is the rate under the contract taking into account the latest revision?" — hint.
   Good example: "What is the rate under the executed contract?" — neutral.

CONDITIONAL UPDATE (priority type): Question sounds neutral, but the correct answer —
only the realized variant (not one discussed as an alternative).
Example: "What rate applies under the executed contract?" (10.5% and 12% were discussed, 10.5% chosen with insurance).
""" + DIFFICULTY_EXTRA_HARD

# Temporal Reasoning Generation by difficulty

temporal_reasoning_gen_easy = temporal_reasoning_probing_question_medium_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Create a SIMPLE question about the time interval between **2 events with explicit dates**.
Trivial calculation — direct date difference.
Example: "How many days passed between the application submission and the approval?"
""" + DIFFICULTY_EXTRA_EASY

temporal_reasoning_gen_medium = temporal_reasoning_probing_question_medium_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM (2 inference steps)
Create a question requiring TWO-STEP temporal inference:
  Step 1 — compute or identify an intermediate date from context (it is NOT stated explicitly).
  Step 2 — from it calculate the target interval or date.

MANDATORY REQUIREMENT: at least one of the involved dates must be COMPUTED, not read directly.
  Example A: "How many days did document processing take?" — the processing start date is computed from
    "2 working days after submission" (step 1: compute start date, step 2: find end date, subtract).
  Example B: "When did the client receive the approval notification?" — approval date is not stated explicitly,
    must be computed from SLA (step 1: application date + N days = approval date, step 2: + M days notification).

FORBIDDEN: question where both dates are stated explicitly and only the difference needs computing — this is EASY.
""" + DIFFICULTY_EXTRA_MEDIUM

temporal_reasoning_gen_hard = temporal_reasoning_probing_question_medium_prompt + """

### DIFFICULTY REQUIREMENT: HARD
Create the MOST COMPLEX question satisfying ALL conditions:
1. Uses **relative dates** ("in N working days", "next week after...").
2. The answer requires **a chain of 3+ steps**: find date A → compute date B → from B compute date C (final answer).
3. The dialogue has **similar trap dates** (another event on a close date — model may stop at B instead of C).
4. Question sounds SIMPLE ("When should I expect the result?"), but the answer requires traversing the FULL chain.

FORBIDDEN: give any dates explicitly in the question. ALL three dates must be computed.
FORBIDDEN: use event words verbatim in the question.
FORBIDDEN: formulate a question whose answer is intermediate date B (not C).

WORKING DAYS: If context says "N working days" — exclude Saturday and Sunday. In thought field show step by step which exact days were counted (A + N wd = B, B + M wd = C).
TRAP: The dialogue must have a similar event on date B (intermediate) — model may accept B as the answer.

Example: "When should I expect the documents to be ready?" (need: application date A → + 5 wd = approval date B → + 3 wd = readiness date C, where B coincides with another trap event).
""" + DIFFICULTY_EXTRA_HARD

# Interference Generation by difficulty

interference_gen_easy = context_interference_probing_question_final_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Create a SIMPLE question distinguishing **two events with a shared participant but in different contexts**.
The difference is obvious.
Example: "What amount did I transfer to mom in January?" (with transfers in January and June).
""" + DIFFICULTY_EXTRA_EASY

interference_gen_medium = context_interference_probing_question_final_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM (2 inference steps)
Create a question requiring TWO-STEP inference to distinguish events:
  Step 1 — identify THE NEEDED one from 2 similar events by its CONTEXTUAL ROLE or CONSEQUENCE.
  Step 2 — extract the needed attribute from the identified event.

MANDATORY REQUIREMENT: the identifier in the question is NOT a date/amount/participant name,
but the contextual role of the event (what it caused, after what it occurred, in connection with what).
  Example A: "What amount did the client transfer when it was needed to close the dispute?"
    (from two transfers to the same recipient choose the one linked to the dispute).
  Example B: "How much did the client pay the time the payment went through with a delay?"
    (from two similar payments choose the one that took longer).

FORBIDDEN: use date, amount or ordinal number of the event as identifier in the question.
""" + DIFFICULTY_EXTRA_MEDIUM

interference_gen_hard = context_interference_probing_question_final_prompt + """

### HARD INPUT DATA (3 events):
- EVENT #1: <bullet_point_1>
- EVENT #2: <bullet_point_2>
- EVENT #3: <bullet_point_3>

### DIFFICULTY REQUIREMENT: HARD
Create the MOST COMPLEX question with triple interference satisfying ALL conditions:
1. Three events are **nearly identical** — each differing in only 1 minor detail.
2. The question **contains no hints** — the model must identify the needed event ONLY by semantic context.
3. **A lot of other information** is placed between the events in the dialogue (batches are far apart).
4. **VERBATIM BAN**: do NOT use unique words/phrases of the target event.
   - Bad: "What amount did I transfer to Ivan on January 15?" ← contains date
   - Good: "What amount did I transfer to Ivan when paying for the repair?" ← contextual anchor

PRIORITY TYPES OF HARD QUESTIONS (choose one of three):
1. CONTEXTUAL DISTINCTION: "What amount did the client transfer to [name] when [situation]?" — answer only with understanding of CONTEXT of all three events.
2. EXCLUSION (negative fact): "Which of the three contacts about [topic] did not end with [result]?" or "In which case did the client fail to [action]?" — model must check ALL three events and find the exception.
3. AGGREGATION WITH EXCLUSION: "How many times did the client [action] — not counting the case when [context]?" — need to find all events and subtract the non-target one.

ABSOLUTE BANS IN QUESTION:
- Ordinal words: "first", "second", "last", "previous"
- Explicit date, amount or unique parameter of the target event
- "of three", "all three", "both times"
- Verbatim quoting of dialogue phrases (≥ 3 consecutive words)
- DIRECT ATTRIBUTES as identifier: product type, operation name, counterparty name.
  ✗ "...under the policy on suburban property" — direct naming of product type/object.
  ✗ "...for utilities via template" — direct naming of payment category.
  ✗ "...car loan" — direct naming of loan purpose.
  Rule: if you remove all three events from the dialogue and leave only the target,
  the question must become UNANSWERABLE — otherwise this is not interference, but information_extraction.

IMPORTANT: in source_bullet add "event_3": "<third event text>".

Example: "In which transfer to Ivan was the amount higher than the utility payment?" (with 3 transfers 5,000, 5,000, 5,200 — need to compare with utility amount and find only the case where it exceeds it).
""" + DIFFICULTY_EXTRA_HARD

# Composite Generation by difficulty

composite_gen_easy = composite_task_generation_prompt + """

### DIFFICULTY REQUIREMENT: EASY
Create a SIMPLE question with **a 2-step chain** (A → B). Direct connection.
Example: "How did my credit application turn out?" (submitted → approved).
""" + DIFFICULTY_EXTRA_EASY

composite_gen_medium = composite_task_generation_prompt + """

### DIFFICULTY REQUIREMENT: MEDIUM (3-step chain A → B → C)
Create a question requiring passage through intermediate link B.
  Step 1 — identify starting event A by its indirect description.
  Step 2 — find intermediate link B (it is NOT mentioned explicitly in the question).
  Step 3 — name the final fact C, reachable only through B.

MANDATORY REQUIREMENT: the answer is RESULT C, which cannot be found without B.
  Example: "How did the fraudulent operation case end?" (A: operation → B: chargeback → C: amount returned).
  Question asks about C, but C cannot be found without knowing B.

FORBIDDEN: question of type "What happened after [explicit event]?" — this is EASY.
Question must ask about RESULT (C), naming neither A nor B explicitly.

COMPETING CHAIN (mandatory to use): the dialogue contains a parallel similar chain with ≥1 shared element, leading to a similar but WRONG result C'. The question must be formulated so the competing chain looks like a plausible answer — the correct C differs from C' precisely because it belongs to the main chain, not the competing one.
""" + DIFFICULTY_EXTRA_MEDIUM

composite_gen_hard = composite_task_generation_prompt + """

### DIFFICULTY REQUIREMENT: HARD
Create the MOST COMPLEX question satisfying ALL conditions:
1. Chain of **4+ steps**, steps scattered across **different parts of the dialogue**.
2. The dialogue has a **parallel false chain** (similar events not linked to the answer).
3. Intermediate links **do not reference each other directly** — the connection is visible only with all steps in mind.
4. Question asks about **the final result, cause-effect link or negative fact**.
5. **VERBATIM BAN**: Question does NOT use the name of the starting event or the final result verbatim.

FALSE CHAIN AS DISTRACTOR (mandatory): The question must be formulated so the false chain looks like a plausible answer for a model that did not traverse the FULL real chain. The correct answer is reachable ONLY through the real chain A→B→C→D; a model that stops at the false chain gets a similar but wrong result. This requirement is more important than question elegance.

FORBIDDEN QUESTION TYPES: "List the steps...", "What happened after X?".

PRIORITY TYPES OF HARD QUESTIONS (choose one of four):
1. CAUSE: "Why...", "What allowed...", "Despite what was it possible..." — require traversing the FULL chain.
2. BLOCK/NEGATION: "What prevented [goal]?", "Why was [result] never achieved?", "What interrupted the process?" — model must find the step that broke the chain.
3. COUNTERFACTUAL EXCLUSION: "What would have happened if [intermediate step] had not occurred?" — requires understanding which step was critical.
4. CANCELLED ACTION: "Which step was performed but then cancelled/annulled?" — need to find the step that was in the chain initially but then left it.

Examples:
- Type 1: "Why was I able to get a mortgage despite the initial rejection?" (rejection → debt → repayment → retry → approval).
- Type 2: "Why did the client never use the approved credit?" (approved → conditions changed → client refused → credit annulled).
- Type 4: "Which client action was cancelled by the bank after confirmation?" (operation confirmed → blocked under 115-FZ → annulled).
""" + DIFFICULTY_EXTRA_HARD


# TASK VALIDATION

task_validation_prompt = """
You are a strict quality validator of QA tasks for the episodic memory LLM benchmark.

TASK: Check whether the generated task is valid.

INPUT DATA:
- TASK TYPE: <task_type>
- DIFFICULTY: <difficulty>
- QUESTION: <question>
- ANSWER: <answer>
- SOURCE (plan bullet): <source_bullet>

═══════════════ VALIDATION CRITERIA ═══════════════

Check ALL 6 criteria. The task is VALID only if ALL = YES.

1. **GROUNDING**: Can the answer be LOGICALLY derived from the source?
   - YES: answer is directly in the source OR logically computable from it.
   - NO: answer contains information NOT in the source (hallucination).

2. **ANSWERABILITY**: Can this question be answered unambiguously?
   - YES: question has one clear answer.
   - NO: question is too vague, or multiple equally valid answers are possible.

3. **NON_TRIVIAL**: Does the question require REAL memory work?
   - YES: answering requires (a) recalling a specific fact from the MIDDLE of context,
     (b) question does NOT contain answer keywords, (c) context has similar
     distractor facts.
   - NO: question can be answered by common sense, or the question contains
     a hint, or the fact is at the beginning/end of the dialogue (recency/primacy bias).

4. **FORMAT**: Is the answer atomic and brief (1–5 words, number or date)?
   - YES: answer is clear and short.
   - NO: answer is a long sentence or description.

5. **DIFFICULTY_FIT**: Does the task match the stated difficulty?
   Count the minimum number of logical inference steps to get the answer:
   - For EASY: 1 step — fact mentioned once, direct keyword search.
     YES if: question requires exactly 1 step. NO if: 2+ steps required (overrated).
   - For MEDIUM: exactly 2 steps — (1) identify object by role/condition/consequence,
     (2) extract attribute. Direct 1-step question ("What is the credit rate?") — NO.
   - For HARD: minimum 3 steps — indirect trigger → intermediate fact →
     conditional/aggregated conclusion. Question answerable in 1–2
     steps is not HARD (NO).
   - YES: task requires at least the stated number of steps.
   - NO: task is simpler than stated level (fewer steps than required).

6. **SELF_ANSWER**: Could someone answer the question WITHOUT the dialogue, using only the question wording and general banking knowledge?
   - YES: impossible to guess — answer is specific and hidden in the dialogue.
   - NO: answer follows from the question itself or is an "obvious" banking fact.

═══════════════ OUTPUT FORMAT ═══════════════
Return ONLY JSON:
{
  "grounding": "YES" or "NO",
  "answerability": "YES" or "NO",
  "non_trivial": "YES" or "NO",
  "format": "YES" or "NO",
  "difficulty_fit": "YES" or "NO",
  "self_answer": "YES" or "NO",
  "verdict": "PASS" or "FAIL",
  "reason": "Brief explanation (1 sentence)"
}
"""


# GRAPH PIPELINE

graph_gen_information_extraction_easy = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: easy
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>% of the maximum possible number of connections.

REQUIREMENTS:
- Only real, valid edges (valid: true, stale: false).
- Nodes: persons, accounts, products, events, amounts, organizations.
- Edges: reflect specific banking operations from events.
- For type INFORMATION_EXTRACTION: key fact — a direct property of one node or one edge.
- Each important fact from events must be represented in node or edge attributes.
- ground_truth: true — for nodes/edges containing the key answer fact.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [
    {
      "id": "person_1",
      "type": "person",
      "label": "Иван Петров",
      "attributes": {"age": 34, "role": "client", "income": "85000 руб."},
      "ground_truth": true
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "person_1",
      "target": "account_1",
      "relation": "owns",
      "attributes": {"since": "10.01.2024"},
      "valid": true,
      "stale": false,
      "ground_truth": false
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_information_extraction_medium = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: medium
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>% of the maximum possible number of connections.

REQUIREMENTS:
- Add 20% stale edges (stale: true) — these are previous parameter values that subsequently changed.
- Mark stale edges: stale: true, valid: true (they were real but are no longer current).
- For type INFORMATION_EXTRACTION: key fact — a specific attribute of a node or edge.
- There must be a clear difference between the current and stale value of a parameter.
- ground_truth: true — for nodes/edges with the current answer fact.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "transfer_2",
      "source": "person_1", "target": "account_ext_1",
      "relation": "transferred_to",
      "attributes": {"amount": "5000 руб.", "date": "10.03.2024", "time": "14:00", "ref": "TXN-002"},
      "valid": true, "stale": true, "ground_truth": false
    },
    {
      "id": "transfer_3",
      "source": "person_1", "target": "account_ext_1",
      "relation": "transferred_to",
      "attributes": {"amount": "5 100 руб.", "date": "15.03.2024", "time": "18:00", "ref": "TXN-003"},
      "valid": true, "stale": false, "ground_truth": false
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_information_extraction_hard = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: hard
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>% of the maximum possible number of connections.

REQUIREMENTS:
- 40% of edges are stale (stale: true).
- 15% of edges are invalid (valid: false) — false connections with plausible values.
- Add 2–3 duplicate nodes (ground_truth: false) — as SIMILAR as possible to the original.
  Duplicates must differ from the original in ONLY ONE attribute (e.g., amount 100 rub. less,
  rate 0.1% different, date 1 day earlier). Duplicate label — same name + "(доп.)" or one letter difference.
- For type INFORMATION_EXTRACTION: key fact — in a node/edge from events 3–6.
- Invalid edges (valid: false) contain plausible values — not "noise", but realistic data.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [
    {
      "id": "account_1",
      "type": "account",
      "label": "Счёт №40817810001234",
      "attributes": {"balance": "4 500 000 руб.", "opened": "15.01.2024"},
      "ground_truth": true
    },
    {
      "id": "account_1_dup",
      "type": "account",
      "label": "Счёт №40817810001234 (доп.)",
      "attributes": {"balance": "4 400 000 руб.", "opened": "15.01.2024"},
      "ground_truth": false
    }
  ],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

# KNOWLEDGE UPDATE

graph_gen_knowledge_update_easy = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: easy
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type KNOWLEDGE_UPDATE: a parameter update chain must be present (minimum 2 edges: old → new).
- Only valid edges (valid: true).
- Each parameter change — a separate edge with "from_value" and "to_value" fields in attributes.
- Last (current) value: ground_truth: true.
- Previous values: stale: true.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "edge_update_1",
      "source": "loan_1",
      "target": "rate_node_1",
      "relation": "rate_changed",
      "attributes": {"from_value": "16.0%", "to_value": "14.5%", "date": "15.02.2024"},
      "valid": true,
      "stale": true,
      "ground_truth": false
    },
    {
      "id": "edge_update_2",
      "source": "loan_1",
      "target": "rate_node_2",
      "relation": "current_rate",
      "attributes": {"value": "12.9%", "since": "05.03.2024"},
      "valid": true,
      "stale": false,
      "ground_truth": true
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_knowledge_update_medium = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: medium
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type KNOWLEDGE_UPDATE: parameter changes at least 3 times (3 update edges).
- 20% stale edges (stale: true), apart from update edges.
- Each update: edge with date and before/after values.
- Last current edge: stale: false, ground_truth: true.
- All intermediate: stale: true, ground_truth: false.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_knowledge_update_hard = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: hard
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type KNOWLEDGE_UPDATE: parameter changes 4+ times with pattern A→B→A→C (rollback and final change).
  Example: rate 15% → 13% → 15% → 12.5%. The intermediate rollback makes the "last" value non-obvious.
- Final current value (C) — strictly in events 4–7 (not in 8).
- 40% stale edges, 15% invalid (valid: false) with plausible values (not "noise").
- Update edges: each has "value" and "date" fields in attributes — without explicit "outdated" markers.
  Only dates (chronology) allow determining which value is final.
- Add 1–2 duplicate parameter nodes with NEAR-CORRECT values (difference ≤0.5% or ≤100 rub.).

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "rate_change_1",
      "source": "loan_1",
      "target": "rate_node_1",
      "relation": "had_rate",
      "attributes": {"value": "15.0%", "date": "01.01.2024"},
      "valid": true, "stale": true, "ground_truth": false
    },
    {
      "id": "rate_change_4",
      "source": "loan_1",
      "target": "rate_node_4",
      "relation": "current_rate",
      "attributes": {"value": "12.5%", "date": "10.04.2024"},
      "valid": true, "stale": false, "ground_truth": true
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

# TEMPORAL REASONING

graph_gen_temporal_reasoning_easy = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: easy
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type TEMPORAL_REASONING: edges MUST contain exact dates in attributes.date or attributes.timestamp.
- All edges are valid (valid: true).
- At least 2 events with dates linked to the same object.
- ground_truth: true — for edges providing information for temporal calculation.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "edge_event_1",
      "source": "person_1",
      "target": "application_1",
      "relation": "submitted",
      "attributes": {"date": "10.01.2024", "status": "на рассмотрении"},
      "valid": true,
      "stale": false,
      "ground_truth": true
    },
    {
      "id": "edge_event_2",
      "source": "application_1",
      "target": "decision_1",
      "relation": "resulted_in",
      "attributes": {"date": "15.01.2024", "status": "одобрено"},
      "valid": true,
      "stale": false,
      "ground_truth": true
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_temporal_reasoning_medium = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: medium
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type TEMPORAL_REASONING: minimum 4 events with dates (including stale ones).
- 20% stale edges (stale: true) with dates of previous states.
- Several close dates (±1–3 days) to complicate the calculation.
- ground_truth: true for edges with dates needed for the answer.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_temporal_reasoning_hard = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: hard
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type TEMPORAL_REASONING: key dates — in events 3–6 (anti-primacy/recency bias).
- 40% stale edges with dates of cancelled or modified events.
- 15% invalid edges (valid: false) with incorrect dates.
- Add duplicate events with dates ±1 day — traps.
- The answer requires computing an interval between two events (not direct reading).

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

# INTERFERENCE

graph_gen_interference_easy = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: easy
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type INTERFERENCE: EXACTLY 2 similar events of the same relation type (e.g., two transfers).
- CRITICAL: difference between events is MINIMAL — amounts differ ≤500 rub., dates ≤5 days.
  Example: 4,800 rub. vs 5,000 rub., or 10.01.2024 vs 13.01.2024.
  FORBIDDEN: difference >500 rub. or >5 days — task becomes trivial.
- All edges are valid (valid: true).
- ground_truth: true — only for one target event.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "edge_transfer_1",
      "source": "person_1",
      "target": "account_2",
      "relation": "transferred",
      "attributes": {"amount": "5000 руб.", "date": "10.01.2024", "ref": "TXN001"},
      "valid": true,
      "stale": false,
      "ground_truth": false
    },
    {
      "id": "edge_transfer_2",
      "source": "person_1",
      "target": "account_2",
      "relation": "transferred",
      "attributes": {"amount": "5200 руб.", "date": "15.01.2024", "ref": "TXN002"},
      "valid": true,
      "stale": false,
      "ground_truth": true
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_interference_medium = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: medium
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type INTERFERENCE: EXACTLY 3 similar events with difference ≤200 rub./≤2 days between neighbors.
  CRITICAL: difference is MINIMAL — example: 4,800/5,000/5,200 rub. within one or two days.
  FORBIDDEN: difference >200 rub. or >2 days — task becomes trivial.
- 20% stale edges (stale: true) with similar parameters — additional traps.
- ground_truth: true — only for one target event.
- CRITICAL: target event — strictly the MIDDLE one of three (not the largest, not the smallest, not the first chronologically).

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_interference_hard = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: hard
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type INTERFERENCE: EXACTLY 3 similar events of the same relation type.
  Amounts/dates/names differ ≤200 rub./≤1 day/≤1 letter — make the difference MINIMAL.
  Example: transfers 4,900 rub., 5,000 rub., 5,100 rub. — all on the same day with 1-hour difference.
- Target (ground_truth: true) — strictly the middle one of three (not first and not last).
- Between the three similar events insert 2–3 unrelated edges of the same general type (distractors).
- 40% stale, 15% invalid edges.
- CRITICAL: all 3 similar events must have valid: true — none marked invalid.
  The model's task is to choose the needed one from three valid events, not filter by valid flag.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "transfer_1",
      "source": "person_1", "target": "account_ext_1",
      "relation": "transferred_to",
      "attributes": {"amount": "4 900 руб.", "date": "15.03.2024", "time": "10:00", "ref": "TXN-001"},
      "valid": true, "stale": false, "ground_truth": false
    },
    {
      "id": "transfer_2",
      "source": "person_1", "target": "account_ext_1",
      "relation": "transferred_to",
      "attributes": {"amount": "5 000 руб.", "date": "15.03.2024", "time": "14:00", "ref": "TXN-002"},
      "valid": true, "stale": false, "ground_truth": true
    },
    {
      "id": "transfer_3",
      "source": "person_1", "target": "account_ext_1",
      "relation": "transferred_to",
      "attributes": {"amount": "5 100 руб.", "date": "15.03.2024", "time": "18:00", "ref": "TXN-003"},
      "valid": true, "stale": false, "ground_truth": false
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

# COMPOSITE

graph_gen_composite_easy = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: easy
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type COMPOSITE: graph must contain an explicit cause-effect chain of 3+ steps.
- All edges are valid (valid: true).
- Each chain step — a separate GT-edge with field "chain_step": N in attributes (1, 2, 3...).
- GT-edges form a CONNECTED PATH: target of each GT-edge = source of the next GT-edge in the chain.
- GT-nodes: only those through which the target chain passes. No "floating" GT-nodes without GT-edges.
- A parallel FALSE chain (ground_truth: false) with ≥1 shared node with the target is mandatory.
- ground_truth: true — for edges and nodes of the target chain.
- ground_truth: false — for edges of the false chain.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [
    {
      "id": "chain_edge_1",
      "source": "event_1",
      "target": "event_2",
      "relation": "caused",
      "attributes": {"chain_step": 1, "description": "подача заявки на ипотеку"},
      "valid": true,
      "stale": false,
      "ground_truth": true
    }
  ]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_composite_medium = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: medium
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type COMPOSITE: target chain of 4+ steps + a parallel similar false chain.
- 20% stale edges (stale: true).
- GT-edges form a CONNECTED PATH: target of each GT-edge = source of the next in chain.
- Each GT-edge contains "chain_step": N in attributes (1, 2, 3, 4...).
- GT-nodes: only those through which the target chain passes. No "floating" GT-nodes.
- False chain has ≥1 shared node with target and a similar endpoint (creates interference).
- False chain starts from the same source as the target but leads to a different result.
- ground_truth: true — only for edges and nodes of the target chain.
- ground_truth: false — for edges of the false chain.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_gen_composite_hard = """
You are a knowledge graph constructor for banking scenarios. Build a graph from the given events.

TOPIC: <topic>
DIFFICULTY: hard
EVENTS: <events>

TASK: Build a knowledge graph of <num_nodes> nodes.
Edge density: <density_percent>%.

REQUIREMENTS:
- For type COMPOSITE: first chain step — events 1–3, last — events 5–7 (not batch 8).
- CRITICAL: EXACTLY 2 parallel false branches required (not one!).
  - False branch A: starts from same node as target, diverges after step 1.
  - False branch B: starts from same node, diverges after step 2 — goes deeper.
  - Both false branches lead to plausible but incorrect final facts.
  - ≥1 node of the target chain is used in both false branches (creates a fork).
- 40% stale, 15% invalid edges (valid: false).
- Duplicate fork-nodes (ground_truth: false) with plausible but false connections.
- Key nodes of the target chain — strictly in positions 3–6.
- GT-edges form a CONNECTED PATH: target of each GT-edge = source of the next in chain.
- Each GT-edge contains "chain_step": N in attributes (1, 2, 3...).
- False chain starts from same node as target but diverges after 1–2 steps.
- GT-nodes connected by GT-edges — no "floating" GT-nodes without GT-edges.

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [...],
  "edges": [...]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""


# GRAPH TASK GENERATION

GRAPH_ANTI_TRIVIALITY_BASE = """
═══════════════ ANTI-TRIVIALITY (MANDATORY) ═══════════════
FORBIDDEN to generate trivial tasks. Each task MUST:
1. Require GRAPH NAVIGATION — the answer cannot be given without traversing edges.
2. Be indirect — the question must NOT directly name the edge type or node id.
3. Avoid templates: NOT «What is the amount of edge_1?», NOT «What is in the date attribute?».
4. Answer — atomic fact (1–5 words): number, date, name, status.
5. Answer language — strictly Russian.
6. Question must paraphrase the fact through role, consequence or synonym.

AUTO-CHECK:
✓ Question contains ≥ 4 words?
✓ Key words of the answer are ABSENT from the question?
✓ Answer — fact (not a sentence)?
→ If any single checkbox fails — redo.

═══════════════ STRICT FORMAT OF "answer" FIELD (MANDATORY) ═══════════════
  Monetary amount  : digits WITHOUT spaces inside number + " руб."  →  "4500000 руб."
  Rate/percentage  : X.X% with DOT                                  →  "12.5%"
  Date             : DD.MM.YYYY                                      →  "15.01.2024"
  Interval         : N дней / N часов / N минут                     →  "5 дней"
  String/status    : exactly from graph attributes                   →  "одобрено"
  Chain            : "Node1 → Node2 → Node3"
FORBIDDEN: spaces inside number, "₽", "рублей", comma in decimals, explanations.
"""

GRAPH_ANTI_TRIVIALITY_EASY_DECOY = """
ADDITIONALLY (MANDATORY for EASY): Add to JSON the field "decoy_answers" with EXACTLY 3 wrong answers.
Each decoy belongs to a separate type:
1. Value from another node/edge of the same type (different object or different participant) — taken from the graph.
2. Similar number with a small difference: ±5–10% from the correct one (NOT ±50%!), so the decoy is plausible.
   Example: correct answer "14.5%" → decoy "13.5%" or "15.5%", NOT "20%" or "9%".
3. Correct concept, wrong context: different period (month instead of quarter), different unit (thousands rub.).
All 3 decoys must be present in graph attributes. FORBIDDEN: invented values, difference >15%.
"""

GRAPH_ANTI_TRIVIALITY_MEDIUM_DECOY = """
ADDITIONALLY (MANDATORY for MEDIUM): Add to JSON the field "decoy_answers" with EXACTLY 3 wrong answers.
Each decoy belongs to a separate type:
1. Stale value of the same parameter (from edge stale: true) — the most dangerous trap.
2. Value from an analogous neighboring node/event of the same type — close in meaning.
3. Correct number in wrong unit or time period.
All 3 decoys must be present in graph attributes.
CRITICAL: decoys must be HARD TO DISTINGUISH — difference from correct answer ≤10%.
FORBIDDEN: "noticeably different", obviously wrong values, invented numbers.
"""

GRAPH_ANTI_TRIVIALITY_BASE_EASY = GRAPH_ANTI_TRIVIALITY_BASE + GRAPH_ANTI_TRIVIALITY_EASY_DECOY
GRAPH_ANTI_TRIVIALITY_BASE_MEDIUM = GRAPH_ANTI_TRIVIALITY_BASE + GRAPH_ANTI_TRIVIALITY_MEDIUM_DECOY

GRAPH_ANTI_TRIVIALITY_HARD_EXTRA = """
COMPUTED ANSWER (MANDATORY for HARD): gold_answer must NEVER be a direct attribute of one node or edge.
The answer must require arithmetic or synthesis from 2+ graph elements:
- Sums: total of several edges, difference of two attributes of the same type
- Rates: difference between rates of two edges or result of a conditional recalculation
- Days: computed interval between dates of two different edges
FORBIDDEN: the answer is an attribute value of one edge/node, copied without any computation.

ADDITIONALLY (MANDATORY for HARD):
- Add to JSON the field "decoy_answers" with EXACTLY 3 wrong answers.
- Each decoy belongs to a separate trap type:
  1. Stale value — specific previous value of the same parameter from the graph.
  2. Value from an analogous neighboring event of the same type (different node, but same relation).
  3. Real value from the graph, close in magnitude to the correct answer (same type), but belonging to a different node or edge.
- CRITICAL: All 3 decoys must be present in graph attributes — not invented.
- CRITICAL: Decoys must look plausible — numeric value in same range,
  date in same month, rate with same precision.
- DECOY RANGE (MANDATORY):
  • Monetary amounts: each decoy differs from gold by NO MORE than 1,000 rub.
  • Percentage rates: each decoy differs from gold by NO MORE than 0.2%
  • Days/dates: each decoy differs from gold by NO MORE than 1 day
- FORBIDDEN: decoy whose difference from gold exceeds these limits.
"""

graph_task_information_extraction_easy = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: easy
GRAPH:
<graph_text>

TASK: Generate 1 task of type INFORMATION_EXTRACTION based on this graph.

RULES:
- Question asks about a SPECIFIC attribute of one node or one edge.
- CRITICAL: Fact — strictly from a node/edge with ground_truth: true (marked [★ key fact]).
- Answer — exact attribute value from the graph (number, date, string).
- Fill "source_node_ids" field — list of node/edge ids needed for the answer.
- Fill "reasoning_path" field — step-by-step path through the graph to the answer as edge strings.

""" + GRAPH_ANTI_TRIVIALITY_BASE_EASY + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with specific fact",
  "reasoning_path": [
    "Client --[owns]--> Account",
    "Account --[has_rate]--> Rate | value: 7.5%"
  ],
  "capability": "information_extraction",
  "difficulty": "easy",
  "source_node_ids": ["person_1", "account_1"],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_information_extraction_medium = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: medium
GRAPH:
<graph_text>

TASK: Generate 1 task of type INFORMATION_EXTRACTION based on this graph.

RULES:
- Graph contains stale edges (stale: true) — question must require finding the CURRENT value.
- CRITICAL: Answer taken only from a node/edge with ground_truth: true and stale: false.
- Question must require distinguishing current and stale values.
- VERBATIM BAN: question must NOT contain ≥2 consecutive words from graph labels.

""" + GRAPH_ANTI_TRIVIALITY_BASE_MEDIUM + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with specific fact",
  "reasoning_path": [
    "Client --[owns]--> Credit | rate: 14.5% (stale: true — outdated)",
    "Client --[owns]--> Credit | rate: 12.0% (ground_truth: true — current)"
  ],
  "capability": "information_extraction",
  "difficulty": "medium",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_information_extraction_hard = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: hard
GRAPH:
<graph_text>

TASK: Generate 1 task of type INFORMATION_EXTRACTION based on this graph.

RULES:
- Graph contains duplicate nodes (ground_truth: false) and invalid edges (valid: false).
- Key fact — strictly from a node/edge with ground_truth: true.
- FORBIDDEN to formulate question with ≥3 consecutive words from graph label text.
- FORBIDDEN to compute difference between dates or time intervals — that is temporal_reasoning.
  Compute ONLY numeric attributes: difference/sum of monetary amounts or percentage rates.

""" + GRAPH_ANTI_TRIVIALITY_BASE + GRAPH_ANTI_TRIVIALITY_HARD_EXTRA + """

FORMAT of reasoning_path — each element is ONE edge string:
  "<source-node> --[<relation>]--> <target-node> | <attribute>: <value>"

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with specific fact",
  "reasoning_path": [
    "Client --[owns]--> Mortgage | amount: 5000000 руб.",
    "Mortgage --[has_rate]--> Rate | value: 11.5%"
  ],
  "capability": "information_extraction",
  "difficulty": "hard",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_knowledge_update_easy = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: easy
GRAPH:
<graph_text>

TASK: Generate 1 task of type KNOWLEDGE_UPDATE based on this graph.

RULES:
- Question asks about the CURRENT (up-to-date) value of the parameter.
- CRITICAL: FORBIDDEN in question: «изменился», «новый», «последний», «итоговый»,
  «финальный», «актуальный», «текущий», «учитывая все изменения», «после изменений».
- Answer — value of edge with stale: false (last current one).
- All previous values (stale: true) — traps, not the answer.
- reasoning_path must show how to filter out stale edges.

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

ADDITIONALLY (MANDATORY — DECOY for KNOWLEDGE_UPDATE EASY):
Add "decoy_answers" field with EXACTLY 3 wrong answers.
All 3 decoys — STALE values from the same update chain in the graph:
1. Previous (stale) value — specifically from edge stale: true in the update chain.
2. Earlier value from the same chain (if chain ≥3 steps); otherwise — analogous parameter of another object.
3. Value of an adjacent parameter of the same object (e.g., another rate or limit from graph).
CRITICAL: All 3 decoys must be present in graph attributes. FORBIDDEN: invented values.

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with current parameter value",
  "reasoning_path": [
    "Client --[has_rate]--> Credit | value: 14.5% (stale: true — outdated)",
    "Client --[has_rate]--> Credit | value: 12.0% (ground_truth: true — current)"
  ],
  "capability": "knowledge_update",
  "difficulty": "easy",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_knowledge_update_medium = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: medium
GRAPH:
<graph_text>

TASK: Generate 1 task of type KNOWLEDGE_UPDATE based on this graph.

RULES:
- Parameter changed 3+ times — graph has multiple update edges (stale: true + 1 current).
- Question sounds neutral, no hint about existence of changes.
- CRITICAL: FORBIDDEN words: «теперь», «новый», «изменился», «после обновления», «сейчас»,
  «итоговый», «финальный», «актуальный», «текущий», «учитывая все изменения».
- Answer — final value (stale: false, ground_truth: true), literally in GT-edge attribute.
- FORBIDDEN: compute derivative values. Answer — exactly what is written in edge attribute.

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

ADDITIONALLY (MANDATORY — DECOY for KNOWLEDGE_UPDATE MEDIUM):
Add "decoy_answers" field with EXACTLY 3 wrong answers.
All 3 decoys — stale values from the same update chain (stale: true edges):
1. Second-to-last parameter value in the chain — most dangerous trap.
2. Earlier value from the same update chain.
3. Value of analogous parameter of another similar object from graph.
CRITICAL: All 3 decoys must be present in graph attributes.
CRITICAL: decoys HARD TO DISTINGUISH — difference from correct answer ≤15%.

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with current parameter value",
  "reasoning_path": [
    "Credit --[has_rate]--> Rate | value: 15.5% (stale: true — early)",
    "Credit --[has_rate]--> Rate | value: 14.0% (stale: true — intermediate)",
    "Credit --[has_rate]--> Rate | value: 12.5% (ground_truth: true — current)"
  ],
  "capability": "knowledge_update",
  "difficulty": "medium",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_knowledge_update_hard = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: hard
GRAPH:
<graph_text>

TASK: Generate 1 task of type KNOWLEDGE_UPDATE based on this graph.

RULES:
- Change pattern in graph: A→B→A→C (rollback and final change).
- Final value (C) — strictly in events 4–7, not in 8.
- Traps: stale B looks like "almost final".
- Forbidden ≥3 consecutive words from graph labels in question.
- CRITICAL: FORBIDDEN in question: «учитывая все изменения», «после всех изменений»,
  «после обновления», «итоговая», «финальная», «актуальная», «текущая».
  Example FORBIDDEN: «Какова итоговая ставка после всех изменений?»
  Example CORRECT: «Под какой процент оформлен ипотечный кредит клиента?»

""" + GRAPH_ANTI_TRIVIALITY_BASE + GRAPH_ANTI_TRIVIALITY_HARD_EXTRA + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with current parameter value",
  "reasoning_path": [
    "Credit --[has_rate]--> Rate | value: 15.0% (stale: true — A initial)",
    "Credit --[has_rate]--> Rate | value: 14.2% (stale: true — B change)",
    "Credit --[has_rate]--> Rate | value: 15.0% (stale: true — A rollback)",
    "Credit --[has_rate]--> Rate | value: 13.0% (ground_truth: true — C final)"
  ],
  "capability": "knowledge_update",
  "difficulty": "hard",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_temporal_reasoning_easy = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: easy
GRAPH:
<graph_text>

TASK: Generate 1 task of type TEMPORAL_REASONING based on this graph.

RULES:
- Question requires computing the time interval between two events (edges with dates).
- Both dates — in attributes of edges with ground_truth: true.
- Answer — computed result: «N дней», «N часов».
- In reasoning_path — step-by-step calculation.

""" + GRAPH_ANTI_TRIVIALITY_BASE_EASY + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "number and unit of measurement",
  "reasoning_path": [
    {"step": 1, "description": "find date of first event (edge submitted, 10.01.2024)"},
    {"step": 2, "description": "find date of second event (edge approved, 15.01.2024)"},
    {"step": 3, "description": "compute difference: 15 - 10 = 5 days"}
  ],
  "capability": "temporal_reasoning",
  "difficulty": "easy",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_temporal_reasoning_medium = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: medium
GRAPH:
<graph_text>

TASK: Generate 1 task of type TEMPORAL_REASONING based on this graph.

RULES:
- Graph has multiple dates (including stale ones). Question requires choosing the CORRECT dates.
- Stale dates (stale: true edges) — traps.
- Answer — computed interval based on current dates.
- In reasoning_path: explicit calculation «DD.MM.YYYY → DD.MM.YYYY = N».

""" + GRAPH_ANTI_TRIVIALITY_BASE_MEDIUM + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "number and unit of measurement",
  "reasoning_path": [...],
  "capability": "temporal_reasoning",
  "difficulty": "medium",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_temporal_reasoning_hard = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: hard
GRAPH:
<graph_text>

TASK: Generate 1 task of type TEMPORAL_REASONING based on this graph.

HARD DIFFICULTY REQUIREMENT (≥3 logical steps):
Simple "find date A and date B, subtract" — this is 2 steps, NOT HARD. Forbidden.

MANDATORY QUESTION TYPE (choose one of three):

TYPE 1 — EVENT CHAIN (A→B→C→answer):
  Step 1: find event A by indirect context.
  Step 2: from A traverse to event B (via edge), record its date.
  Step 3: from B traverse to event C (via edge), record its date.
  Answer: date of C OR total interval A→C.
  TRAP: graph has a similar event on the date of intermediate step B.

TYPE 2 — AGGREGATION WITH EXCLUSION (sum of intervals):
  Step 1: determine which events belong to the target process.
  Step 2: for each target event find its date.
  Step 3: compute SUM of two intervals, excluding non-target events.

TYPE 3 — CONDITIONAL DATE:
  Graph has an edge with relative offset attribute ("N days after...").
  Step 1: find source event X by its role.
  Step 2: find "N days after" attribute.
  Step 3: compute D_x + N days = date of Y.

RULES:
- GT-dates — strictly from edges/nodes with ground_truth: true.
- Invalid edges (valid: false) contain false dates.
- FORBIDDEN to name both events explicitly in question.
- reasoning_path must contain 4+ steps.

""" + GRAPH_ANTI_TRIVIALITY_BASE + GRAPH_ANTI_TRIVIALITY_HARD_EXTRA + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "number and unit of measurement, or date DD.MM.YYYY",
  "reasoning_path": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4 (calculation): ..."
  ],
  "capability": "temporal_reasoning",
  "difficulty": "hard",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_interference_easy = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: easy
GRAPH:
<graph_text>

TASK: Generate 1 task of type INTERFERENCE based on this graph.

STEP 1 — FIND THE INTERFERENCE GROUP:
Find 2 edges [★ key fact] with the same relation and close attributes (amounts ≤500 rub., dates ≤5 days).

STEP 2 — BUILD THE QUESTION:
- Question points to ONE specific edge via an indirect feature.
- FORBIDDEN: «первый», «второй», «последний», «из двух».
- Answer — attribute of ONLY the target edge (ground_truth: true).

STEP 3 — DECOY from interference group:
- All 3 decoys — attributes of ANOTHER edge from the same interference group.
- Same dimension, all present in graph attributes.

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with specific value",
  "reasoning_path": [...],
  "capability": "interference",
  "difficulty": "easy",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_interference_medium = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: medium
GRAPH:
<graph_text>

TASK: Generate 1 task of type INTERFERENCE based on this graph.

STEP 1: Find a group of 3+ edges [★ key fact] with same relation, one shared node,
and attributes differing in only one parameter.

STEP 2: Question points to ONE specific edge via indirect feature.
Target edge — NOT first and not last. FORBIDDEN: «первый», «последний».

STEP 3 — DECOY from interference group:
All 3 decoys — attributes of other edges in the same interference group.

""" + GRAPH_ANTI_TRIVIALITY_BASE_MEDIUM + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with specific value",
  "reasoning_path": [
    "Client --[paid]--> Credit | date: 15.03.2024, amount: 25000 руб. (target edge)",
    "Client --[paid]--> Credit | date: 20.03.2024, amount: 25000 руб. (similar — different date)",
    "Difference: payment date 15.03.2024 (target) vs 20.03.2024 (other)"
  ],
  "capability": "interference",
  "difficulty": "medium",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_interference_hard = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: hard
GRAPH:
<graph_text>

TASK: Generate 1 task of type INTERFERENCE based on this graph.

STEP 1: Find group of 3+ edges [★ key fact] with same relation, one shared source or target,
and attributes differing in only one parameter.

STEP 2: Question points to ONE edge via indirect feature.
FORBIDDEN: position words, ≥3 consecutive words from graph attributes.
Answer — attribute of ONLY the target edge (NOT first and not last in group).

STEP 3 — DECOY from interference group (MANDATORY PROXIMITY):
All 3 decoys — attributes of OTHER edges from the same interference group.
DECOY RANGE:
  • Monetary amounts: ≤1,000 rub. difference from gold
  • Rates: ≤0.2% difference from gold
  • Dates: ≤1 day difference from gold

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "string with specific value",
  "reasoning_path": [...],
  "capability": "interference",
  "difficulty": "hard",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_composite_easy = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: easy
GRAPH:
<graph_text>

TASK: Generate 1 task of type COMPOSITE based on this graph.

RULES:
- Question asks about a cause-effect relationship or outcome.
- FORBIDDEN: «Перечисли шаги», «Что произошло после X?», «Назови этапы».
- Answer — linking element or final fact of the target chain (ground_truth: true, 2–6 words).
- Fill answer_format: "Event1 → Event2 → Event3".
- reasoning_path — chain edge traversal.
- source_node_ids — all nodes of the target chain.

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

ADDITIONALLY (MANDATORY — DECOY for COMPOSITE EASY):
Add "decoy_answers" with EXACTLY 3 wrong answers from the false chain:
1. Final point of false chain — main trap.
2. Intermediate fact from divergence point of false chain.
3. Attribute of one false chain edge.
CRITICAL: All 3 from graph attributes. FORBIDDEN: invented values.

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "Event1 → Event2 → Event3",
  "reasoning_path": [
    {"step": 1, "description": "find starting node of chain"},
    {"step": 2, "description": "traverse caused edge to intermediate event"},
    {"step": 3, "description": "traverse resulted_in edge to final node"}
  ],
  "capability": "composite",
  "difficulty": "easy",
  "source_node_ids": ["event_1", "event_2", "event_3"],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_composite_medium = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: medium
GRAPH:
<graph_text>

TASK: Generate 1 task of type COMPOSITE based on this graph.

RULES:
- Graph has target chain (ground_truth: true) + false chain with ≥1 shared node.
- Question requires determining the correct path.
- Answer — final fact or linking element of the TARGET chain.

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

ADDITIONALLY (MANDATORY — DECOY for COMPOSITE MEDIUM):
Add "decoy_answers" with EXACTLY 3 wrong answers from the false chain:
1. Final node/fact of false chain — main trap.
2. Intermediate fact from divergence point.
3. Stale value from adjacent graph object.
CRITICAL: All 3 from graph attributes. HARD TO DISTINGUISH from correct answer.

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "Event1 → Event2 → Event3",
  "reasoning_path": [
    "EventA --[resulted_in]--> EventB | chain_step: 1",
    "EventB --[resulted_in]--> EventC | chain_step: 2",
    "EventC --[resulted_in]--> Result | chain_step: 3"
  ],
  "capability": "composite",
  "difficulty": "medium",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""

graph_task_composite_hard = """
You are a generator of episodic memory tasks based on a structured knowledge graph.

TOPIC: <topic>
DIFFICULTY: hard
GRAPH:
<graph_text>

TASK: Generate 1 task of type COMPOSITE based on this graph.

RULES:
- First chain step — nodes from events 1–3, last — from events 5–7.
- EXACTLY 2 parallel false branches with ≥1 shared element.
- Forbidden ≥3 consecutive words from graph attributes in question.

""" + GRAPH_ANTI_TRIVIALITY_BASE + """

ADDITIONALLY (MANDATORY — DECOY for COMPOSITE HARD):
Add "decoy_answers" with EXACTLY 3 wrong answers — strictly from false chains:
1. Final fact of False branch A — diverges from target after step 1.
2. Final fact of False branch B — diverges after step 2.
3. Intermediate node from intersection point of false and target chains.
CRITICAL: All 3 from graph attributes. Outwardly plausible, hard to distinguish without full traversal.

OUTPUT FORMAT (strictly JSON):
{
  "question": "...",
  "answer": "...",
  "answer_format": "Event1 → Event2 → Event3",
  "reasoning_path": [
    "EventA --[resulted_in]--> EventB | chain_step: 1",
    "EventB --[resulted_in]--> EventC | chain_step: 2",
    "EventC --[resulted_in]--> EventD | chain_step: 3",
    "EventD --[resulted_in]--> Result | chain_step: 4"
  ],
  "capability": "composite",
  "difficulty": "hard",
  "source_node_ids": [...],
  "decoy_answers": ["...", "...", "..."]
}

Generate ONLY JSON — no explanations or markdown blocks.
"""


graph_task_validation_prompt = """
Check a task based on a structured knowledge graph.

GRAPH:
<graph_text>

TASK:
Question: <question>
Answer: <answer>
Type: <task_type>
Difficulty: <difficulty>

Check against 5 criteria and reply strictly in JSON:

1. GROUNDING: Is the answer explicitly present in node or edge attributes of the graph?
2. ANSWERABILITY: Does the answer follow unambiguously from the graph taking valid/stale flags into account?
3. NON_TRIVIAL: Does the question require navigation across ≥2 edges or comparison of ≥2 nodes?
4. DIFFICULTY_FIT: Does the question match the stated difficulty <difficulty>?
   - easy:   direct fact, 1–2 steps
   - medium: requires analyzing stale/current edges without explicit hints
   - hard:   ≥3 navigation steps, decoy traps present
5. ADVERSARIAL: Is the question paraphrased (does not quote ≥3 consecutive words from graph labels)?

FAIL rule: verdict="FAIL" if any of criteria 1, 2, 3 is violated or difficulty_fit=false.

{
  "grounded": true/false,
  "answerable": true/false,
  "non_trivial": true/false,
  "difficulty_fit": true/false,
  "adversarial": true/false,
  "verdict": "PASS" / "FAIL",
  "reason": "brief explanation (1 sentence)"
}

Generate ONLY JSON.
"""

graph_grounding_validation_prompt = """
You are a knowledge graph reviewer for banking scenarios.

CONTEXT: the graph intentionally contains noise:
- stale edges (outdated connections) — intentionally added
- valid=false edges (unreliable connections) — intentionally added
- duplicate nodes with modified attributes — intentionally added
These elements are NOT hallucinations — they are part of the design.

CHECK ONLY ground_truth=true elements — they must accurately reflect the plan.

EVENT PLAN (source of truth):
<plan_text>

GROUND_TRUTH ELEMENTS (ground_truth=true) to check:
<gt_elements>

TASK: verify that each ground_truth element is grounded in the plan.

FAIL if in ground_truth elements:
- numbers (amounts, rates, dates) are NOT found in the plan — explicit hallucination
- person or organization names are completely invented (no matching word with plan)
- more than 2 gt-elements contain facts not in the plan

PASS if:
- key numbers and names in gt-elements are in the plan (paraphrasing acceptable)
- name in gt-element is a partial match — NOT a hallucination
- chain edges (composite) — dates and labels don't need to be literal, chain connectivity matters
- few gt-elements and verification is difficult — doubt in favor of PASS

Return ONLY JSON (no markdown):
{
  "verdict": "PASS" or "FAIL",
  "grounded_count": <number of correct gt-elements>,
  "total_count": <total gt-elements checked>,
  "hallucinated": ["brief description of issue, if any"],
  "reason": "1-2 sentences"
}
"""

GRAPH_SERIALIZATION_HEADER = "[STRUCTURED EVENT DATA]\n\nPARTICIPANTS AND OBJECTS:\n"

GRAPH_STRICT_ANSWER_FORMAT = """
═══════════════ STRICT FORMAT OF "answer" FIELD (MANDATORY) ═══════════════
The "answer" field must strictly follow the format:
  Monetary amount  : digits WITHOUT spaces inside number + " руб."  →  "4500000 руб."
  Rate/percentage  : X.X% with DOT as separator                     →  "12.5%"
  Date             : DD.MM.YYYY                                      →  "15.01.2024"
  Interval         : N дней / N часов / N минут                     →  "5 дней"
  String/status    : exactly from graph attributes                   →  "одобрено"
  Chain            : "Node1 → Node2 → Node3"

FORBIDDEN in "answer":
  ✗ spaces inside number: "4 500 000 руб." → write "4500000 руб."
  ✗ comma in decimals:    "12,5%"           → write "12.5%"
  ✗ symbol "₽" or word "рублей"            → write "руб."
  ✗ explanations, ranges, extra words
"""

GRAPH_EPISODE_INSTRUCTION = """
═══════════════ EPISODIC STRUCTURE (MANDATORY) ═══════════════
Input data is marked as EPISODES (=== EPISODE N ===).
Each episode is a separate time period. The graph MUST reflect this structure.

MANDATORY to create:
1. Episode nodes — one for each episode from input data:
   {"id": "ep_N", "type": "episode", "label": "Episode N",
    "attributes": {"order": N, "summary": "<1 phrase about key event>"},
    "ground_truth": false}

2. followed_by edges — episode chain in chronological order:
   {"id": "ep_chain_N", "source": "ep_N", "target": "ep_{N+1}",
    "relation": "followed_by", "attributes": {}, "valid": true, "stale": false, "ground_truth": false}

3. occurred_in edges — each key object/event is linked to its episode:
   {"id": "occ_X_N", "source": "<node_id>", "target": "ep_N",
    "relation": "occurred_in", "attributes": {}, "valid": true, "stale": false,
    "ground_truth": <true if this object contains a key fact, otherwise false>}

RULE: Each ground_truth node MUST have an occurred_in edge to its episode.
"""

GRAPH_GEN_ATTR_DECOY_MEDIUM = """
═══════════════ DECOY ATTRIBUTES (MANDATORY for MEDIUM) ═══════════════
Add at least 1 decoy node: an object of THE SAME TYPE as ground_truth,
with an attribute similar to the key fact (difference ≤10%).
Example: if ground_truth is a credit with rate 12.5%, add another credit with rate 12.0%.
Decoy: ground_truth: false, valid: true — a complete object with real attributes, not noise.
"""

GRAPH_GEN_ATTR_DECOY_HARD = """
═══════════════ DECOY ATTRIBUTES (MANDATORY for HARD) ═══════════════
Add at least 2 decoy nodes: objects of the same type as ground_truth,
with attributes differing ≤5% from the key fact.
Decoys: ground_truth: false, valid: true, stale: false — structurally indistinguishable from the original.
CRITICAL: decoys must have ALL the same fields as the original, only one value changes.
"""

GRAPH_GEN_PROMPTS = {
    ("information_extraction", "easy"):   graph_gen_information_extraction_easy   + GRAPH_EPISODE_INSTRUCTION,
    ("information_extraction", "medium"): graph_gen_information_extraction_medium + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_MEDIUM,
    ("information_extraction", "hard"):   graph_gen_information_extraction_hard   + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_HARD,
    ("knowledge_update", "easy"):         graph_gen_knowledge_update_easy         + GRAPH_EPISODE_INSTRUCTION,
    ("knowledge_update", "medium"):       graph_gen_knowledge_update_medium       + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_MEDIUM,
    ("knowledge_update", "hard"):         graph_gen_knowledge_update_hard         + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_HARD,
    ("temporal_reasoning", "easy"):       graph_gen_temporal_reasoning_easy       + GRAPH_EPISODE_INSTRUCTION,
    ("temporal_reasoning", "medium"):     graph_gen_temporal_reasoning_medium     + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_MEDIUM,
    ("temporal_reasoning", "hard"):       graph_gen_temporal_reasoning_hard       + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_HARD,
    ("interference", "easy"):             graph_gen_interference_easy             + GRAPH_EPISODE_INSTRUCTION,
    ("interference", "medium"):           graph_gen_interference_medium           + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_MEDIUM,
    ("interference", "hard"):             graph_gen_interference_hard             + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_HARD,
    ("composite", "easy"):                graph_gen_composite_easy               + GRAPH_EPISODE_INSTRUCTION,
    ("composite", "medium"):              graph_gen_composite_medium             + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_MEDIUM,
    ("composite", "hard"):                graph_gen_composite_hard               + GRAPH_EPISODE_INSTRUCTION + GRAPH_GEN_ATTR_DECOY_HARD,
}

graph_pad_prompt = """
You are a knowledge graph constructor for banking scenarios.
You have an existing graph and an event plan. The graph is too small — it needs to be EXPANDED.

EVENT PLAN:
<plan_text>

CURRENT GRAPH:
<current_graph>

TASK:
Add approximately <need_nodes> new nodes and <need_edges> new edges.

RULES:
- New nodes and edges must reflect real details from the event plan above.
- Do NOT duplicate already existing nodes (check by label and attributes).
- ground_truth: true — only for nodes/edges with key facts from the plan.
- occurred_in and followed_by — always ground_truth: false.
- Node attributes: specific values (amounts, dates, names, account numbers).
- Node types: person, account, product, event, organization, amount, document, rate.
- Edge types: owns, applied_for, transferred_to, changed_to, caused, resulted_in,
  had_rate, submitted, approved, rejected, paid, received, signed, cancelled, contains.

RETURN STRICTLY JSON:
{
  "nodes": [{"id": "...", "type": "...", "label": "...", "attributes": {}, "ground_truth": false}],
  "edges": [{"id": "...", "source": "...", "target": "...", "relation": "...", "attributes": {}, "valid": true, "stale": false, "ground_truth": false}]
}

Return ONLY new elements to add, not the entire graph.
"""

GRAPH_TASK_PROMPTS = {
    ("information_extraction", "easy"):   graph_task_information_extraction_easy,
    ("information_extraction", "medium"): graph_task_information_extraction_medium,
    ("information_extraction", "hard"):   graph_task_information_extraction_hard,
    ("knowledge_update", "easy"):         graph_task_knowledge_update_easy,
    ("knowledge_update", "medium"):       graph_task_knowledge_update_medium,
    ("knowledge_update", "hard"):         graph_task_knowledge_update_hard,
    ("temporal_reasoning", "easy"):       graph_task_temporal_reasoning_easy,
    ("temporal_reasoning", "medium"):     graph_task_temporal_reasoning_medium,
    ("temporal_reasoning", "hard"):       graph_task_temporal_reasoning_hard,
    ("interference", "easy"):             graph_task_interference_easy,
    ("interference", "medium"):           graph_task_interference_medium,
    ("interference", "hard"):             graph_task_interference_hard,
    ("composite", "easy"):                graph_task_composite_easy,
    ("composite", "medium"):              graph_task_composite_medium,
    ("composite", "hard"):                graph_task_composite_hard,
}


# SKELETON PROMPTS

_SKEL_FORMAT = """
═══ ATTRIBUTE STANDARD ═══
  Amount:  {"amount": "4500000 руб."}  — no spaces inside number
  Rate:    {"rate": "12.5%"}           — dot, not comma
  Date:    {"date": "15.01.2024"}      — DD.MM.YYYY strictly

OUTPUT FORMAT (strictly JSON, only this):
{
  "gt_nodes": [
    {
      "id": "person_1",
      "type": "person",
      "label": "Иван Петров",
      "attributes": {"income": "85000 руб.", "role": "client"},
      "ground_truth": true,
      "episode_hint": 1
    }
  ],
  "gt_edges": [
    {
      "id": "e_owns_1",
      "source": "person_1",
      "target": "account_1",
      "relation": "owns",
      "attributes": {"since": "15.01.2024"},
      "valid": true,
      "stale": false,
      "ground_truth": true,
      "episode_hint": 1
    }
  ]
}
Generate ONLY JSON.
"""

graph_skeleton_information_extraction_easy = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: easy
PLAN (fact source):
<events>

TASK: create 5–8 GT-elements — nodes and edges containing one key fact
(amount, rate, date or status) that will become the answer in the task.

RULES:
- Do NOT create non-GT elements — they will be added separately.
- episode_hint = episode number (1,2,3...) in which the event occurs.
- Key GT fact: direct attribute of one node or one edge.
- All numbers and dates — strictly from the plan.
- valid: true, stale: false for all edges.
""" + _SKEL_FORMAT

graph_skeleton_information_extraction_medium = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: medium
PLAN (fact source):
<events>

TASK: create 8–12 GT-elements. Key fact + stale version of the same parameter.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number from plan.
- MANDATORY: 2 GT-edges for one parameter:
  (1) stale: stale=true, ground_truth=false → contains old value
  (2) current: stale=false, ground_truth=true → contains new value (answer)
- All other GT-edges: valid=true, stale=false.
""" + _SKEL_FORMAT

graph_skeleton_information_extraction_hard = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: hard
PLAN (fact source):
<events>

TASK: create 12–18 GT-elements. Key fact strictly in episodes 3–6.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number (key fact — ONLY episode_hint 3, 4, 5 or 6).
- MANDATORY: 1–2 duplicate nodes (ground_truth=false) with NEAR-CORRECT value (±5%).
  Add duplicates in gt_nodes with ground_truth=false and episode_hint from episodes 1–2 or 7–8.
- Stale edges stale=true — traps with previous values.
""" + _SKEL_FORMAT

graph_skeleton_knowledge_update_easy = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: easy
PLAN (fact source):
<events>

TASK: create 5–8 GT-elements — parameter update chain (minimum 2 steps).

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number.
- MANDATORY 2 GT-update edges — BOTH placed in gt_edges with ground_truth=true:
  (1) old value: stale=true, ground_truth=true + fields from_value/to_value/date
  (2) new value:  stale=false, ground_truth=true  + fields value/since
- CRITICAL: stale=true ≠ ground_truth=false. Stale edges are also GT (change history).
- Both edges have episode_hint from different episodes.
""" + _SKEL_FORMAT

graph_skeleton_knowledge_update_medium = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: medium
PLAN (fact source):
<events>

TASK: create 8–12 GT-elements — 3-step update chain for a numeric parameter.

════ MANDATORY STRUCTURE: 3 changed_to edges (update chain) ════
gt_edges MUST contain EXACTLY THREE edges of this type:

STEP 1 — old value #1, stale=true (NO LONGER CURRENT):
  {"id": "e_update_step1", "source": "account_1", "target": "evt_state_1", "relation": "changed_to",
   "attributes": {"value": "15.0%", "date": "12.04.2024"}, "valid": true, "stale": true, "ground_truth": true, "episode_hint": 1}

STEP 2 — old value #2, stale=true (ALSO NOT CURRENT):
  {"id": "e_update_step2", "source": "account_1", "target": "evt_state_2", "relation": "changed_to",
   "attributes": {"value": "14.5%", "date": "19.04.2024"}, "valid": true, "stale": true, "ground_truth": true, "episode_hint": 2}

STEP 3 — FINAL CURRENT VALUE, stale=false ← CORRECT ANSWER TO TASK:
  {"id": "e_update_step3", "source": "account_1", "target": "evt_state_3", "relation": "changed_to",
   "attributes": {"value": "12.9%", "date": "05.06.2024"}, "valid": true, "stale": false, "ground_truth": true, "episode_hint": 4}

RULES:
- Do NOT create non-GT elements.
- Values, rates and dates — strictly from plan.
- CRITICAL: stale=true ≠ ground_truth=false. Stale GT-edges — change history, they ARE GT.
- episode_hint of final step is 2+ positions later than step 1.
""" + _SKEL_FORMAT

graph_skeleton_knowledge_update_hard = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: hard
PLAN (fact source):
<events>

TASK: create 12–18 GT-elements — pattern A→B→A→C (rollback and final change).

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number. Final value C — episode_hint 4, 5, 6 or 7 (not 8!).
- MANDATORY 4 update edges with dates — ALL placed in gt_edges with ground_truth=true:
  A (stale=true, ground_truth=true):  {"value": "15.0%", "date": "01.01.2024"} episode_hint=1
  B (stale=true, ground_truth=true):  {"value": "13.0%", "date": "01.02.2024"} episode_hint=2
  A (stale=true, ground_truth=true):  {"value": "15.0%", "date": "01.03.2024"} episode_hint=3
  C (stale=false, ground_truth=true): {"value": "12.5%", "date": "10.04.2024"} episode_hint=5 ← ANSWER
- CRITICAL: stale=true ≠ ground_truth=false. All 4 chain edges — GT.
""" + _SKEL_FORMAT

graph_skeleton_temporal_reasoning_easy = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: easy
PLAN (fact source):
<events>

TASK: create 5–8 GT-elements — two events with dates for interval calculation.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number.
- MANDATORY at least 2 GT-edges with "date" field in attributes (DD.MM.YYYY format).
- Both GT-edges: ground_truth=true, valid=true, stale=false.
- Edges in different episodes (episode_hint differ by 1+).
""" + _SKEL_FORMAT

graph_skeleton_temporal_reasoning_medium = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: medium
PLAN (fact source):
<events>

TASK: create 8–12 GT-elements — 4+ events with dates, including stale traps.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number.
- MANDATORY at least 4 GT-edges with "date":
  - 2 current (stale=false, gt=true) — answer computed from these
  - 2 stale (stale=true, gt=false) — traps with close dates (±1–3 days)
- Current edges: in different episodes (episode_hint differ by 2+).
""" + _SKEL_FORMAT

graph_skeleton_temporal_reasoning_hard = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: hard
PLAN (fact source):
<events>

TASK: create 12–18 GT-elements — key dates strictly in episodes 3–6.

RULES:
- Do NOT create non-GT elements.
- episode_hint of key dates = strictly 3, 4, 5 or 6 (anti-primacy/recency).
- MANDATORY at least 3 GT-edges with "date" (gt=true, stale=false) — for ≥3 calculation steps.
- MANDATORY 2+ GT-trap-edges (stale=true or gt=false) with dates ±1 day from correct ones.
""" + _SKEL_FORMAT

graph_skeleton_interference_easy = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: easy
PLAN (fact source):
<events>

TASK: create 5–8 GT-elements — EXACTLY 2 similar events of the same type.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number.
- MANDATORY 2 GT-edges with same relation:
  (1) ground_truth=false — similar event (trap)
  (2) ground_truth=true  — target event (answer)
- CRITICAL: amounts differ ≤500 rub., dates ≤5 days.
- Edges in different episodes.
""" + _SKEL_FORMAT

graph_skeleton_interference_medium = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: medium
PLAN (fact source):
<events>

TASK: create 8–12 GT-elements — EXACTLY 3 similar events, target strictly in the middle.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number (each event in its own).
- MANDATORY 3 GT-edges with same relation:
  (1) ground_truth=false — smaller value (trap)
  (2) ground_truth=true  — middle value (ANSWER)
  (3) ground_truth=false — larger value (trap)
- CRITICAL: difference between neighbors ≤200 rub./≤2 days.
- Target (gt=true) NOT first and NOT last chronologically.
""" + _SKEL_FORMAT

graph_skeleton_interference_hard = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: hard
PLAN (fact source):
<events>

TASK: create 12–18 GT-elements — triple interference, target strictly in the middle.

RULES:
- Do NOT create non-GT elements.
- episode_hint of key events = 3, 4, 5 or 6.
- MANDATORY 3 GT-edges of same relation:
  (1) ground_truth=false — minimum value
  (2) ground_truth=true  — middle value (ANSWER, episode_hint strictly between 1 and 3)
  (3) ground_truth=false — maximum value
- CRITICAL: difference between neighbors ≤200 rub./≤1 day.
- Additionally: 2+ distractor edges of same relation type (gt=false) from other episodes.
""" + _SKEL_FORMAT

graph_skeleton_composite_easy = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: easy
PLAN (fact source):
<events>

TASK: create 5–8 GT-elements — 3-step cause-effect chain + 1 false branch.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number.
- MANDATORY 3 GT-chain edges (chain_step in attributes):
  e1: source=A→target=B, chain_step=1, gt=true, episode_hint=1
  e2: source=B→target=C, chain_step=2, gt=true, episode_hint=2
  e3: source=C→target=D, chain_step=3, gt=true, episode_hint=3
  target of each = source of next (connected path!).
- MANDATORY 1 false branch edge (gt=false) from node B to a different target.
- Corresponding nodes A,B,C,D: gt=true.
""" + _SKEL_FORMAT

graph_skeleton_composite_medium = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: medium
PLAN (fact source):
<events>

TASK: create 8–12 GT-elements — 4-step chain + parallel false chain.

RULES:
- Do NOT create non-GT elements.
- episode_hint = episode number (chain must span ≥2 different episodes).
- MANDATORY 4 GT-chain edges (chain_step 1,2,3,4) — connected path A→B→C→D→E.
  chain_step MANDATORY in attributes of each edge.
  EXAMPLE: {"id":"edge_chain_1","source":"node_a","target":"node_b","relation":"caused",
           "attributes":{"chain_step":1,"amount":"500000 руб."},
           "valid":true,"stale":false,"ground_truth":true,"episode_hint":1}
- CRITICAL: edge i must have source = target of edge i-1.
- MANDATORY false chain (2 edges, gt=false) from same starting node.
- Chain spans episode_hint from 1 to 4+.
""" + _SKEL_FORMAT

graph_skeleton_composite_hard = """
You are a GT-skeleton constructor for a knowledge graph. STAGE 1 of 2: ground_truth=true elements only.

TOPIC: <topic> | DIFFICULTY: hard
PLAN (fact source):
<events>

TASK: create 12–18 GT-elements — chain of exactly 5 steps (connected path A→B→C→D→E→F).

RULES:
- Do NOT create non-GT elements.
- episode_hint: steps 1–2 in ep 1–3, steps 3–5 in ep 5–7 (last step NOT in ep 8!).
- MANDATORY exactly 5 GT-chain edges with chain_step in attributes:
  edge_chain_1: source=A, target=B, chain_step=1, episode_hint=1
  edge_chain_2: source=B, target=C, chain_step=2, episode_hint=2
  edge_chain_3: source=C, target=D, chain_step=3, episode_hint=5
  edge_chain_4: source=D, target=E, chain_step=4, episode_hint=6
  edge_chain_5: source=E, target=F, chain_step=5, episode_hint=7
  CRITICAL: each edge i must have source = target of edge i-1.
- chain_step MANDATORY in attributes of each chain edge.
- All nodes A,B,C,D,E,F: ground_truth=true.

EXAMPLE chain edge:
  {
    "id": "edge_chain_1",
    "source": "node_a",
    "target": "node_b",
    "relation": "caused",
    "attributes": {"chain_step": 1, "amount": "1500000 руб.", "date": "15.03.2023"},
    "valid": true, "stale": false, "ground_truth": true, "episode_hint": 2
  }
""" + _SKEL_FORMAT

GRAPH_SKELETON_PROMPTS = {
    ("information_extraction", "easy"):   graph_skeleton_information_extraction_easy,
    ("information_extraction", "medium"): graph_skeleton_information_extraction_medium,
    ("information_extraction", "hard"):   graph_skeleton_information_extraction_hard,
    ("knowledge_update", "easy"):         graph_skeleton_knowledge_update_easy,
    ("knowledge_update", "medium"):       graph_skeleton_knowledge_update_medium,
    ("knowledge_update", "hard"):         graph_skeleton_knowledge_update_hard,
    ("temporal_reasoning", "easy"):       graph_skeleton_temporal_reasoning_easy,
    ("temporal_reasoning", "medium"):     graph_skeleton_temporal_reasoning_medium,
    ("temporal_reasoning", "hard"):       graph_skeleton_temporal_reasoning_hard,
    ("interference", "easy"):             graph_skeleton_interference_easy,
    ("interference", "medium"):           graph_skeleton_interference_medium,
    ("interference", "hard"):             graph_skeleton_interference_hard,
    ("composite", "easy"):                graph_skeleton_composite_easy,
    ("composite", "medium"):              graph_skeleton_composite_medium,
    ("composite", "hard"):                graph_skeleton_composite_hard,
}

graph_fill_prompt = """
You are a knowledge graph fill constructor. STAGE 2 of 2: add non-GT context.

TOPIC: <topic> | DIFFICULTY: <difficulty>
EXISTING GT-SKELETON (do NOT change or duplicate it!):
<skeleton_text>

PLAN (for context):
<events>

TASK: add exactly <need_nodes> new nodes and <need_edges> new edges as background context.

CRITICAL RULES:
- ALL new elements: ground_truth=false
- Do NOT duplicate ids from skeleton (new ids: ctx_node_1, ctx_edge_1, etc.)
- Do NOT add episode nodes (ep_1, ep_2...) — they are added programmatically
- Edges can connect: skeleton nodes with each other, skeleton with new, new with new
- Nodes must have meaningful labels and non-empty attributes
- All new edges: valid=true, stale=false (noise is added programmatically after)
<difficulty_rules>

OUTPUT FORMAT (strictly JSON):
{
  "nodes": [
    {
      "id": "ctx_node_1",
      "type": "organization",
      "label": "JSC Sberbank",
      "attributes": {"type": "bank", "region": "Moscow"},
      "ground_truth": false
    }
  ],
  "edges": [
    {
      "id": "ctx_edge_1",
      "source": "person_1",
      "target": "ctx_node_1",
      "relation": "applied_for",
      "attributes": {"date": "10.01.2024", "channel": "онлайн"},
      "valid": true,
      "stale": false,
      "ground_truth": false
    }
  ]
}
Generate ONLY JSON.
"""

_FILL_RULES_EASY = ""
_FILL_RULES_MEDIUM = """
ADDITIONALLY (medium): create realistic context — documents, intermediate events,
additional accounts. Edges must connect objects through multiple steps (not directly).

CROSS-EPISODE CONNECTIONS (MANDATORY for medium): add at least 3 edges between nodes from DIFFERENT
episodes (caused, resulted_in, changed_to, approved). Without them the graph — disconnected clusters.
"""
_FILL_RULES_HARD = """
ADDITIONALLY (hard): create a MAXIMALLY DENSE graph. Density requirements:
- EACH new node must have MINIMUM 2–3 edges (no isolated nodes!)
- Create multiple alternative paths between main nodes
- Add organizations, documents, intermediaries, intermediate events
- Edges between skeleton GT-nodes — MANDATORY (link them through non-GT context)

CROSS-EPISODE CONNECTIONS (MANDATORY for hard): add at least 5 edges between nodes from DIFFERENT
episodes (caused, resulted_in, changed_to, approved). Without them composite tasks are impossible.

IMPORTANT: <need_edges> is a HARD MINIMUM. Create EXACTLY that many edges or more.
"""

GRAPH_FILL_RULES = {
    "easy":   _FILL_RULES_EASY,
    "medium": _FILL_RULES_MEDIUM,
    "hard":   _FILL_RULES_HARD,
}
