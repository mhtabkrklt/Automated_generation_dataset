import random
import json
from typing import Optional
import numpy as np
import networkx as nx
from faker import Faker
from llm import gemini_base

fake = Faker('ru_RU')

FINANCIAL_ARCHETYPES = {
    "Conservative_Saver": "Этот человек — 'Консервативный сберегатель'. Он избегает рисков, предпочитает надежные вклады и накопительные счета. Боится кредитов, всегда имеет подушку безопасности. Тщательно проверяет все комиссии. В диалоге может быть недоверчивым и дотошным, задает много вопросов о гарантиях и страховании вкладов.",
    "Impulsive_Spender": "Этот человек — 'Импульсивный потребитель'. Часто живет не по средствам, активно пользуется кредитными картами и рассрочками. Склонен совершать эмоциональные покупки. В общении с банком его интересуют быстрые деньги, увеличение лимитов и грейс-периоды. Может забывать о датах платежей.",
    "Strategic_Investor": "Этот человек — 'Стратегический инвестор'. Хорошо разбирается в финансах, ищет максимальную доходность. Использует ИИС, брокерские счета, кэшбэк-сервисы. В диалоге использует профессиональные термины, вежлив, но требователен к условиям обслуживания и качеству приложения.",
    "Anxious_Debtor": "Этот человек — 'Тревожный заемщик'. Имеет высокую долговую нагрузку (ипотека или несколько кредитов). Очень переживает из-за просрочек или изменений ставок. Любая новость от банка вызывает у него стресс. Ищет способы рефинансирования или снижения платежа.",
    "Digital_Native": "Этот человек — 'Цифровой пользователь'. Делает всё через приложение, ненавидит ходить в офисы. Активно пользуется СБП, подписками, экосистемой банка. При возникновении технических сбоев быстро раздражается. Ценит скорость и удобство интерфейса.",
    "Cash_Traditionalist": "Этот человек — 'Традиционалист'. Предпочитает наличные, с трудом доверяет цифровым сервисам. Часто посещает отделения, любит бумажные договоры с печатями. Может задавать наивные вопросы о том, как работает мобильный банк или переводы."
}

def extract_profile_from_events(events: list) -> dict:
    import re
    result = {}

    fio_pattern = r'[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+'
    for event in events:
        match = re.search(fio_pattern, event)
        if match:
            result['name'] = match.group(0)
            break

    bank_keywords = ['банк', 'bank']
    for event in events:
        company_match = re.search(r'(?:ООО|АО|ЗАО)\s*[«"]([^»"]+)[»"]', event)
        if company_match:
            company_name = company_match.group(1)
            if not any(kw in company_name.lower() for kw in bank_keywords):
                result['employer'] = company_match.group(0)
                break

    return result

def drop_by_id(char_list: list[dict], target_id: int) -> None:
    char_list[:] = [c for c in char_list if c["id"] != target_id]

def pick_archetype(k=1):
    keys = list(FINANCIAL_ARCHETYPES.keys())
    subset = random.sample(keys, k)
    archetype_desc = ""
    for type_key in subset:
        archetype_desc += f"{type_key}: {FINANCIAL_ARCHETYPES[type_key]}\n\n"
    return archetype_desc

def random_name(gender: str) -> str:
    if gender == "male":
        return fake.first_name_male()
    else:
        return fake.first_name_female()

def make_profile_spec():
    gender = random.choice(["male", "female"])
    if gender == "male":
        name = fake.name_male()
    else:
        name = fake.name_female()

    income = random.randint(30, 350) * 1000

    spec = {
        "name": name,
        "gender": gender,
        "age": random.randint(20, 70),
        "location": fake.city(),
        "job_title": fake.job(),
        "monthly_income": f"{income} рублей",
        "financial_archetype": pick_archetype(k=1),
    }
    return spec

SYSTEM_BIO_PROMPT = """
Ты — эксперт по профилированию банковских клиентов.
Твоя задача — создать живое, реалистичное описание человека на основе сухих данных.

ВХОДНЫЕ ДАННЫЕ: Имя, возраст, профессия, доход, финансовый архетип.

ЗАДАЧА: Напиши убедительный портрет этого человека (100-150 слов) на РУССКОМ языке.
Опиши его отношение к деньгам, привычки тратить или копить, и то, как он обычно общается с поддержкой банка.
Используй указанный "Финансовый Архетип" как основу, но не называй его прямо. Покажи это через поведение.

СТИЛЬ: Профессиональный, психологический портрет.
"""

def generate_bio(spec, model=None):
    if model is None:
        model = gemini_base
    user_prompt = "АНКЕТА КЛИЕНТА:\n" + json.dumps(spec, indent=2, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_BIO_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = model.invoke(messages).content
    return response

def _apply_name_collision(relationships: dict) -> None:
    partner_list = relationships.get("partner", [])
    children_list = relationships.get("children", [])
    parent_list = relationships.get("parent", [])

    if partner_list and children_list:
        partner_first = partner_list[0]["name"].split()[0]
        partner_gender = partner_list[0].get("gender", "")
        target_child = next(
            (c for c in children_list if c.get("gender") == partner_gender),
            children_list[0]
        )
        target_child["name"] = partner_first
        return

    if len(children_list) >= 2:
        first_name = children_list[0]["name"].split()[0]
        children_list[1]["name"] = first_name
        return

    if parent_list and children_list:
        parent_first = parent_list[0]["name"].split()[0]
        children_list[0]["name"] = parent_first

def create_profile(friends_size: int = 3,
                   acquaintances_size: int = 5,
                   events_overrides: Optional[dict] = None,
                   name_collision_prob: float = 0.0):

    all_profile_size = friends_size + acquaintances_size + 5
    main_spec = make_profile_spec()

    if events_overrides:
        if 'name' in events_overrides:
            main_spec['name'] = events_overrides['name']
        if 'employer' in events_overrides:
            main_spec['job_title'] = f"сотрудник {events_overrides['employer']}"

    main_spec["bio"] = generate_bio(main_spec)
    main_spec["personality_traits"] = main_spec["bio"]

    relationships = {}

    main_spec["id"] = 0
    main_age = main_spec["age"]
    main_gender = main_spec["gender"]

    chars_other = []
    ages = np.random.randint(10, 85, size=all_profile_size)

    for i, age in enumerate(ages):
        gender = random.choice(["male", "female"])
        name = fake.name_male() if gender == "male" else fake.name_female()
        chars_other.append({
            "id": i + 1,
            "gender": gender,
            "name": name,
            "age": int(age)
        })

    chars = [main_spec] + chars_other

    G = nx.Graph()
    for c in chars:
        G.add_node(c["id"], **c)

    def add_edge(a, b, rel):
        if G.has_edge(a, b):
            if "type" in G[a][b]:
                G[a][b]["type"].add(rel)
            else:
                G[a][b]["type"] = {rel}
        else:
            G.add_edge(a, b, type={rel})

    MAIN_ID = 0

    parents = []
    if random.random() < 0.85:
        num_parents = random.choice([1, 2])

        for role in ["mother", "father"][:num_parents]:
            gender = "female" if role == "mother" else "male"

            pool = [p for p in chars
                    if p["gender"] == gender
                    and (main_age + 18) <= p["age"] <= (main_age + 45)
                    and p["id"] != MAIN_ID]

            if pool:
                p = random.choice(pool)
            else:
                p = {
                    "id": len(chars),
                    "gender": gender,
                    "name": random_name(gender) + " " + fake.last_name(),
                    "age": main_age + random.randint(20, 40)
                }
                G.add_node(p["id"], **p)
                chars.append(p)  # добавляем в общий список, чтобы ID не плыли

            add_edge(MAIN_ID, p["id"], "parent")
            parents.append(p)
            drop_by_id(chars, p["id"])

        relationships["parent"] = parents

    partner_rel = []
    if main_age >= 18 and random.random() < (0.6 if main_age < 30 else 0.8):
        partner_pool = [
            q for q in chars[1:]
            if q["gender"] != main_gender
               and abs(q["age"] - main_age) <= 10
               and q["age"] >= 18
               and not G.has_edge(MAIN_ID, q["id"])
        ]

        if partner_pool:
            partner = random.choice(partner_pool)
        else:
            p_gender = "male" if main_gender == "female" else "female"
            partner = {
                "id": len(chars),
                "gender": p_gender,
                "name": random_name(p_gender) + " " + fake.last_name(),
                "age": main_age + random.randint(-5, 5)
            }
            G.add_node(partner["id"], **partner)
            chars.append(partner)

        partner_rel.append(partner)
        add_edge(MAIN_ID, partner["id"], "partner")
        drop_by_id(chars, partner["id"])
        relationships["partner"] = partner_rel

    children = []
    has_children_prob = np.interp(main_age, [20, 30, 45, 80], [0, 0.5, 0.8, 0.8])
    if main_age >= 22 and random.random() < has_children_prob:
        num_kids = random.choice([1, 1, 2, 2, 3])
        for _ in range(num_kids):
            max_kid_age = main_age - 18
            if max_kid_age < 0:
                break

            kid_age = random.randint(0, max_kid_age)
            kid_gender = random.choice(["male", "female"])
            kid = {
                "id": len(chars),
                "gender": kid_gender,
                "name": random_name(kid_gender),  # фамилия как у героя подразумевается
                "age": kid_age
            }
            G.add_node(kid["id"], **kid)
            chars.append(kid)

            add_edge(MAIN_ID, kid["id"], "child")
            children.append(kid)

        relationships["children"] = children

    friends: list[dict] = []
    potential_friends = [p for p in chars[1:] if
                         p["id"] not in [rel["id"] for rel_list in relationships.values() for rel in rel_list]]

    for person in potential_friends:
        if abs(person["age"] - main_age) <= 10 and random.random() < 0.25:
            if len(friends) < friends_size:
                add_edge(MAIN_ID, person["id"], "friend")
                drop_by_id(chars, person["id"])
                friends.append(person)

    relationships["friends"] = friends

    acquaintances: list[dict] = []
    potential_acquaintances = [p for p in chars[1:] if not G.has_edge(MAIN_ID, p["id"])]

    for person in potential_acquaintances:
        if random.random() < 0.15:
            if len(acquaintances) < acquaintances_size:
                add_edge(MAIN_ID, person["id"], "acquaintance")
                acquaintances.append(person)

    relationships["acquaintances"] = acquaintances

    if random.random() < name_collision_prob:
        _apply_name_collision(relationships)

    return main_spec, relationships

if __name__ == "__main__":
    import os

    try:
        spec, rels = create_profile(2, 2)
        print("Главный герой:", json.dumps(spec, indent=2, ensure_ascii=False))
        print("Связи:", json.dumps(rels, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Ошибка при тесте: {e}")
        print("Проверьте, настроен ли API ключ в src/llm.py или переменных среды.")
