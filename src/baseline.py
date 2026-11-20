import argparse
import json
import os
import re
import pandas as pd

# ============================================================ TỪ ĐIỂN SONG NGỮ (EN + ES)
ASPECT_KEYWORDS = {
    "location": [
        "location", "central", "center", "centre", "metro", "subway", "walk", "walking", "close to", "near", "neighborhood", "neighbourhood", "area", "transport",
        "ubicación", "ubicacion", "céntrico", "centrico", "centro", "cerca", "barrio", "zona", "transporte", "estación", "estacion", "caminando",
    ],
    "host": [
        "host", "owner", "communication", "responsive", "helpful", "friendly", "check-in", "checkin", "check in", "welcome", "welcoming", "kind",
        "anfitrión", "anfitrion", "anfitriona", "propietario", "comunicación", "comunicacion", "atento", "atenta", "amable", "llegada", "recepción", "recepcion", "trato",
    ],
    "amenities": [
        "wifi", "wi-fi", "internet", "kitchen", "bed", "beds", "shower", "bathroom", "ac", "a/c", "air conditioning", "heating", "elevator", "lift", "towels", "balcony", "amenities", "equipped",
        "cocina", "cama", "camas", "ducha", "baño", "bano", "aire acondicionado", "calefacción", "calefaccion", "ascensor", "toallas", "equipado", "equipada", "servicios",
    ],
    "cleanliness": [
        "clean", "cleanliness", "spotless", "tidy", "dirty", "dust", "dusty", "stain", "stains", "smell", "smelly", "hair", "mold", "mould", "cockroach", "bugs",
        "limpio", "limpia", "limpieza", "impecable", "sucio", "sucia", "polvo", "mancha", "manchas", "olor", "mal olor", "pelo", "pelos", "cucaracha",
    ],
    "value": [
        "value", "price", "priced", "expensive", "overpriced", "cheap", "worth", "money", "affordable", "cost",
        "precio", "caro", "cara", "barato", "barata", "económico", "economico", "calidad-precio", "calidad precio", "vale la pena", "dinero",
    ],
    "noise": [
        "noise", "noisy", "loud", "quiet", "peaceful", "silent", "sound", "walls", "street noise", "neighbors", "neighbours", "party", "traffic",
        "ruido", "ruidoso", "ruidosa", "tranquilo", "tranquila", "silencioso", "silenciosa", "vecinos", "calle", "fiesta", "tráfico", "trafico",
    ],
}

# Các từ riêng cho khía cạnh noise mang sẵn sắc thái
NOISE_NEG = {"noise", "noisy", "loud", "street noise", "party", "ruido", "ruidoso", "ruidosa", "fiesta"}
NOISE_POS = {"quiet", "peaceful", "silent", "tranquilo", "tranquila", "silencioso", "silenciosa"}

POS_WORDS = {
    "great", "good", "nice", "lovely", "amazing", "awesome", "excellent", "wonderful", "perfect", "clean", "spotless",
    "comfortable", "cozy", "friendly", "helpful", "recommend", "recommended", "best", "enjoyed", "fantastic", "quiet",
    "excelente", "bueno", "buena", "genial", "perfecto", "perfecta", "limpio", "limpia", "impecable", "cómodo", "comodo",
    "cómoda", "comoda", "amable", "atento", "atenta", "recomiendo", "recomendable", "encantó", "encanto", "tranquilo", "maravilloso",
}

NEG_WORDS = {
    "bad", "poor", "terrible", "horrible", "awful", "worst", "dirty", "noisy", "loud", "broken", "smell", "smelly",
    "rude", "uncomfortable", "cold", "hot", "small", "tiny", "expensive", "overpriced", "problem", "problems", "issue",
    "issues", "disappointed", "disappointing", "unfortunately", "never", "missing", "stain", "dust", "mold", "wait",
    "malo", "mala", "terrible", "horrible", "peor", "sucio", "sucia", "ruido", "ruidoso", "ruidosa", "roto", "rota",
    "olor", "incómodo", "incomodo", "frío", "frio", "pequeño", "pequeno", "caro", "cara", "problema", "problemas",
    "decepcionado", "decepcionante", "desafortunadamente", "lamentablemente", "nunca", "falta", "faltaba", "mancha", "polvo",
}

NEGATION_PATTERNS = [
    r"\bnot\s+(very\s+)?(good|great|clean|nice|comfortable|quiet|helpful|working|worth)",
    r"\bno\s+(estaba\s+)?(limpio|limpia|bueno|buena|funciona|funcionaba|había|habia|tiene|tenía|tenia|recomiendo)",
    r"\b(didn't|did not|doesn't|does not|wasn't|was not|weren't|were not)\s+\w+",
]

EN_STOP = {"the", "and", "was", "very", "with", "for", "this", "that", "from", "have", "were", "great", "place", "stay", "apartment", "room"}
ES_STOP = {"el", "la", "los", "las", "muy", "con", "para", "una", "que", "todo", "bien", "piso", "apartamento", "estancia", "excelente", "ubicación"}
ES_CHARS = set("áéíóúñü¿¡")


# ============================================================ CÁC HÀM PHÂN LOẠI
def detect_language(text):
    """Phân loại ngôn ngữ đơn giản về 2 giá trị: 'en' hoặc 'es'."""
    t = text.lower()
    words = set(re.findall(r"[a-zA-Záéíóúñü]+", t)) 
    es_score = len(words & ES_STOP) + (2 if any(ch in ES_CHARS for ch in t) else 0)
    en_score = len(words & EN_STOP)
    return "es" if es_score > en_score else "en"


def has_negation(clause):
    return any(re.search(pat, clause) for pat in NEGATION_PATTERNS)


def analyze_review(text):
    """Trả về (sentiment, aspects_positive, aspects_negative, language)."""
    if not isinstance(text, str) or not text.strip():
        return "positive", [], [], "en"

    lang = detect_language(text)
    t = text.lower()

    # Tách theo câu hoặc từ nối đối lập để bắt sắc thái từng vế
    clauses = re.split(r"[.!?;]+|\b(?:but|however|although|though|except|pero|aunque|sin embargo)\b", t)
    clauses = [c.strip() for c in clauses if c.strip()]

    pos_hits = 0
    neg_hits = 0
    asp_pos = set()
    asp_neg = set()

    for clause in clauses:
        words = set(re.findall(r"[a-zA-Záéíóúñü'-]+", clause))
        c_neg = len(words & NEG_WORDS) + (1 if has_negation(clause) else 0)
        c_pos = len(words & POS_WORDS)

        pos_hits += c_pos
        neg_hits += c_neg

        for aspect, keywords in ASPECT_KEYWORDS.items():
            matched = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", clause)]
            if not matched:
                continue

            # Xử lý riêng khía cạnh noise vì từ khóa tự mang nghĩa tích cực/tiêu cực
            if aspect == "noise":
                if any(kw in NOISE_NEG for kw in matched) and not re.search(r"\b(no|sin|not)\s+(noise|ruido)\b", clause):
                    asp_neg.add("noise")
                    neg_hits += 1
                elif any(kw in NOISE_POS for kw in matched) or re.search(r"\b(no|sin|not)\s+(noise|ruido)\b", clause):
                    asp_pos.add("noise")
                    pos_hits += 1
                continue

            if c_neg > 0:
                asp_neg.add(aspect)
            elif c_pos > 0:
                asp_pos.add(aspect)
            else:
                # Nhắc tới khía cạnh nhưng không có từ cảm xúc rõ -> tạm xếp theo xu hướng chung của review
                asp_pos.add(aspect)

    # Nếu 1 khía cạnh vừa bị chê vừa bị xếp nhầm sang positive do fallback, ưu tiên giữ ở negative
    asp_pos -= asp_neg

    # Quyết định sentiment: mixed / negative / positive
    if neg_hits > 0 and pos_hits > 0:
        sentiment = "mixed"
    elif neg_hits > 0 and pos_hits == 0:
        sentiment = "negative"
    else:
        sentiment = "positive"

    return sentiment, sorted(asp_pos), sorted(asp_neg), lang


INPUT_PATH = "data/samples/llm_reviews.csv"
OUTPUT_PATH = "data/processed/baseline_results.csv"


def main():
    df = pd.read_csv(INPUT_PATH, dtype=str)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        sent, pos_list, neg_list, lang = analyze_review(row.get("comments", ""))
        rows.append({
            "review_id": row["review_id"],
            "listing_id": row.get("listing_id", ""),
            "sentiment": sent,
            "aspects_positive": json.dumps(pos_list, ensure_ascii=False),
            "aspects_negative": json.dumps(neg_list, ensure_ascii=False),
            "language": lang,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[baseline] Đã xử lý {len(out_df)} dòng -> {OUTPUT_PATH}")
    print("\nPhân bố sentiment:\n", out_df["sentiment"].value_counts().to_string())
    print("\nPhân bố language:\n", out_df["language"].value_counts().to_string())


if __name__ == "__main__":
    main()