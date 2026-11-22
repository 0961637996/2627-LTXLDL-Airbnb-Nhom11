import os
import pandas as pd

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN FILE
# ==========================================
HAND_FILE = "data/samples/hand_labeled_reviews.csv"
BASE_FILE = "data/processed/baseline_results.csv"   # Chưa có file này code tự bỏ qua
LLM_FILE  = "data/processed/llm_results.csv"    # Chạy xong 1000 dòng thì đổi tên file ở đây

# ==========================================
# 2. CÁC HÀM CHẤM ĐIỂM
# ==========================================
def clean(series):
    return series.fillna("").astype(str).str.strip().str.lower()


def to_set(text):
    if pd.isna(text) or not str(text).strip():
        return set()
    return {x.strip().lower() for x in str(text).split(",") if x.strip()}


def score_aspects(df, true_col, pred_col):
    exact = tp = fp = fn = 0
    for t, p in zip(df[true_col], df[pred_col]):
        t_set, p_set = to_set(t), to_set(p)
        if t_set == p_set:
            exact += 1
        tp += len(t_set & p_set)  # Đoán đúng
        fp += len(p_set - t_set)  # Đoán thừa
        fn += len(t_set - p_set)  # Đoán thiếu

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return exact / len(df), prec, rec, f1


def evaluate_one(df, prefix):
    scores = {}
    scores["Sentiment Accuracy"] = (clean(df["sentiment"]) == clean(df[f"{prefix}sentiment"])).mean()

    if "language" in df.columns and f"{prefix}language" in df.columns:
        scores["Language Accuracy"] = (clean(df["language"]) == clean(df[f"{prefix}language"])).mean()

    for col, tag in [("aspects_positive", "Pos"), ("aspects_negative", "Neg")]:
        if f"{prefix}{col}" in df.columns:
            ex, p, r, f1 = score_aspects(df, col, f"{prefix}{col}")
            scores[f"Aspects {tag} - Exact"] = ex
            scores[f"Aspects {tag} - Precision"] = p
            scores[f"Aspects {tag} - Recall"] = r
            scores[f"Aspects {tag} - F1"] = f1
    return scores


# ==========================================
# 3. GỘP BẢNG TỪ CÁC FILE ĐANG CÓ
# ==========================================
valid = ["positive", "negative", "mixed"]

df = pd.read_csv(HAND_FILE, dtype={"review_id": str}, encoding="utf-8-sig")
df = df[clean(df["sentiment"]).isin(valid)]
models = {}

if os.path.exists(BASE_FILE):
    df_base = pd.read_csv(BASE_FILE, dtype={"review_id": str}, encoding="utf-8-sig")
    cols = ["review_id"] + [c for c in df_base.columns if c.startswith("base_")]
    df = df.merge(df_base[cols], on="review_id", how="inner")
    models["Baseline"] = "base_"

if os.path.exists(LLM_FILE):
    df_llm = pd.read_csv(LLM_FILE, dtype={"review_id": str}, encoding="utf-8-sig")
    df_llm = df_llm[clean(df_llm["llm_sentiment"]).isin(valid)]  # Chỉ lấy các dòng đã chạy xong
    cols = ["review_id"] + [c for c in df_llm.columns if c.startswith("llm_")]
    df = df.merge(df_llm[cols], on="review_id", how="inner")
    models["LLM"] = "llm_"

if not models or df.empty:
    print("Chưa có file dự đoán nào (hoặc không có id nào khớp) để chấm!")
    exit()


# ==========================================
# 4. IN BẢNG ĐIỂM & LƯU KẾT QUẢ
# ==========================================
print(f"=== KẾT QUẢ ĐÁNH GIÁ TRÊN {len(df)} DÒNG ===\n")

summary = pd.DataFrame({name: evaluate_one(df, prefix) for name, prefix in models.items()})
if "Baseline" in summary and "LLM" in summary:
    summary["LLM - Baseline"] = summary["LLM"] - summary["Baseline"]

out = summary.copy()
for col in out.columns:
    fmt = "{:+.2%}" if col == "LLM - Baseline" else "{:.2%}"
    out[col] = out[col].map(lambda x, f=fmt: "N/A" if pd.isna(x) else f.format(x))

print(out.to_string(), "\n")

for name, prefix in models.items():
    print(f"--- Confusion Matrix: {name} ---")
    print(pd.crosstab(clean(df["sentiment"]), clean(df[f"{prefix}sentiment"]), rownames=["Tay"], colnames=[name]), "\n")

os.makedirs("reports", exist_ok=True)
out.to_csv("reports/eval_summary.csv", encoding="utf-8-sig")
df.to_csv("reports/eval_merged.csv", index=False, encoding="utf-8-sig")
print("Đã lưu reports/eval_summary.csv và reports/eval_merged.csv")