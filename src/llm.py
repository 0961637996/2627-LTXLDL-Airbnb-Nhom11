import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

# ==========================================
# 1. CẤU HÌNH
# ==========================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Chưa tìm thấy GEMINI_API_KEY trong file .env!")

INPUT_FILE = "data/samples/llm_reviews.csv"
OUTPUT_FILE = "data/samples/llm_results.csv"
MODEL_NAME = "gemini-3.5-flash-lite"
TEST_LIMIT = 10   # Đặt = 10 để test thử. Đổi thành None khi chạy hết cả file.
SAVE_EVERY = 20      # Tự động lưu sau mỗi 20 dòng
SLEEP_SECONDS = 6    # Nghỉ 6s/request để an toàn với giới hạn bản free

ALLOWED_ASPECTS = {"location", "host", "amenities", "cleanliness", "value", "noise"}
ALLOWED_SENTIMENTS = {"positive", "negative", "mixed"}
LLM_COLS = ["llm_sentiment", "llm_aspects_positive", "llm_aspects_negative", "llm_language"]

client = genai.Client(api_key=API_KEY)


# Khung cấu trúc JSON bắt buộc Gemini phải trả về
class ReviewAnalysis(BaseModel):
    sentiment: str
    aspects_positive: list[str]
    aspects_negative: list[str]
    language: str


# ==========================================
# 2. CÁC HÀM XỬ LÝ
# ==========================================
def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def filter_aspects(aspect_list) -> str:
    """Chỉ giữ lại các khía cạnh hợp lệ trong 6 từ khóa đã chốt và bỏ trùng lặp."""
    if not isinstance(aspect_list, list):
        return ""
    valid = [str(a).strip().lower() for a in aspect_list if str(a).strip().lower() in ALLOWED_ASPECTS]
    return ", ".join(dict.fromkeys(valid))


def save_df(df: pd.DataFrame, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def analyze_review(review_text: str) -> dict:
    cleaned = clean_text(review_text)
    prompt = f"""
You are an Airbnb review analyst. Analyze the review below and fill in the JSON fields:
- sentiment: strictly one of ["positive", "negative", "mixed"]. Use "mixed" if there is BOTH praise and criticism.
- aspects_positive: list of praised aspects, ONLY from ["location", "host", "amenities", "cleanliness", "value", "noise"]. Empty list [] if none.
- aspects_negative: list of criticized aspects, ONLY from ["location", "host", "amenities", "cleanliness", "value", "noise"]. Empty list [] if none.
- language: 2-letter ISO 639-1 code (e.g., "en", "es", "fr").
Rules:
- Only include an aspect if it is explicitly mentioned; DO NOT infer "host" or "value" from the host's name, generic praise, or general thanks.
- An aspect can appear in BOTH aspects_positive and aspects_negative if the guest both praises and criticizes it.
Review: "{cleaned}"
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReviewAnalysis,
                    temperature=0.0,
                ),
            )
            if not response.text:
                raise ValueError("API trả về kết quả rỗng.")

            data = json.loads(response.text)
            sentiment = str(data.get("sentiment", "")).strip().lower()

            return {
                "llm_sentiment": sentiment if sentiment in ALLOWED_SENTIMENTS else "unknown",
                "llm_aspects_positive": filter_aspects(data.get("aspects_positive", [])),
                "llm_aspects_negative": filter_aspects(data.get("aspects_negative", [])),
                "llm_language": str(data.get("language", "")).strip().lower(),
            }
        except Exception as e:
            err_msg = str(e)
            if any(k in err_msg for k in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                wait_time = 15 * (attempt + 1)
                print(f"\n[Server quá tải / Chạm giới hạn] Đợi {wait_time}s rồi thử lại (lần {attempt + 1}/3)...")
                time.sleep(wait_time)
            else:
                print(f"\n[Lỗi API]: {err_msg}")
                break

    return {
        "llm_sentiment": "error",
        "llm_aspects_positive": "",
        "llm_aspects_negative": "",
        "llm_language": "error",
    }


# ==========================================
# 3. LUỒNG CHẠY CHÍNH 
# ==========================================
def main():
    if os.path.exists(OUTPUT_FILE):
        print(f"Đọc file cũ để chạy tiếp: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE, dtype={"review_id": str, "listing_id": str}, encoding="utf-8-sig")
    else:
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE, dtype={"review_id": str, "listing_id": str}, encoding="utf-8")

    # Đảm bảo 4 cột đích luôn tồn tại và là kiểu chuỗi (tránh bug NaN -> float của Pandas)
    for col in LLM_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    processed_count = 0

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Gán nhãn LLM"):
            # Bỏ qua dòng đã có kết quả hợp lệ từ lần chạy trước
            if df.at[idx, "llm_sentiment"].strip() in ALLOWED_SENTIMENTS:
                continue

            # Bỏ qua dòng không có nội dung review
            if pd.isna(row["comments"]) or str(row["comments"]).strip() == "":
                continue

            result = analyze_review(row["comments"])
            for col, val in result.items():
                df.at[idx, col] = val

            processed_count += 1

            if processed_count % SAVE_EVERY == 0:
                save_df(df, OUTPUT_FILE)

            if TEST_LIMIT and processed_count >= TEST_LIMIT:
                print(f"\nĐã chạy đủ {TEST_LIMIT} dòng test. Dừng lại để kiểm tra.")
                break

            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\nĐã dừng thủ công (Ctrl+C). Đang lưu tiến độ hiện tại...")
    finally:
        if processed_count > 0:
            save_df(df, OUTPUT_FILE)
            print(f"Đã lưu {processed_count} dòng mới xử lý vào: {OUTPUT_FILE}")
        else:
            print("Không có dòng mới nào cần xử lý.")


if __name__ == "__main__":
    main()