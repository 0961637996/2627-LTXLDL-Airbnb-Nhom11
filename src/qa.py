import numpy as np
import pandas as pd
import os, yaml, argparse

# 1. ĐỌC CẤU HÌNH & TẠO THƯ MỤC
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config, encoding="utf-8"))
snapshots = [str(x) for x in cfg["snapshots"]]

for d in ["data/processed", "data/samples", "reports"]: os.makedirs(d, exist_ok=True)
QA_LOG = []

def log_qa(snap, table, rule, total, affected, action):
    QA_LOG.append({"snapshot": snap, "table": table, "rule": rule, 
                   "total_rows": total, "affected_rows": affected, "action": action})

# 2. ĐỌC FILE PHÂN CÔNG
df_list_assign = pd.read_csv("data/raw/phan_cong_1000_listing.csv", dtype=str).rename(columns={"listing_id": "id"})
list_ids = set(df_list_assign["id"].dropna())

# Bỏ cột listing_id trong file phân công review để khi merge không bị sinh ra listing_id_x, listing_id_y
df_rev_assign = pd.read_csv("data/raw/phan_cong_5000_review.csv", dtype=str).rename(columns={"review_id": "id"})
df_rev_assign = df_rev_assign.drop(columns=["listing_id"], errors="ignore")

# 3. XỬ LÝ TỪNG MỐC THỜI GIAN
for moc in snapshots:
    print(f"--- Đang xử lý mốc: {moc} ---")
    
    # --- A. BẢNG LISTINGS (Đọc 1 lần vì file nhỏ) ---
    df_l = pd.read_csv(f"data/raw/{moc}_listings.csv.gz", dtype={"id": str})
    n_raw = len(df_l)
    df_l = df_l.merge(df_list_assign, on="id", how="inner") # Lọc 1000 ID
    
    if "price" in df_l.columns:
        df_l["price"] = df_l["price"].replace(r"[\$,]", "", regex=True).astype(float)
        bad_price = (df_l["price"] <= 0) | (df_l["price"].isna())
        df_l.loc[bad_price, "price"] = None
        log_qa(moc, "listings", "clean_price", n_raw, bad_price.sum(), "set_invalid_price_to_nan")

    log_qa(moc, "listings", "filter_assigned_ids", n_raw, n_raw - len(df_l), "dropped_unassigned")
    df_l.to_csv(f"data/processed/{moc}_listings.csv", index=False)

    # --- B. BẢNG CALENDAR (Đọc theo Chunk vì file rất nặng) ---
    first = True
    n_cal = 0
    usecols = ["listing_id", "date", "available", "price"] if moc == snapshots[0] else ["listing_id", "date", "available"]
    
    for chunk in pd.read_csv(f"data/raw/{moc}_calendar.csv.gz", chunksize=200_000, usecols=usecols, dtype={"listing_id": str}):
        chunk = chunk[chunk["listing_id"].isin(list_ids)].copy() # Lọc giữ lại lịch của 1000 ID
        if chunk.empty: continue
        
        n_cal += len(chunk)
        if "available" in chunk.columns:
            chunk["available"] = chunk["available"].map({'t': 1, 'f': 0, 'True': 1, 'False': 0, True: 1, False: 0})
        if "price" in chunk.columns:
            chunk["price"] = chunk["price"].replace(r"[\$,]", "", regex=True).astype(float)
            
        chunk.to_csv(f"data/processed/{moc}_calendar.csv", mode="w" if first else "a", header=first, index=False)
        first = False
    log_qa(moc, "calendar", "clean_available_and_price", n_cal, 0, "processed_chunks")

    # --- C. BẢNG REVIEWS & TÁCH MẪU ---
    first = True
    n_rev = 0
    df_samples = []
    
    for chunk in pd.read_csv(f"data/raw/{moc}_reviews.csv.gz", chunksize=100_000, dtype={"id": str, "listing_id": str}):
        chunk = chunk.merge(df_rev_assign, on="id", how="inner") # Lọc theo 5000 đánh giá
        if chunk.empty: continue
        
        n_rev += len(chunk)
        if "comments" in chunk.columns:
            chunk["comments"] = chunk["comments"].replace(r"<[^>]+>", " ", regex=True) # Xóa HTML
            chunk = chunk[chunk["comments"].str.strip() != ""] # Bỏ bình luận rỗng
        
        chunk.to_csv(f"data/processed/{moc}_reviews.csv", mode="w" if first else "a", header=first, index=False)
        if moc == snapshots[0]: df_samples.append(chunk) # Lưu trữ mốc đầu để tách mẫu
        first = False
    log_qa(moc, "reviews", "clean_html_and_empty", n_rev, 0, "processed_chunks")

    # Tách mẫu LLM và Gán tay (chỉ làm ở mốc đầu)
    if moc == snapshots[0] and df_samples:
        pool = pd.concat(df_samples, ignore_index=True)
        if "mau_llm" in pool.columns and "nhan_tay" in pool.columns:
            pool[pool["mau_llm"].str.lower() == "x"].to_csv("data/samples/llm_reviews.csv", index=False)
            pool[pool["nhan_tay"].str.lower() == "x"].to_csv("data/samples/hand_labeled_reviews.csv", index=False)

# 4. XUẤT BÁO CÁO
pd.DataFrame(QA_LOG).to_csv("reports/qa_report.csv", index=False)
print("\nHoàn tất! Đã lưu báo cáo tại reports/qa_report.csv")