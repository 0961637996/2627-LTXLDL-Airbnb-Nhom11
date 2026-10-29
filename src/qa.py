import pandas as pd
import os, yaml, argparse

# 1. ĐỌC CẤU HÌNH & TẠO THƯ MỤC
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config, encoding="utf-8"))
snapshots = [str(x) for x in cfg["snapshots"]]

for d in ["data/processed", "data/samples", "reports"]: 
    os.makedirs(d, exist_ok=True)

QA_LOG = []

def log_qa(snap, table, rule, total, affected, action):
    QA_LOG.append({"snapshot": snap, "table": table, "rule": rule, 
                   "total_rows": total, "affected_rows": affected, "action": action})

# 2. ĐỌC FILE PHÂN CÔNG
df_list_assign = pd.read_csv("data/raw/phan_cong_1000_listing.csv", dtype=str).rename(columns={"listing_id": "id"})
list_ids = set(df_list_assign["id"].dropna())

df_rev_assign = pd.read_csv("data/raw/phan_cong_5000_review.csv", dtype=str).rename(columns={"review_id": "id"})
df_rev_assign = df_rev_assign.drop(columns=["listing_id"], errors="ignore") # Tránh trùng cột listing_id

# 3. XỬ LÝ TỪNG MỐC THỜI GIAN
for moc in snapshots:
    print(f"\n--- Đang xử lý lọc cho mốc: {moc} ---")
    
    # --- A. BẢNG LISTINGS ---
    df_l = pd.read_csv(f"data/raw/{moc}_listings.csv.gz", dtype={"id": str})
    n_raw_l = len(df_l)
    
    # Ghép bảng để lọc ID và lấy thêm cột analysis_area từ file phân công
    df_l = df_l.merge(df_list_assign, on="id", how="inner")
    
    # =====================================================================
    # THÊM MỚI: LÀM SẠCH GIÁ TIỀN (LISTINGS) TRƯỚC KHI LƯU
    # =====================================================================
    if "price" in df_l.columns:
        df_l["price"] = df_l["price"].replace(r"[\$,]", "", regex=True).astype(float)
    
    # QA Log cho Listings (Chỉ log việc lọc)
    log_qa(moc, "listings", "filter_1000_ids", n_raw_l, n_raw_l - len(df_l), "dropped_unassigned")
    df_l.to_csv(f"data/processed/{moc}_listings.csv", index=False)

    # =====================================================================
    # TỰ KIỂM TRA SAU KHI LỌC BẢNG LISTINGS (YÊU CẦU 4)
    # =====================================================================
    print(f"  [Tự kiểm tra QA] Mốc {moc}:")
    
    # 4.1 Kiểm tra loại phòng
    if "room_type" in df_l.columns:
        apt = (df_l["room_type"] == "Entire home/apt").sum()
        pri = (df_l["room_type"] == "Private room").sum()
        print(f"    - Entire home/apt: {apt}")
        print(f"    - Private room: {pri}")
    
    # 4.2 Kiểm tra number_of_reviews = 0
    if "number_of_reviews" in df_l.columns:
        # Ép kiểu an toàn sang số thực, điền -1 cho khoảng trống để đếm đúng số 0
        so_0_rev = (pd.to_numeric(df_l["number_of_reviews"], errors='coerce').fillna(-1) == 0).sum()
        print(f"    - number_of_reviews = 0: {so_0_rev}")

    # 4.3 Kiểm tra phân bố vùng (analysis_area)
    if "analysis_area" in df_l.columns:
        area_counts = df_l["analysis_area"].value_counts()
        all_valid = all(120 <= count <= 130 for count in area_counts)
        print(f"    - Vùng phân tích (120-130 listing/vùng): {all_valid}")
        if not all_valid:
            print(f"      Chi tiết lỗi phân bổ: {area_counts.to_dict()}")

    # 4.4 Kiểm tra thiếu giá (price) ở mốc cuối cùng
    if moc == snapshots[-1] and "price" in df_l.columns:
        thieu_gia = df_l["price"].isna().sum()
        print(f"    - Số listing thiếu price: {thieu_gia} ")
    # =====================================================================

    # --- B. BẢNG CALENDAR ---
    first = True
    n_raw_cal = 0
    n_filtered_cal = 0
    usecols = ["listing_id", "date", "available", "price"] if moc == snapshots[0] else ["listing_id", "date", "available"]
    
    for chunk in pd.read_csv(f"data/raw/{moc}_calendar.csv.gz", chunksize=200_000, usecols=usecols, dtype={"listing_id": str}):
        n_raw_cal += len(chunk)
        chunk = chunk[chunk["listing_id"].isin(list_ids)]
        n_filtered_cal += len(chunk)
        
        # =====================================================================
        # THÊM MỚI: LÀM SẠCH AVAILABLE VÀ PRICE (CALENDAR) TRƯỚC KHI LƯU
        # =====================================================================
        if "available" in chunk.columns:
            chunk["available"] = chunk["available"].map({'t': 1, 'f': 0, 'True': 1, 'False': 0, True: 1, False: 0})
        if "price" in chunk.columns:
            chunk["price"] = chunk["price"].replace(r"[\$,]", "", regex=True).astype(float)

        if not chunk.empty:
            chunk.to_csv(f"data/processed/{moc}_calendar.csv", mode="w" if first else "a", header=first, index=False)
            first = False
            
    # QA Log cho Calendar (Chỉ log việc lọc)
    log_qa(moc, "calendar", "filter_1000_ids", n_raw_cal, n_raw_cal - n_filtered_cal, "dropped_unassigned")

    # --- C. BẢNG REVIEWS & TÁCH MẪU ---
    first = True
    n_raw_rev = 0
    n_filtered_rev = 0
    df_samples = []
    
    for chunk in pd.read_csv(f"data/raw/{moc}_reviews.csv.gz", chunksize=100_000, dtype={"id": str, "listing_id": str}):
        n_raw_rev += len(chunk)
        chunk = chunk.merge(df_rev_assign, on="id", how="inner")
        n_filtered_rev += len(chunk)
        
        # =====================================================================
        # THÊM MỚI: XÓA THẺ HTML TRONG BÌNH LUẬN (REVIEWS) TRƯỚC KHI LƯU
        # =====================================================================
        if "comments" in chunk.columns:
            chunk["comments"] = chunk["comments"].replace(r"<[^>]+>", " ", regex=True)
            
        if not chunk.empty:
            chunk.to_csv(f"data/processed/{moc}_reviews.csv", mode="w" if first else "a", header=first, index=False)
            if moc == snapshots[0]: 
                df_samples.append(chunk)
            first = False
            
    # QA Log cho Reviews (Chỉ log việc lọc)
    log_qa(moc, "reviews", "filter_5000_ids", n_raw_rev, n_raw_rev - n_filtered_rev, "dropped_unassigned")

    # Tách mẫu LLM và Gán tay
    if moc == snapshots[0] and df_samples:
        pool = pd.concat(df_samples, ignore_index=True)
        if "mau_llm" in pool.columns and "nhan_tay" in pool.columns:
            pool[pool["mau_llm"] == "x"].to_csv("data/samples/llm_reviews.csv", index=False)
            pool[pool["nhan_tay"] == "x"].to_csv("data/samples/hand_labeled_reviews.csv", index=False)

# 4. XUẤT BÁO CÁO QA LỌC DỮ LIỆU
pd.DataFrame(QA_LOG).to_csv("reports/qa_report.csv", index=False)
print("\nHoàn tất việc lọc và làm sạch! Đã lưu báo cáo tại reports/qa_report.csv")