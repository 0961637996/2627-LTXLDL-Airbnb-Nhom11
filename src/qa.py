# import pandas as pd
# import os
# import argparse
# import yaml

# parser = argparse.ArgumentParser()
# parser.add_argument("--config",type = str,default="configs/madrid.yaml")
# args = parser.parse_args()

# with open(args.config,"r",encoding="utf-8") as file:
#      config = yaml.safe_load(file)

# danh_sach_moc=[str(a) for a in config["snapshots"]]

# os.makedirs("data/processed", exist_ok=True)
# os.makedirs("data/samples", exist_ok=True)

# df1000 = pd.read_csv(
#     "data/raw/phan_cong_1000_listing.csv",
#     dtype={"listing_id":"str"},
# )
# df1000 = df1000.rename(columns={"listing_id":"id"})
# id_can_tim = set(df1000["id"].dropna())

# df5000 = pd.read_csv(
#     "data/raw/phan_cong_5000_review.csv",
#     dtype={"listing_id":"str","review_id":"str"},
# )
# df5000 = df5000.rename(columns={"review_id":"id"})

# for moc in danh_sach_moc:
#     first_chunk_listings=True
#     for chunk in pd.read_csv(
#         f"data/raw/{moc}_listings.csv.gz",
#         chunksize = 100_000,
#         dtype={"id":"str"}
#     ):
#         ket_qua1=chunk.merge(
#             df1000,
#             on="id",
#             how ="inner"
#         )
#         if not ket_qua1.empty:
#             ket_qua1.to_csv(
#                 f"data/processed/{moc}_listings.csv",
#                 mode = "w" if first_chunk_listings else "a",
#                 header = first_chunk_listings,
#                 index= False
#             )
#             first_chunk_listings = False

# for moc in danh_sach_moc:
#     first_chunk_calendar=True
#     if moc == "2025-09-14":
#         cols = ["listing_id","date","available","price"]
#     else:
#         cols = ["listing_id","date","available"]
#     for chunk in pd.read_csv(
#         f"data/raw/{moc}_calendar.csv.gz",
#         chunksize=100_000,
#         usecols=cols,
#         dtype ={"listing_id":"str"},
#     ):
#         ket_qua2=chunk[chunk["listing_id"].isin(id_can_tim)]
#         if not ket_qua2.empty:
#             ket_qua2.to_csv(
#                 f"data/processed/{moc}_calendar.csv",
#                 mode ="w" if first_chunk_calendar else "a",
#                 header = first_chunk_calendar,
#                 index = False,
#             )
#             first_chunk_calendar = False

# for moc in danh_sach_moc:
#     first_chunk_reviews=True
#     for chunk in pd.read_csv(
#         f"data/raw/{moc}_reviews.csv.gz",
#         chunksize=100_000,
#         dtype ={"id":"str","listing_id":"str"},
#     ):
#         ket_qua3 = chunk.merge(
#             df5000,
#             on="id",
#             how="inner",
#         )
#         if not ket_qua3.empty:
#             ket_qua3.to_csv(
#                 f"data/processed/{moc}_reviews.csv",
#                 mode ="w" if first_chunk_reviews else "a",
#                 header = first_chunk_reviews,
#                 index = False,
#             )
#             first_chunk_reviews = False
#     if moc == danh_sach_moc[0]:
#         read = pd.read_csv(
#             f"data/processed/{moc}_reviews.csv",
#             dtype = str,
#         )
#         read1= read[read["mau_llm"]=="x"]
#         read2= read[read["nhan_tay"]=="x"]
#         read1.to_csv(
#             "data/samples/llm_reviews.csv",
#             index= False
#         )
#         read2.to_csv(
#             "data/samples/hand_labeled_reviews.csv",
#             index= False
#         )


# for moc in danh_sach_moc:
#     print(moc)
#     check =  pd.read_csv(f"data/processed/{moc}_listings.csv")

#     room_type = check["room_type"].value_counts()
#     apt = room_type.get('Entire home/apt', 0)
#     pri = room_type.get('Private room', 0)
#     print(f"Số lượng apt là {apt}, số lượng pri là {pri}")

#     so_0_review = (check["number_of_reviews"].fillna(-1).astype(float)==0).sum()
#     print(f"số lượng review = 0 là {so_0_review}")

#     vung = check["analysis_area"].value_counts()
#     v1 = vung.get("A01",0)
#     v2 = vung.get("A02",0)
#     v3 = vung.get("A03",0)
#     v4 = vung.get("A04",0)
#     v5 = vung.get("A05",0)
#     v6 = vung.get("A06",0)
#     v7 = vung.get("A07",0)
#     v8 = vung.get("A08",0)

#     print(f"{v1},{v2},{v3},{v4},{v5},{v6},{v7},{v8}")
    
#     if moc == "2026-06-20":
#         thieu_gia = check["price"].isna().sum()
#         print(f"Số lượng thiếu price ở mốc cuối là: {thieu_gia}")




import pandas as pd
import os
import argparse
import yaml

# 1. ĐỌC CẤU HÌNH TỪ YAML
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

danh_sach_moc = [str(a) for a in config["snapshots"]]

# Tạo các thư mục đầu ra bắt buộc
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/samples", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 2. ĐỌC DỮ LIỆU PHÂN CÔNG
df1000 = pd.read_csv(
    "data/raw/phan_cong_1000_listing.csv",
    dtype={"listing_id": "str"}
).rename(columns={"listing_id": "id"})
id_can_tim = set(df1000["id"].dropna())

df5000 = pd.read_csv(
    "data/raw/phan_cong_5000_review.csv",
    dtype={"listing_id": "str", "review_id": "str"}
).rename(columns={"review_id": "id"})

# Khởi tạo danh sách lưu log để xuất báo cáo
qa_logs = []

# ==========================================
# 3. XỬ LÝ BẢNG LISTINGS
# ==========================================
print("Đang xử lý bảng Listings...")
for moc in danh_sach_moc:
    first_chunk_listings = True
    for chunk in pd.read_csv(f"data/raw/{moc}_listings.csv.gz", chunksize=100_000, dtype={"id": "str"}):
        ket_qua1 = chunk.merge(df1000, on="id", how="inner")
        
        if not ket_qua1.empty:
            # LÀM SẠCH GIÁ TIỀN (ÉP SANG FLOAT)
            if "price" in ket_qua1.columns:
                ket_qua1["price"] = ket_qua1["price"].replace(r"[\$,]", "", regex=True).astype(float)
            
            ket_qua1.to_csv(
                f"data/processed/{moc}_listings.csv",
                mode="w" if first_chunk_listings else "a",
                header=first_chunk_listings,
                index=False
            )
            first_chunk_listings = False
            
    # THU THẬP THỐNG KÊ CHO QA REPORT
    if os.path.exists(f"data/processed/{moc}_listings.csv"):
        check = pd.read_csv(f"data/processed/{moc}_listings.csv", dtype={"id": "str"})
        
        apt = (check["room_type"] == 'Entire home/apt').sum() if "room_type" in check.columns else 0
        pri = (check["room_type"] == 'Private room').sum() if "room_type" in check.columns else 0
        so_0_rev = (check["number_of_reviews"].fillna(-1).astype(float) == 0).sum() if "number_of_reviews" in check.columns else 0
        thieu_gia = check["price"].isna().sum() if "price" in check.columns else 0
        vung_dict = check["analysis_area"].value_counts().to_dict() if "analysis_area" in check.columns else {}
        
        qa_logs.append({
            "snapshot_date": moc,
            "dataset": "listings",
            "total_processed_rows": len(check),
            "missing_price_count": thieu_gia,
            "zero_reviews_count": so_0_rev,
            "apt_count": apt,
            "pri_count": pri,
            "area_distribution": str(vung_dict),
            "action": "Cleaned price format"
        })

# ==========================================
# 4. XỬ LÝ BẢNG CALENDAR
# ==========================================
print("Đang xử lý bảng Calendar...")
for moc in danh_sach_moc:
    first_chunk_calendar = True
    cols = ["listing_id", "date", "available", "price"] if moc == danh_sach_moc[0] else ["listing_id", "date", "available"]
    tong_dong = 0
    
    for chunk in pd.read_csv(f"data/raw/{moc}_calendar.csv.gz", chunksize=100_000, usecols=cols, dtype={"listing_id": "str"}):
        ket_qua2 = chunk[chunk["listing_id"].isin(id_can_tim)].copy()
        
        if not ket_qua2.empty:
            # LÀM SẠCH AVAILABLE VÀ PRICE
            if "available" in ket_qua2.columns:
                ket_qua2["available"] = ket_qua2["available"].map({'t': 1, 'f': 0, 'True': 1, 'False': 0, True: 1, False: 0})
            if "price" in ket_qua2.columns:
                ket_qua2["price"] = ket_qua2["price"].replace(r"[\$,]", "", regex=True).astype(float)
            
            tong_dong += len(ket_qua2)
            ket_qua2.to_csv(
                f"data/processed/{moc}_calendar.csv",
                mode="w" if first_chunk_calendar else "a",
                header=first_chunk_calendar,
                index=False
            )
            first_chunk_calendar = False
            
    qa_logs.append({
        "snapshot_date": moc,
        "dataset": "calendar",
        "total_processed_rows": tong_dong,
        "action": "Converted available to 1/0 and cleaned price format"
    })

# ==========================================
# 5. XỬ LÝ BẢNG REVIEWS & TÁCH MẪU
# ==========================================
print("Đang xử lý bảng Reviews...")
for moc in danh_sach_moc:
    first_chunk_reviews = True
    tong_dong = 0
    
    for chunk in pd.read_csv(f"data/raw/{moc}_reviews.csv.gz", chunksize=100_000, dtype={"id": "str", "listing_id": "str"}):
        # SỬA LỖI TRÙNG CỘT BẰNG on=["id", "listing_id"]
        ket_qua3 = chunk.merge(df5000, on=["id", "listing_id"], how="inner")
        
        if not ket_qua3.empty:
            tong_dong += len(ket_qua3)
            ket_qua3.to_csv(
                f"data/processed/{moc}_reviews.csv",
                mode="w" if first_chunk_reviews else "a",
                header=first_chunk_reviews,
                index=False
            )
            first_chunk_reviews = False
            
    qa_logs.append({
        "snapshot_date": moc,
        "dataset": "reviews",
        "total_processed_rows": tong_dong,
        "action": "Merged securely without listing_id duplication"
    })
            
    # TÁCH MẪU Ở MỐC ĐẦU TIÊN
    if moc == danh_sach_moc[0] and os.path.exists(f"data/processed/{moc}_reviews.csv"):
        read = pd.read_csv(f"data/processed/{moc}_reviews.csv", dtype=str)
        if "mau_llm" in read.columns and "nhan_tay" in read.columns:
            read1 = read[read["mau_llm"] == "x"]
            read2 = read[read["nhan_tay"] == "x"]
            read1.to_csv("data/samples/llm_reviews.csv", index=False)
            read2.to_csv("data/samples/hand_labeled_reviews.csv", index=False)
            print(f"-> Đã tách {len(read1)} mẫu LLM và {len(read2)} mẫu gán nhãn tay.")

# ==========================================
# 6. XUẤT BÁO CÁO QA REPORT
# ==========================================
df_report = pd.DataFrame(qa_logs)
df_report.to_csv("reports/qa_report.csv", index=False)
print("\nHoàn tất! Báo cáo thống kê QA đã được lưu tại 'reports/qa_report.csv'")