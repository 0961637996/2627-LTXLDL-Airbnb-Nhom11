import pandas as pd
import os
import yaml
import argparse

#    ----------------ĐỌC CẤU HÌNH TỪ FILE CONFIG ĐƯỢC TRUYỀN VÀO TỪ TERMINAL BẰNG THAM SỐ --config, MẶC ĐỊNH SẼ LÀ madrid.yaml----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
config = yaml.safe_load(open(parser.parse_args().config, encoding="utf-8"))
danh_sach_moc = [str(x) for x in config["snapshots"]]

#               --------------------------------   Kiểm tra xem thư mục có tồn tại chưa, chưa có thì tạo   --------------------------------
for d in ["data/processed", "data/samples", "reports"]: 
    os.makedirs(d, exist_ok=True)


QA_LOG = []
#            --------------------------------   Tạo hàm log_qa để thêm thông tin đã sửa vào list QA_LOG   --------------------------------
def log_qa(snap, table, rule, total, affected, action):
    QA_LOG.append({"snapshot": snap, "table": table, "rule_name": rule, 
                   "total_rows": total, "affected_rows": affected, "action": action})

 #        --------------------------------   Đọc 2 file phân công listing(1000 dòng) và review(5000 dòng)   --------------------------------
df_listing = pd.read_csv("data/raw/phan_cong_1000_listing.csv", dtype=str).rename(columns={"listing_id": "id"})
list_ids = set(df_listing["id"].dropna())

df_review = pd.read_csv("data/raw/phan_cong_5000_review.csv", dtype=str).rename(columns={"review_id": "id"})

for moc in danh_sach_moc:
    print(f"\n--- Đang xử lý lọc cho mốc: {moc} ---")
    
    #                                 --------------------------------   Bảng Listing   --------------------------------
    df_l = pd.read_csv(f"data/raw/{moc}_listings.csv.gz", dtype={"id": str})
    
    #Ghép bảng để lọc ID và lấy thêm cột analysis_area từ file phân công
    df_l = df_l.merge(df_listing, on="id", how="inner")
    len_df_l = len(df_l)

    # Làm sạch dữ liệu listing và thêm vào QA_LOG

    # 1.Làm sạch ký tự bẩn trong cột giá (Lỗi Kiểu sai)
    if "price" in df_l.columns:
        affected_price_rows = df_l["price"].astype(str).str.contains(r"[\$,]", regex=True).sum()
        df_l["price"] = df_l["price"].replace(r"[\$,]", "", regex=True).astype(float)
        log_qa(moc, "listings", "type_cast_price", len_df_l, affected_price_rows, " Bỏ dấu $ ở cột giá ")

    # 2.Làm sạch giá âm và vượt ngưỡng quy định(Lỗi Thiếu và Ngoại lai)
    if "price" in df_l.columns:
        min_p = config.get("qa", {}).get("min_price", 1)
        df_l.loc[df_l["price"]<=min_p,"price"] = None
        invalid_price_mask = df_l["price"].isna()
        df_l["flag_invalid_price"] = invalid_price_mask
        log_qa(moc,"listings","invalid_price",len_df_l,invalid_price_mask.sum()," Gắn cờ các cột không có giá và có giá âm ")

        max = config.get("qa",{}).get("max_price",5000)
        outlier_price_mask = df_l["price"] > max
        df_l["flag_outlier_price"] = outlier_price_mask
        log_qa(moc,"listings","outlier_price",len_df_l,outlier_price_mask.sum()," Gắn cờ các giá ngoại lai ")

    # 3.Làm sạch danh mục vùng và loại phòng (Lỗi Danh mục lệch)
    if "analysis_area" in df_l.columns:
        valid_area = config.get("qa",{}).get("valid_areas",["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"])
        invalid_area_mask = ~df_l["analysis_area"].isin(valid_area)
        df_l["flag_invalid_area"] = invalid_area_mask
        log_qa(moc,"listings","invalid_area",len_df_l,invalid_area_mask.sum()," Gắn cờ các vùng không hợp lệ ")

    if "room_type" in df_l.columns:
        valid_room_type = config.get("qa",{}).get("valid_room_types",["Entire home/apt", "Private room", "Shared room", "Hotel room"])
        invalid_room_type_mask = ~df_l["room_type"].isin(valid_room_type)
        df_l["flag_invalid_room_type"] = invalid_room_type_mask
        log_qa(moc,"listings","invalid_room_type",len_df_l,invalid_room_type_mask.sum()," Gắn cờ các loại phòng không hợp lệ ")

    # 4.Làm sạch id trùng lặp trong 1 mốc
    dup_mask = df_l.duplicated(subset=["id"], keep=False)
    df_l["flag_dup_id"] = dup_mask
    log_qa(moc, "listings", "duplicate_id_in_snap", len_df_l, dup_mask.sum(), "Gắn cờ những id bị lặp")

    df_l.to_csv(f"data/processed/{moc}_listings.csv", index=False)


    #                                     --------------------------------   Kiểm tra điều kiện   --------------------------------
    print(f"kiểm tra mốc {moc}:")
    
    if "room_type" in df_l.columns:
        apt = (df_l["room_type"] == "Entire home/apt").sum()
        pri = (df_l["room_type"] == "Private room").sum()
        print(f"    - Entire home/apt: {apt}")
        print(f"    - Private room: {pri}")
    
    if "number_of_reviews" in df_l.columns:
        # Ép kiểu sang số thực, điền -1 cho khoảng trống để đếm đúng số 0
        so_0_rev = (pd.to_numeric(df_l["number_of_reviews"], errors='coerce').fillna(-1) == 0).sum()
        print(f"    - number_of_reviews = 0: {so_0_rev}")

    if "analysis_area" in df_l.columns:
        area_counts = df_l["analysis_area"].value_counts()
        all_valid = all(120 <= count <= 130 for count in area_counts)
        print(f"    - Vùng phân tích (120-130 listing/vùng): {all_valid}")
        if not all_valid:
            print(f"      Chi tiết lỗi phân bổ: {area_counts.to_dict()}")

    if moc == danh_sach_moc[-1] and "price" in df_l.columns:
        thieu_gia = df_l["price"].isna().sum()
        print(f"    - Số listing thiếu price: {thieu_gia} ")


    #                                   --------------------------------   BẢNG CALENDAR   --------------------------------
    first_chunk = True
    len_df_c = 0

    count_change =0
    affected_price_c = 0
    count_invalid_price = 0
    usecols = ["listing_id", "date", "available", "price"] if moc == danh_sach_moc[0] else ["listing_id", "date", "available"]
    
    for chunk in pd.read_csv(f"data/raw/{moc}_calendar.csv.gz", chunksize=100_000, usecols=usecols, dtype={"listing_id": str}):
        chunk = chunk[chunk["listing_id"].isin(list_ids)]
        len_df_c += len(chunk)
        if not chunk.empty:
            # Chuyển ký tự "t" , "f" thành số 0 , 1 để dễ dàng tính toán 
            if "available" in chunk.columns:
                change = chunk["available"].map({'t': 1, 'f': 0})
                count_change += change.notna().sum()
                chunk["available"] = change

            # Xóa dấu $ ở cột giá và số cột có giá <=0 hoặc thiếu giá
            if "price" in chunk.columns:
                affected_price_c += chunk["price"].astype(str).str.contains(r"[\$,]", regex=True).sum()
                chunk["price"] = chunk["price"].replace(r"[\$,]", "", regex=True).astype(float)

                invalid_price_mask = (chunk["price"] <=0) | chunk["price"].isna()
                count_invalid_price += invalid_price_mask.sum()
                chunk.loc[chunk["price"]<=0,"price"] = None
                chunk["flag_invalid_price"] = invalid_price_mask

            
            chunk.to_csv(f"data/processed/{moc}_calendar.csv", mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False

    # QA Log cho Calendar         
    log_qa(moc, "calendar", "change",len_df_c,count_change, " Chuyển ký tự t , f thành số 0 , 1 ")
    log_qa(moc, "calendar", "type_cast_price",len_df_c,affected_price_c, " Bỏ dấu $ ở cột giá ")
    log_qa(moc, "calendar", "invalid_price",len_df_c,count_invalid_price, " Gắn cờ các cột không có giá và có giá âm ")

    #                             --------------------------------   BẢNG REVIEWS & TÁCH MẪU   --------------------------------
    first_chunk = True
    len_df_r = 0
    count_html_cleaned = 0
    count_empty_comments = 0
    
    for chunk in pd.read_csv(f"data/raw/{moc}_reviews.csv.gz", chunksize=100_000, dtype={"id": str, "listing_id": str}):
        chunk = chunk.merge(df_review, on="id", how="inner")
        len_df_r += len(chunk)
        if not chunk.empty:
            # Xử lý kí tự bẩn trong comment và thiếu comment
            if "comments" in chunk.columns:
                dirty_html_mask = chunk["comments"].fillna("").str.contains(r"<[^>]+>", regex=True)
                count_html_cleaned += dirty_html_mask.sum()
                chunk["comments"] = chunk["comments"].replace(r"<[^>]+>", " ", regex=True)

                empty_comment_mask = chunk["comments"].isna() | (chunk["comments"].str.strip() == "")
                count_empty_comments += empty_comment_mask.sum()
                chunk["flag_empty_comment"] = empty_comment_mask
            
            chunk.to_csv(f"data/processed/{moc}_reviews.csv", mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False
    df = pd.read_csv(f"data/processed/{moc}_reviews.csv", dtype=str)
    # Xử lý comment và id bị trùng lặp
    dup_id_mask = df.duplicated(subset=["id"], keep=False)
    count_dup_id = dup_id_mask.sum()
    df["flag_dup_id"] = dup_id_mask

    dup_comment_mask = df.duplicated(subset=["comments"], keep=False) & ~(df["comments"].isna() | (df["comments"].str.strip() == ""))
    count_dup_comment = dup_comment_mask.sum()
    df["flag_dup_comment"] = dup_comment_mask

    df.to_csv(f"data/processed/{moc}_reviews.csv", index=False)
            
    # QA Log cho Reviews 
    log_qa(moc, "reviews", "clean_html", len_df_r,count_html_cleaned , "Làm sạch comment")
    log_qa(moc, "reviews", "missing_or_empty_comment", len_df_r,count_empty_comments , "Xử lý comment bị thiếu hoặc khoảng trắng")
    log_qa(moc, "reviews", "duplicate_id", len_df_r, count_dup_id, "Xử lý trùng lặp id")
    log_qa(moc, "reviews", "duplicate_comment", len_df_r,count_dup_comment , "Xử lý trùng lặp comment")

    # Tách mẫu LLM và nhãn tay
    if moc == danh_sach_moc[0]:
        df_processed = df
        if "mau_llm" in df_processed.columns and "nhan_tay" in df_processed.columns:
            df_processed[df_processed["mau_llm"].str.lower() == "x"].to_csv("data/samples/llm_reviews.csv", index=False)
            df_processed[df_processed["nhan_tay"].str.lower() == "x"].to_csv("data/samples/hand_labeled_reviews.csv", index=False)



#                --------------------------------   Xuất báo cáo QA vào file qa_report.csv   --------------------------------
pd.DataFrame(QA_LOG).to_csv("reports/qa_report.csv", index=False)
print("\nHoàn tất việc lọc và làm sạch! Đã lưu báo cáo tại reports/qa_report.csv")
