import pandas as pd
import numpy as np
import os
import yaml
import argparse

# ĐỌC CẤU HÌNH TỪ FILE CONFIG ĐƯỢC TRUYỀN VÀO TỪ TERMINAL BẰNG THAM SỐ --config, MẶC ĐỊNH SẼ LÀ madrid.yaml
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
config = yaml.safe_load(open(parser.parse_args().config, encoding="utf-8"))
danh_sach_moc = [str(x) for x in config["snapshots"]]

# Kiểm tra xem thư mục có tồn tại chưa, chưa có thì tạo   
for d in ["data/processed", "data/samples", "reports"]:
    os.makedirs(d, exist_ok=True)

# Tham số QA lấy từ khối qa: trong YAML (không ghi cứng trong code)   
qa_cfg = config.get("qa") or {}
MIN_PRICE = qa_cfg.get("min_price", 1)                                   # giá < ngưỡng này coi là không hợp lệ
MAX_PRICE = qa_cfg.get("max_price", qa_cfg.get("price_max", 1500))       # giá > ngưỡng này coi là ngoại lai
VALID_AREAS = qa_cfg.get("valid_areas", ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"])
VALID_ROOM_TYPES = qa_cfg.get("valid_room_types", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
AREA_RANGE = qa_cfg.get("expected_area_range", [120, 130])               # số listing mong đợi mỗi analysis_area
SCRAPE_TOL = qa_cfg.get("scrape_tolerance_days", 7)                      # last_scraped lệch ngày mốc tối đa
CAL_WINDOW = qa_cfg.get("calendar_window_days", 400)                     # calendar không vượt mốc + N ngày
MIN_COMMENT_CHARS = qa_cfg.get("review_min_chars", 3)
PRICE_JUMP = qa_cfg.get("price_jump_ratio", 3.0)                         # giá đổi >= N lần giữa 2 mốc liên tiếp
HTML_COLS = qa_cfg.get("html_cols", ["name", "description", "neighborhood_overview", "license"])
AUTOMATED = qa_cfg.get("automated_patterns", ["this is an automated posting", "the host canceled this reservation",
                                              "the reservation was canceled"])
AREA_COL = (config.get("kpi") or {}).get("area_col", "neighbourhood_cleansed")

QA_LOG = []
# Tạo hàm log_qa để thêm thông tin đã sửa vào list QA_LOG  

# Mỗi quy tắc: tên -> điều kiện -> lý do -> hành động; kèm số dòng bị tác động và KPI trước/sau (nếu có)
def log_qa(snap, table, rule, total, affected, action, condition="", reason="",
           kpi_name="", kpi_before=None, kpi_after=None, note=""):
    total, affected = int(total), int(affected)
    def r(v):
        return None if v is None or pd.isna(v) else round(float(v), 4)
    QA_LOG.append({"snapshot": snap, "table": table, "rule_name": rule, "condition": condition, "reason": reason,
                   "total_rows": total, "affected_rows": affected,
                   "pct_affected": round(100 * affected / total, 2) if total else 0.0,
                   "action": action, "kpi_name": kpi_name, "kpi_before": r(kpi_before), "kpi_after": r(kpi_after),
                   "note": note})


def to_number(s):
    """'$1,234.50' -> 1234.5 ; chuỗi lỗi hoặc rỗng -> NaN (không làm chương trình dừng)."""
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def strip_html(s):
    """Bỏ thẻ HTML (<br />, <b>...) và gộp khoảng trắng."""
    return s.str.replace(r"<[^>]+>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


# Đọc 2 file phân công listing(1000 dòng) và review(5000 dòng)
df_listing = pd.read_csv("data/raw/phan_cong_1000_listing.csv", dtype=str).rename(columns={"listing_id": "id"}).drop_duplicates("id")
list_ids = set(df_listing["id"].dropna())

df_review = pd.read_csv("data/raw/phan_cong_5000_review.csv", dtype=str).rename(columns={"review_id": "id"}).drop_duplicates("id")

first_cols = None
for moc in danh_sach_moc:
    print(f"\n--- Đang xử lý lọc cho mốc: {moc} ---")
    snap_ts = pd.Timestamp(moc)

    # Xoá file đã xử lý của lần chạy trước để không đọc nhầm dữ liệu cũ
    for t in ("listings", "calendar", "reviews"):
        old = f"data/processed/{moc}_{t}.csv"
        if os.path.exists(old):
            os.remove(old)

    # Bảng Listing 
    df_l = pd.read_csv(f"data/raw/{moc}_listings.csv.gz", dtype={"id": str})

    # QA cấu trúc: tập cột có đổi so với mốc đầu không (Inside Airbnb thêm/bớt cột giữa các mốc)
    cols_now = list(df_l.columns)
    if first_cols is None:
        first_cols = cols_now
    else:
        added = [c for c in cols_now if c not in first_cols]
        removed = [c for c in first_cols if c not in cols_now]
        log_qa(moc, "listings", "schema_drift", len(first_cols), len(added) + len(removed), "Chỉ báo cáo",
               condition="tập cột khác mốc đầu tiên", reason="so sánh giữa các mốc cần cùng lược đồ",
               note=f"thêm: {added}; mất: {removed}")

    # Ghép bảng để lọc ID và lấy thêm cột analysis_area từ file phân công
    # (chỉ lấy cột chưa có trong bảng gốc để không sinh cột _x/_y)
    extra = [c for c in df_listing.columns if c == "id" or c not in df_l.columns]
    df_l = df_l.merge(df_listing[extra], on="id", how="inner")
    len_df_l = len(df_l)

    # Làm sạch dữ liệu listing và thêm vào QA_LOG

    # 1.Làm sạch ký tự bẩn trong cột giá (Lỗi Kiểu sai)
    if "price" in df_l.columns:
        raw_price = df_l["price"]
        affected_price_rows = raw_price.astype(str).str.contains(r"[\$,]", regex=True, na=False).sum()
        price = to_number(raw_price)
        unparseable = raw_price.notna() & price.isna()
        df_l["price"] = price
        median_cast = price.median()
        log_qa(moc, "listings", "type_cast_price", len_df_l, affected_price_rows, "Bỏ dấu $ , rồi ép sang số",
               condition="price chứa $ hoặc ,", reason="giá dạng chuỗi không tính toán được",
               kpi_name="median_price", kpi_before=median_cast, kpi_after=median_cast)
        log_qa(moc, "listings", "price_unparseable", len_df_l, unparseable.sum(), "Đặt rỗng (NaN)",
               condition="price có giá trị nhưng không chuyển được sang số", reason="định dạng lạ, không đoán được giá trị")

    # 2.Giá thiếu, giá không hợp lệ, giá ngoại lai (Lỗi Thiếu và Ngoại lai)
    if "price" in df_l.columns:
        missing_mask = df_l["price"].isna()
        df_l["flag_missing_price"] = missing_mask
        pct_missing = 100 * missing_mask.mean() if len_df_l else 0
        log_qa(moc, "listings", "missing_price", len_df_l, missing_mask.sum(), "Gắn cờ",
               condition="price rỗng", reason="không tính được KPI giá; vẫn giữ dòng cho KPI khác",
               kpi_name="median_price", kpi_before=df_l["price"].median(), kpi_after=df_l["price"].median(),
               note="MỐC NÀY KHÔNG DÙNG ĐƯỢC CHO KPI GIÁ (thiếu >= 50%)" if pct_missing >= 50 else "")

        median_before = df_l["price"].median()
        invalid_price_mask = df_l["price"].notna() & (df_l["price"] < MIN_PRICE)
        df_l.loc[invalid_price_mask, "price"] = np.nan
        df_l["flag_invalid_price"] = invalid_price_mask
        log_qa(moc, "listings", "invalid_price", len_df_l, invalid_price_mask.sum(), "Đặt rỗng và gắn cờ",
               condition=f"price < {MIN_PRICE}", reason="giá âm/bằng 0 là không hợp lệ",
               kpi_name="median_price", kpi_before=median_before, kpi_after=df_l["price"].median())

        outlier_price_mask = df_l["price"] > MAX_PRICE
        df_l["flag_outlier_price"] = outlier_price_mask
        log_qa(moc, "listings", "outlier_price", len_df_l, outlier_price_mask.sum(), "Gắn cờ",
               condition=f"price > {MAX_PRICE}", reason="ngoại lai kéo lệch giá trung bình; loại khỏi KPI giá",
               kpi_name="median_price", kpi_before=df_l["price"].median(),
               kpi_after=df_l.loc[~outlier_price_mask, "price"].median())

    # 3.Làm sạch danh mục vùng và loại phòng (Lỗi Danh mục lệch)
    if "analysis_area" in df_l.columns:
        invalid_area_mask = ~df_l["analysis_area"].isin(VALID_AREAS)
        df_l["flag_invalid_area"] = invalid_area_mask
        log_qa(moc, "listings", "invalid_area", len_df_l, invalid_area_mask.sum(), "Gắn cờ",
               condition=f"analysis_area không thuộc {VALID_AREAS}", reason="vùng lạ làm sai thống kê theo vùng")

    if "room_type" in df_l.columns:
        df_l["room_type"] = df_l["room_type"].str.strip()
        invalid_room_type_mask = ~df_l["room_type"].isin(VALID_ROOM_TYPES)
        df_l["flag_invalid_room_type"] = invalid_room_type_mask
        log_qa(moc, "listings", "invalid_room_type", len_df_l, invalid_room_type_mask.sum(), "Gắn cờ",
               condition=f"room_type không thuộc {VALID_ROOM_TYPES}", reason="gõ sai/danh mục lạ làm sai thống kê theo loại")

    # 4.Id trùng lặp trong 1 mốc (gắn cờ các bản sao từ lần xuất hiện thứ 2 trở đi, giữ lại bản đầu để còn lọc được)
    dup_mask = df_l.duplicated(subset=["id"], keep="first")
    df_l["flag_dup_id"] = dup_mask
    log_qa(moc, "listings", "duplicate_id_in_snap", len_df_l, dup_mask.sum(), "Gắn cờ những id bị lặp",
           condition="id đã xuất hiện trước đó trong cùng mốc", reason="đếm đôi listing")

    # 5.Thẻ HTML trong cột văn bản (description, license... còn <br />)
    n_cells = n_html = 0
    for c in HTML_COLS:
        if c in df_l.columns and df_l[c].notna().any():
            s = df_l[c].astype("string")
            has_html = s.str.contains(r"<[^>]+>", regex=True, na=False)
            n_cells += int(s.notna().sum())
            n_html += int(has_html.sum())
            df_l.loc[has_html, c] = strip_html(s[has_html])
    log_qa(moc, "listings", "clean_html", n_cells, n_html, "Xoá thẻ HTML",
           condition=f"{HTML_COLS} chứa <...>", reason="thẻ HTML làm nhiễu văn bản")

    # 6.last_scraped lệch ngày mốc (listing được thu thập muộn)
    if "last_scraped" in df_l.columns:
        ls = pd.to_datetime(df_l["last_scraped"], errors="coerce")
        drift_mask = ls.isna() | ((ls - snap_ts).abs().dt.days > SCRAPE_TOL)
        df_l["flag_scrape_drift"] = drift_mask
        log_qa(moc, "listings", "scrape_date_drift", len_df_l, drift_mask.sum(), "Gắn cờ",
               condition=f"last_scraped không đọc được hoặc lệch ngày mốc > {SCRAPE_TOL} ngày",
               reason="bản ghi thu thập sai thời điểm, lịch calendar cũng bị lệch theo")

    df_l.to_csv(f"data/processed/{moc}_listings.csv", index=False)
    ids_snapshot = set(df_l["id"].dropna())

    # Kiểm tra điều kiện
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
        log_qa(moc, "listings", "zero_reviews", len_df_l, so_0_rev, "Chỉ báo cáo",
               condition="number_of_reviews = 0", reason="không có review để phân tích; điểm đánh giá sẽ rỗng")

    if "analysis_area" in df_l.columns:
        area_counts = df_l["analysis_area"].value_counts()
        lech = area_counts[(area_counts < AREA_RANGE[0]) | (area_counts > AREA_RANGE[1])]
        all_valid = lech.empty
        print(f"    - Vùng phân tích ({AREA_RANGE[0]}-{AREA_RANGE[1]} listing/vùng): {all_valid}")
        if not all_valid:
            print(f"      Chi tiết lỗi phân bổ: {area_counts.to_dict()}")
        log_qa(moc, "listings", "area_balance", len(area_counts), len(lech), "Chỉ báo cáo",
               condition=f"số listing mỗi vùng ngoài [{AREA_RANGE[0]}, {AREA_RANGE[1]}]",
               reason="mẫu phân công phải cân bằng giữa các vùng", note=str(area_counts.to_dict()))

    if moc == danh_sach_moc[-1] and "price" in df_l.columns:
        thieu_gia = df_l["price"].isna().sum()
        print(f"    - Số listing thiếu price: {thieu_gia} ")

    # BẢNG CALENDAR 
    first_chunk = True
    len_df_c = 0

    count_change = 0
    count_bad_available = 0
    affected_price_c = 0
    count_invalid_price = 0
    n_bad_date = n_out_window = n_orphan = n_flag_any = 0
    sum_all = n_all = sum_ok = n_ok = 0          # để tính tỷ lệ ngày trống trước/sau khi loại dòng bị cờ
    cal_stats = []                               # gom theo listing để tạo bảng master
    usecols = ["listing_id", "date", "available", "price"] if moc == danh_sach_moc[0] else ["listing_id", "date", "available"]

    for chunk in pd.read_csv(f"data/raw/{moc}_calendar.csv.gz", chunksize=100_000, usecols=usecols, dtype={"listing_id": str}):
        chunk = chunk[chunk["listing_id"].isin(list_ids)].copy()
        len_df_c += len(chunk)
        if not chunk.empty:
            # Chuyển ký tự "t" , "f" thành số 0 , 1 để dễ dàng tính toán (giá trị lạ -> rỗng và được đếm lại)
            if "available" in chunk.columns:
                change = chunk["available"].astype(str).str.lower().map({'t': 1, 'f': 0, 'true': 1, 'false': 0})
                count_change += change.notna().sum()
                count_bad_available += (chunk["available"].notna() & change.isna()).sum()
                chunk["available"] = change

            # Kiểm tra ngày: sai kiểu, ngoài cửa sổ quanh mốc, listing không còn trong bảng listings của mốc
            dates = pd.to_datetime(chunk["date"], errors="coerce")
            bad_date = dates.isna()
            out_window = ~bad_date & ((dates < snap_ts - pd.Timedelta(days=1)) | (dates > snap_ts + pd.Timedelta(days=CAL_WINDOW)))
            orphan = ~chunk["listing_id"].isin(ids_snapshot)
            chunk["flag_bad_date"] = bad_date
            chunk["flag_out_of_window"] = out_window
            chunk["flag_orphan_listing"] = orphan
            n_bad_date += bad_date.sum()
            n_out_window += out_window.sum()
            n_orphan += orphan.sum()
            ok = ~(bad_date | out_window | orphan)
            n_flag_any += (~ok).sum()
            sum_all += chunk["available"].sum()
            n_all += chunk["available"].notna().sum()
            sum_ok += chunk.loc[ok, "available"].sum()
            n_ok += chunk.loc[ok, "available"].notna().sum()

            # Gom theo listing (chỉ các dòng không bị cờ): tổng số ngày và số ngày KHÔNG còn trống (available = 0)
            part = chunk.loc[ok].assign(_u=(chunk.loc[ok, "available"] == 0))
            cal_stats.append(part.groupby("listing_id").agg(total_days=("date", "count"), unavailable_days=("_u", "sum")).reset_index())

            # Xóa dấu $ ở cột giá và số cột có giá <=0 hoặc thiếu giá
            if "price" in chunk.columns:
                affected_price_c += chunk["price"].astype(str).str.contains(r"[\$,]", regex=True, na=False).sum()
                chunk["price"] = to_number(chunk["price"])

                invalid_price_mask = (chunk["price"] <= 0) | chunk["price"].isna()
                count_invalid_price += invalid_price_mask.sum()
                chunk.loc[chunk["price"] <= 0, "price"] = None
                chunk["flag_invalid_price"] = invalid_price_mask

            chunk.to_csv(f"data/processed/{moc}_calendar.csv", mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False

    # Trùng (listing_id, date): một listing có hai dòng cùng ngày làm đếm đôi ngày trống
    count_dup_cal = 0
    cal_path = f"data/processed/{moc}_calendar.csv"
    if os.path.exists(cal_path):
        count_dup_cal = pd.read_csv(cal_path, usecols=["listing_id", "date"], dtype=str).duplicated().sum()

    # QA Log cho Calendar
    log_qa(moc, "calendar", "change", len_df_c, count_change, " Chuyển ký tự t , f thành số 0 , 1 ",
           condition="available là t/f", reason="cần số để tính tỷ lệ ngày trống")
    log_qa(moc, "calendar", "invalid_available", len_df_c, count_bad_available, "Đặt rỗng",
           condition="available khác t/f/true/false", reason="kiểu sai, không đoán được giá trị")
    log_qa(moc, "calendar", "type_cast_price", len_df_c, affected_price_c, " Bỏ dấu $ ở cột giá ",
           condition="price chứa $ hoặc ,", reason="giá dạng chuỗi không tính toán được")
    log_qa(moc, "calendar", "invalid_price", len_df_c, count_invalid_price, " Gắn cờ các cột không có giá và có giá âm ",
           condition="price rỗng hoặc <= 0", reason="giá không hợp lệ",
           note="calendar không có giá (Inside Airbnb để trống)" if len_df_c and count_invalid_price == len_df_c else "")
    log_qa(moc, "calendar", "invalid_date", len_df_c, n_bad_date, "Gắn cờ",
           condition="date không đọc được", reason="không đặt lên trục thời gian được")
    log_qa(moc, "calendar", "out_of_window", len_df_c, n_out_window, "Gắn cờ",
           condition=f"date < mốc-1 hoặc > mốc+{CAL_WINDOW} ngày", reason="calendar chỉ nên nằm trong khoảng một năm kể từ mốc")
    log_qa(moc, "calendar", "orphan_listing", len_df_c, n_orphan, "Gắn cờ",
           condition="listing_id không có trong listings của cùng mốc", reason="không gắn được khu vực/loại phòng")
    log_qa(moc, "calendar", "duplicate_listing_date", len_df_c, count_dup_cal, "Chỉ báo cáo",
           condition="(listing_id, date) xuất hiện nhiều hơn một lần", reason="đếm đôi ngày trống/đã đặt")
    log_qa(moc, "calendar", "flags_total", len_df_c, n_flag_any, "Tổng hợp: KPI sau = bỏ dòng bị cờ ngày/cửa sổ/mồ côi",
           condition="dòng bị ít nhất một cờ ngày", reason="đo tác động lên tỷ lệ ngày trống",
           kpi_name="availability_rate", kpi_before=sum_all / n_all if n_all else None,
           kpi_after=sum_ok / n_ok if n_ok else None)

    # BẢNG REVIEWS & TÁCH MẪU 
    first_chunk = True
    len_df_r = 0
    count_html_cleaned = 0
    count_empty_comments = 0
    sum_len_before = n_len_before = 0

    for chunk in pd.read_csv(f"data/raw/{moc}_reviews.csv.gz", chunksize=100_000, dtype={"id": str, "listing_id": str}):
        # Chỉ lấy từ file phân công các cột chưa có (mau_llm, nhan_tay) -> không sinh cột listing_id_x / listing_id_y
        extra = [c for c in df_review.columns if c == "id" or c not in chunk.columns]
        chunk = chunk.merge(df_review[extra], on="id", how="inner")
        len_df_r += len(chunk)
        if not chunk.empty:
            # Xử lý kí tự bẩn trong comment và thiếu comment
            if "comments" in chunk.columns:
                sum_len_before += chunk["comments"].fillna("").str.len().sum()
                n_len_before += len(chunk)
                dirty_html_mask = chunk["comments"].fillna("").str.contains(r"<[^>]+>", regex=True)
                count_html_cleaned += dirty_html_mask.sum()
                chunk["comments"] = chunk["comments"].replace(r"<[^>]+>", " ", regex=True)

                empty_comment_mask = chunk["comments"].isna() | (chunk["comments"].str.strip() == "")
                count_empty_comments += empty_comment_mask.sum()
                chunk["flag_empty_comment"] = empty_comment_mask

            chunk.to_csv(f"data/processed/{moc}_reviews.csv", mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False

    review_path = f"data/processed/{moc}_reviews.csv"
    if os.path.exists(review_path):
        df = pd.read_csv(review_path, dtype=str)
        text = df["comments"].fillna("")
        mean_len_before = sum_len_before / n_len_before if n_len_before else None
        mean_len_clean = text.str.len().mean()

        # Xử lý comment và id bị trùng lặp
        dup_id_mask = df.duplicated(subset=["id"], keep="first")
        count_dup_id = dup_id_mask.sum()
        df["flag_dup_id"] = dup_id_mask

        dup_comment_mask = df.duplicated(subset=["comments"], keep=False) & ~(df["comments"].isna() | (df["comments"].str.strip() == ""))
        count_dup_comment = dup_comment_mask.sum()
        df["flag_dup_comment"] = dup_comment_mask

        # Thêm: review ở tương lai, quá ngắn, tự động (hệ thống), mồ côi
        future_mask = pd.to_datetime(df["date"], errors="coerce") > snap_ts
        short_mask = (text.str.strip().str.len() > 0) & (text.str.strip().str.len() < MIN_COMMENT_CHARS)
        auto_mask = text.str.contains("|".join(AUTOMATED), case=False, regex=True)
        orphan_mask = ~df["listing_id"].isin(ids_snapshot)
        df["flag_future_date"], df["flag_short_comment"] = future_mask, short_mask
        df["flag_automated_comment"], df["flag_orphan_listing"] = auto_mask, orphan_mask
        empty_mask = df["flag_empty_comment"].astype(str) == "True"
        text_flag = empty_mask | short_mask | auto_mask
        mean_len_after = text.loc[~text_flag].str.len().mean()

        df.to_csv(review_path, index=False)

        # QA Log cho Reviews
        log_qa(moc, "reviews", "clean_html", len_df_r, count_html_cleaned, "Làm sạch comment",
               condition="comments chứa <...> (vd <br/>)", reason="thẻ HTML làm nhiễu bước LLM",
               kpi_name="mean_comment_chars", kpi_before=mean_len_before, kpi_after=mean_len_clean)
        log_qa(moc, "reviews", "missing_or_empty_comment", len_df_r, count_empty_comments, "Xử lý comment bị thiếu hoặc khoảng trắng",
               condition="comments rỗng hoặc chỉ khoảng trắng", reason="không có nội dung để phân tích")
        log_qa(moc, "reviews", "duplicate_id", len_df_r, count_dup_id, "Xử lý trùng lặp id",
               condition="id đã xuất hiện trước đó trong cùng mốc", reason="đếm đôi đánh giá")
        log_qa(moc, "reviews", "duplicate_comment", len_df_r, count_dup_comment, "Xử lý trùng lặp comment",
               condition="cùng nội dung comment với review khác", reason="có thể là spam/sao chép; chỉ gắn cờ")
        log_qa(moc, "reviews", "future_date", len_df_r, future_mask.sum(), "Gắn cờ",
               condition="date > ngày mốc chụp", reason="không thể có sau thời điểm chụp")
        log_qa(moc, "reviews", "short_comment", len_df_r, short_mask.sum(), "Gắn cờ",
               condition=f"độ dài < {MIN_COMMENT_CHARS} ký tự", reason="không đủ thông tin cho cảm xúc/khía cạnh")
        log_qa(moc, "reviews", "automated_comment", len_df_r, auto_mask.sum(), "Gắn cờ",
               condition="khớp mẫu thông báo hệ thống (huỷ đặt phòng...)", reason="không phải ý kiến khách; loại khỏi mẫu LLM")
        log_qa(moc, "reviews", "orphan_listing", len_df_r, orphan_mask.sum(), "Gắn cờ",
               condition="listing_id không có trong listings của cùng mốc", reason="không gắn được khu vực/giá")
        log_qa(moc, "reviews", "flags_total", len_df_r, text_flag.sum(), "Tổng hợp: KPI sau = bỏ comment rỗng/ngắn/tự động",
               condition="comment rỗng, quá ngắn hoặc tự động", reason="đo tác động lên độ dài comment trung bình",
               kpi_name="mean_comment_chars", kpi_before=mean_len_clean, kpi_after=mean_len_after)

        # Tách mẫu LLM và nhãn tay (mốc đầu). Nhãn đã gán tay được giữ lại qua các lần chạy.
        if moc == danh_sach_moc[0] and "mau_llm" in df.columns and "nhan_tay" in df.columns:
            hand_path = "data/samples/hand_labeled_reviews.csv"
            old_labels = {}
            if os.path.exists(hand_path):
                old = pd.read_csv(hand_path, dtype=str)
                if {"review_id", "nhan"} <= set(old.columns):
                    old_labels = dict(old.dropna(subset=["nhan"])[["review_id", "nhan"]].values)

            llm_sample = df[df["mau_llm"].str.lower() == "x"]
            hand_sample = df[df["nhan_tay"].str.lower() == "x"]
            if old_labels:
                kept = df[df["id"].isin(old_labels)]
                llm_sample = pd.concat([llm_sample, kept]).drop_duplicates("id")
                hand_sample = pd.concat([hand_sample, kept]).drop_duplicates("id")

            cols = [c for c in ("id", "listing_id", "date", "comments") if c in df.columns]
            llm_sample[cols].rename(columns={"id": "review_id"}).to_csv("data/samples/llm_reviews.csv", index=False)
            hand_out = hand_sample[cols].rename(columns={"id": "review_id"}).copy()
            hand_out["nhan"] = hand_out["review_id"].map(old_labels)   # điền: positive / mixed / negative
            hand_out.to_csv(hand_path, index=False)
            print(f"-> Mẫu LLM: {len(llm_sample)} dòng | mẫu nhãn tay: {len(hand_out)} dòng ({hand_out['nhan'].notna().sum()} đã có nhãn)")

    # BẢNG MASTER (mỗi listing một dòng, SAU QA)
    # Khác bản cũ: booked_days -> unavailable_days (gồm cả ngày chủ nhà tự chặn lịch, không chỉ ngày đã đặt);
    # review_count -> reviews_in_sample (chỉ đếm review nằm trong mẫu được giao);
    # flag_zero_reviews dựa trên number_of_reviews của listing; giữ thêm khu vực để tính KPI theo khu vực.
    cal_agg = (pd.concat(cal_stats).groupby("listing_id").sum().reset_index().rename(columns={"listing_id": "id"})
               if cal_stats else pd.DataFrame(columns=["id", "total_days", "unavailable_days"]))
    rev_agg = (df.groupby("listing_id").size().reset_index(name="reviews_in_sample").rename(columns={"listing_id": "id"})
               if os.path.exists(review_path) else pd.DataFrame(columns=["id", "reviews_in_sample"]))
    keep = [c for c in ["id", "room_type", "analysis_area", AREA_COL, "price", "number_of_reviews", "review_scores_rating",
                        "flag_missing_price", "flag_invalid_price", "flag_outlier_price", "flag_scrape_drift"] if c in df_l.columns]
    master = df_l.loc[~df_l["flag_dup_id"], keep].merge(cal_agg, on="id", how="left").merge(rev_agg, on="id", how="left")
    for c in ("total_days", "unavailable_days", "reviews_in_sample"):
        master[c] = master[c].fillna(0).astype(int)
    master["flag_missing_price"] = master["flag_missing_price"].astype(int)
    master["flag_price_outlier"] = (master["flag_invalid_price"] | master["flag_outlier_price"]).astype(int)
    master["flag_zero_reviews"] = (pd.to_numeric(master["number_of_reviews"], errors="coerce") == 0).astype(int)
    master = master.drop(columns=["flag_invalid_price", "flag_outlier_price"])
    master.to_csv(f"data/processed/{moc}_master_clean.csv", index=False)
    print(f"    -> Đã tạo bảng tổng hợp: data/processed/{moc}_master_clean.csv ({len(master)} dòng)")
    log_qa(moc, "master", "listings_without_calendar", len(master), (master["total_days"] == 0).sum(), "Chỉ báo cáo",
           condition="total_days = 0", reason="không tính được tỷ lệ ngày không còn trống")

# KIỂM TRA GIỮA CÁC MỐC (trùng / lệch giữa các mốc) 
prev_l, seen_reviews = None, set()
for moc in danh_sach_moc:
    cur_l = pd.read_csv(f"data/processed/{moc}_listings.csv", dtype={"id": str}).drop_duplicates("id").set_index("id")
    rev_file = f"data/processed/{moc}_reviews.csv"
    cur_r = set(pd.read_csv(rev_file, usecols=["id"], dtype=str)["id"]) if os.path.exists(rev_file) else set()

    if prev_l is not None:
        common = cur_l.index.intersection(prev_l.index)
        a, b = prev_l.loc[common], cur_l.loc[common]
        log_qa(moc, "listings", "gone_since_prev", len(prev_l), (~prev_l.index.isin(cur_l.index)).sum(), "Chỉ báo cáo",
               condition="id có ở mốc trước, không có ở mốc này", reason="churn ảnh hưởng so sánh theo thời gian")
        if "last_scraped" in b.columns:
            stale = (b["last_scraped"] == a["last_scraped"]) & b["last_scraped"].notna()
            log_qa(moc, "listings", "stale_record_repeated", len(common), stale.sum(), "Chỉ báo cáo",
                   condition="cùng id và cùng last_scraped với mốc trước", reason="bản ghi cũ lặp lại giữa các mốc")
        if "room_type" in b.columns:
            ch = (a["room_type"] != b["room_type"]) & a["room_type"].notna() & b["room_type"].notna()
            log_qa(moc, "listings", "room_type_changed", len(common), ch.sum(), "Chỉ báo cáo",
                   condition="room_type khác mốc trước", reason="đổi loại hình làm so sánh theo loại thiếu nhất quán")
        if AREA_COL in b.columns:
            ch = (a[AREA_COL] != b[AREA_COL]) & a[AREA_COL].notna() & b[AREA_COL].notna()
            log_qa(moc, "listings", "area_changed", len(common), ch.sum(), "Chỉ báo cáo",
                   condition=f"{AREA_COL} khác mốc trước", reason="đổi khu vực làm KPI theo khu vực thiếu nhất quán")
        if "price" in b.columns:
            ratio = pd.to_numeric(b["price"], errors="coerce") / pd.to_numeric(a["price"], errors="coerce")
            jump = (ratio >= PRICE_JUMP) | (ratio <= 1 / PRICE_JUMP)
            log_qa(moc, "listings", "price_jump", len(common), jump.sum(), "Chỉ báo cáo",
                   condition=f"giá đổi >= {PRICE_JUMP} lần so với mốc trước",
                   reason="có thể do lỗi nhập hoặc giá đổi định nghĩa giữa các mốc",
                   note="lưu ý: giá giữa các mốc không cùng định nghĩa (giá niêm yết / giá báo / thiếu)")
        log_qa(moc, "reviews", "repeated_from_prev", len(cur_r), len(cur_r & seen_reviews), "Chỉ báo cáo",
               condition="review_id đã có ở mốc trước", reason="reviews tích luỹ qua các mốc; đếm theo thời gian phải bỏ trùng theo review_id",
               note="tập review giống hệt mốc trước" if cur_r and cur_r <= seen_reviews else "")
    prev_l, seen_reviews = cur_l, seen_reviews | cur_r

# Xuất báo cáo QA vào file qa_report.csv
pd.DataFrame(QA_LOG).to_csv("reports/qa_report.csv", index=False)
print("\nHoàn tất việc lọc và làm sạch! Đã lưu báo cáo tại reports/qa_report.csv")