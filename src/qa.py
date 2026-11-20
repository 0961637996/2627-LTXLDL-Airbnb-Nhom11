import argparse
import os
import numpy as np
import pandas as pd
import yaml

# ============================================================ CẤU HÌNH
parser = argparse.ArgumentParser(description="Lọc dữ liệu phân công và chạy bộ quy tắc QA")
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
with open(parser.parse_args().config, encoding="utf-8") as f:
    config = yaml.safe_load(f)

danh_sach_moc = [str(x) for x in config["snapshots"]]
for d in ["data/processed", "data/samples", "reports"]:
    os.makedirs(d, exist_ok=True)

qa_cfg = config.get("qa") or {}
MIN_PRICE = qa_cfg.get("min_price", 1)
MAX_PRICE = qa_cfg.get("max_price", 1500)
VALID_AREAS = qa_cfg.get("valid_areas", ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"])
VALID_ROOM_TYPES = qa_cfg.get("valid_room_types", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
AREA_RANGE = qa_cfg.get("expected_area_range", [120, 130])
AUTOMATED = qa_cfg.get("automated_patterns", [
    "this is an automated posting",
    "the host canceled this reservation",
    "the reservation was canceled",
])

QA_LOG = []


def log_qa(snap, table, rule, total, affected, action, condition="", reason="",
           kpi_name="", kpi_before=None, kpi_after=None, note=""):
    """Ghi nhận 1 quy tắc QA theo đúng cấu trúc: tên -> điều kiện -> lý do -> hành động."""
    total, affected = int(total), int(affected)
    fmt = lambda v: None if v is None or pd.isna(v) else round(float(v), 4)
    QA_LOG.append({
        "snapshot": snap, "table": table, "rule_name": rule,
        "condition": condition, "reason": reason,
        "total_rows": total, "affected_rows": affected,
        "pct_affected": round(100 * affected / total, 2) if total else 0.0,
        "action": action, "kpi_name": kpi_name,
        "kpi_before": fmt(kpi_before), "kpi_after": fmt(kpi_after),
        "note": note,
    })


def to_number(s):
    """Chuyển '$1,234.50' -> 1234.5; chuỗi lỗi/rỗng -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


# ============================================================ ĐỌC FILE PHÂN CÔNG
df_listing = (
    pd.read_csv("data/raw/phan_cong_1000_listing.csv", dtype=str)
    .rename(columns={"listing_id": "id"})
    .drop_duplicates("id")
)
list_ids = set(df_listing["id"].dropna())

df_review = (
    pd.read_csv("data/raw/phan_cong_5000_review.csv", dtype=str)
    .rename(columns={"review_id": "id"})
    .drop_duplicates("id")
)

# ============================================================ XỬ LÝ TỪNG MỐC
for moc in danh_sach_moc:
    print(f"\n--- Đang xử lý QA cho mốc: {moc} ---")

    # Xóa file cũ để tránh ghi nối đè dữ liệu của lần chạy trước
    for t in ("listings", "calendar", "reviews"):
        old = f"data/processed/{moc}_{t}.csv"
        if os.path.exists(old):
            os.remove(old)

    # -------------------------------- 1. BẢNG LISTINGS --------------------------------
    df_l = pd.read_csv(f"data/raw/{moc}_listings.csv.gz", dtype={"id": str})
    extra_l = [c for c in df_listing.columns if c == "id" or c not in df_l.columns]
    df_l = df_l.merge(df_listing[extra_l], on="id", how="inner")
    n_l = len(df_l)

    if "price" in df_l.columns:
        # Kiểu sai: ép price sang số thực (bắt buộc là numeric để kpi.py đọc)
        raw_price = df_l["price"]
        n_dirty_price = raw_price.astype(str).str.contains(r"[\$,]", regex=True, na=False).sum()
        price = to_number(raw_price)
        unparseable = raw_price.notna() & price.isna()

        # Thiếu & giá không hợp lệ (< MIN_PRICE) -> gộp chung vào cờ flag_invalid_price cho kpi.py
        missing_mask = price.isna()
        neg_mask = price.notna() & (price < MIN_PRICE)
        median_before_neg = price.median()
        price = price.mask(neg_mask, np.nan)
        df_l["price"] = price
        df_l["flag_invalid_price"] = missing_mask | neg_mask

        # Ngoại lai (> MAX_PRICE) -> cờ flag_outlier_price cho kpi.py
        outlier_mask = df_l["price"] > MAX_PRICE
        df_l["flag_outlier_price"] = outlier_mask

        pct_missing = 100 * missing_mask.mean() if n_l else 0
        log_qa(moc, "listings", "type_cast_price", n_l, n_dirty_price, "Xóa ký tự $ , và ép sang float",
               condition="price chứa $ hoặc ,", reason="chuỗi không tính toán được")
        log_qa(moc, "listings", "price_unparseable", n_l, unparseable.sum(), "Đặt NaN",
               condition="price khác rỗng nhưng không ép được sang số", reason="định dạng lỗi")
        log_qa(moc, "listings", "missing_price", n_l, missing_mask.sum(), "Gắn cờ flag_invalid_price",
               condition="price rỗng (NaN)", reason="không tính được KPI giá, giữ dòng cho KPI khác",
               note="MỐC NÀY THIẾU >= 50% GIÁ" if pct_missing >= 50 else "")
        log_qa(moc, "listings", "invalid_price", n_l, neg_mask.sum(), "Đặt NaN và gắn cờ flag_invalid_price",
               condition=f"price < {MIN_PRICE}", reason="giá <= 0 là vô lý",
               kpi_name="median_price", kpi_before=median_before_neg, kpi_after=df_l["price"].median())
        log_qa(moc, "listings", "outlier_price", n_l, outlier_mask.sum(), "Gắn cờ flag_outlier_price",
               condition=f"price > {MAX_PRICE}", reason="ngoại lai làm lệch phân phối giá; loại khỏi K1, K3",
               kpi_name="median_price", kpi_before=df_l["price"].median(),
               kpi_after=df_l.loc[~outlier_mask, "price"].median())

    # Danh mục lệch: vùng phân tích và loại phòng
    if "analysis_area" in df_l.columns:
        bad_area = ~df_l["analysis_area"].isin(VALID_AREAS)
        df_l["flag_invalid_area"] = bad_area
        log_qa(moc, "listings", "invalid_area", n_l, bad_area.sum(), "Gắn cờ flag_invalid_area",
               condition=f"analysis_area ngoài {VALID_AREAS}", reason="sai lệch khi gom nhóm theo vùng")

    if "room_type" in df_l.columns:
        df_l["room_type"] = df_l["room_type"].astype(str).str.strip()
        bad_room = ~df_l["room_type"].isin(VALID_ROOM_TYPES)
        df_l["flag_invalid_room_type"] = bad_room
        log_qa(moc, "listings", "invalid_room_type", n_l, bad_room.sum(), "Gắn cờ flag_invalid_room_type",
               condition=f"room_type ngoài {VALID_ROOM_TYPES}", reason="sai lệch khi tính K5 (entire home)")

    # Trùng lặp ID trong cùng 1 mốc (keep="first" khớp với drop_duplicates("id") ở kpi.py)
    dup_l = df_l.duplicated(subset=["id"], keep="first")
    df_l["flag_dup_id"] = dup_l
    log_qa(moc, "listings", "duplicate_id_in_snap", n_l, dup_l.sum(), "Gắn cờ flag_dup_id",
           condition="trùng id trong cùng mốc", reason="tránh đếm đôi listing")

    df_l.to_csv(f"data/processed/{moc}_listings.csv", index=False)

    # Kiểm tra nhanh phân bổ điều kiện
    if "analysis_area" in df_l.columns:
        area_counts = df_l["analysis_area"].value_counts()
        lech = area_counts[(area_counts < AREA_RANGE[0]) | (area_counts > AREA_RANGE[1])]
        log_qa(moc, "listings", "area_balance", len(area_counts), len(lech), "Chỉ báo cáo",
               condition=f"số listing mỗi vùng ngoài {AREA_RANGE}", reason="kiểm tra cân bằng mẫu phân công",
               note=str(area_counts.to_dict()))
    print(f"  Listings: {n_l} dòng | thiếu/lỗi giá: {int(df_l['flag_invalid_price'].sum()) if 'flag_invalid_price' in df_l else 0} | ngoại lai: {int(df_l['flag_outlier_price'].sum()) if 'flag_outlier_price' in df_l else 0}")

    # -------------------------------- 2. BẢNG CALENDAR --------------------------------
    first_chunk = True
    n_c = n_cast_avail = n_bad_avail = n_bad_date = 0
    sum_all = cnt_all = sum_ok = cnt_ok = 0
    cal_cols = {"listing_id", "date", "available"}

    for chunk in pd.read_csv(
        f"data/raw/{moc}_calendar.csv.gz",
        chunksize=100_000,
        usecols=lambda c: c in cal_cols,
        dtype={"listing_id": str},
    ):
        chunk = chunk[chunk["listing_id"].isin(list_ids)].copy()
        if chunk.empty:
            continue
        n_c += len(chunk)

        # Kiểu sai: chuyển t/f -> 1/0 cho kpi.py
        if "available" in chunk.columns:
            mapped = chunk["available"].astype(str).str.lower().map({"t": 1, "f": 0, "true": 1, "false": 0})
            n_cast_avail += mapped.notna().sum()
            n_bad_avail += (chunk["available"].notna() & mapped.isna()).sum()
            chunk["available"] = mapped

        # Kiểm tra ngày sai định dạng
        bad_date = pd.to_datetime(chunk["date"], errors="coerce").isna()
        chunk["flag_bad_date"] = bad_date
        n_bad_date += bad_date.sum()

        sum_all += chunk["available"].sum()
        cnt_all += chunk["available"].notna().sum()
        sum_ok += chunk.loc[~bad_date, "available"].sum()
        cnt_ok += chunk.loc[~bad_date, "available"].notna().sum()

        chunk.to_csv(f"data/processed/{moc}_calendar.csv", mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False

    cal_path = f"data/processed/{moc}_calendar.csv"
    n_dup_cal = pd.read_csv(cal_path, usecols=["listing_id", "date"], dtype=str).duplicated().sum() if os.path.exists(cal_path) else 0

    log_qa(moc, "calendar", "type_cast_available", n_c, n_cast_avail, "Chuyển t/f sang 1/0",
           condition="available là chuỗi t/f", reason="cần kiểu số để tính K2 (occupancy)")
    log_qa(moc, "calendar", "invalid_available", n_c, n_bad_avail, "Đặt NaN",
           condition="available khác t/f/true/false", reason="giá trị ngoài miền hợp lệ")
    log_qa(moc, "calendar", "invalid_date", n_c, n_bad_date, "Gắn cờ flag_bad_date",
           condition="date không đọc được định dạng ngày", reason="ảnh hưởng khi lọc cửa sổ occupancy_days",
           kpi_name="availability_rate",
           kpi_before=sum_all / cnt_all if cnt_all else None,
           kpi_after=sum_ok / cnt_ok if cnt_ok else None)
    log_qa(moc, "calendar", "duplicate_listing_date", n_c, n_dup_cal, "Chỉ báo cáo",
           condition="trùng cặp (listing_id, date)", reason="đếm đôi số ngày trống/kín")

    # -------------------------------- 3. BẢNG REVIEWS & TÁCH MẪU LLM --------------------------------
    first_chunk = True
    n_r = n_html = n_empty = 0
    sum_len_before = cnt_len_before = 0

    for chunk in pd.read_csv(f"data/raw/{moc}_reviews.csv.gz", chunksize=100_000, dtype={"id": str, "listing_id": str}):
        extra_r = [c for c in df_review.columns if c == "id" or c not in chunk.columns]
        chunk = chunk.merge(df_review[extra_r], on="id", how="inner")
        if chunk.empty:
            continue
        n_r += len(chunk)

        if "comments" in chunk.columns:
            raw_c = chunk["comments"].fillna("")
            sum_len_before += raw_c.str.len().sum()
            cnt_len_before += len(chunk)

            has_html = raw_c.str.contains(r"<[^>]+>", regex=True)
            n_html += has_html.sum()
            chunk["comments"] = raw_c.str.replace(r"<[^>]+>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

            empty_mask = chunk["comments"] == ""
            n_empty += empty_mask.sum()
            chunk["flag_empty_comment"] = empty_mask

        chunk.to_csv(f"data/processed/{moc}_reviews.csv", mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False

    rev_path = f"data/processed/{moc}_reviews.csv"
    if os.path.exists(rev_path):
        df_r = pd.read_csv(rev_path, dtype=str)
        text = df_r["comments"].fillna("")
        mean_before = sum_len_before / cnt_len_before if cnt_len_before else None
        mean_clean = text.str.len().mean()

        dup_r = df_r.duplicated(subset=["id"], keep="first")
        auto_r = text.str.contains("|".join(AUTOMATED), case=False, regex=True)
        empty_r = df_r["flag_empty_comment"].astype(str) == "True"
        df_r["flag_dup_id"] = dup_r
        df_r["flag_automated_comment"] = auto_r
        df_r.to_csv(rev_path, index=False)

        mean_after = text.loc[~(empty_r | auto_r)].str.len().mean()
        log_qa(moc, "reviews", "clean_html", n_r, n_html, "Xóa thẻ HTML và chuẩn hóa khoảng trắng",
               condition="comments chứa thẻ <...>", reason="giảm nhiễu văn bản cho bước LLM",
               kpi_name="mean_comment_chars", kpi_before=mean_before, kpi_after=mean_clean)
        log_qa(moc, "reviews", "empty_comment", n_r, n_empty, "Gắn cờ flag_empty_comment",
               condition="comments rỗng hoặc chỉ có khoảng trắng", reason="không có nội dung phân tích cảm xúc")
        log_qa(moc, "reviews", "automated_comment", n_r, auto_r.sum(), "Gắn cờ flag_automated_comment",
               condition="khớp mẫu thông báo hủy phòng tự động", reason="không phải nhận xét của khách",
               kpi_name="mean_comment_chars", kpi_before=mean_clean, kpi_after=mean_after)
        log_qa(moc, "reviews", "duplicate_id_in_snap", n_r, dup_r.sum(), "Gắn cờ flag_dup_id",
               condition="trùng id review trong cùng mốc", reason="tránh đếm đôi đánh giá")

        # Tách mẫu LLM (1000 dòng) và mẫu nhãn tay (100 dòng) ở mốc đầu, bảo toàn nhãn đã gán tay
        if moc == danh_sach_moc[0] and {"mau_llm", "nhan_tay"} <= set(df_r.columns):
            hand_path = "data/samples/hand_labeled_reviews.csv"
            old_labels = {}
            if os.path.exists(hand_path):
                old = pd.read_csv(hand_path, dtype=str)
                if {"review_id", "nhan"} <= set(old.columns):
                    old_labels = dict(old.dropna(subset=["nhan"])[["review_id", "nhan"]].values)

            llm_sample = df_r[df_r["mau_llm"].fillna("").str.lower() == "x"]
            hand_sample = df_r[df_r["nhan_tay"].fillna("").str.lower() == "x"]
            if old_labels:
                kept = df_r[df_r["id"].isin(old_labels)]
                llm_sample = pd.concat([llm_sample, kept]).drop_duplicates("id")
                hand_sample = pd.concat([hand_sample, kept]).drop_duplicates("id")

            cols = [c for c in ("id", "listing_id", "date", "comments") if c in df_r.columns]
            llm_sample[cols].rename(columns={"id": "review_id"}).to_csv("data/samples/llm_reviews.csv", index=False)

            hand_out = hand_sample[cols].rename(columns={"id": "review_id"}).copy()
            hand_out["nhan"] = hand_out["review_id"].map(old_labels)
            hand_out.to_csv(hand_path, index=False)
            print(f"  -> Đã tách mẫu LLM: {len(llm_sample)} dòng | Nhãn tay: {len(hand_out)} dòng (đã gán: {hand_out['nhan'].notna().sum()})")

# ============================================================ KIỂM TRA TRÙNG / LỆCH GIỮA CÁC MỐC
prev_l, seen_reviews = None, set()
for moc in danh_sach_moc:
    cur_l = pd.read_csv(f"data/processed/{moc}_listings.csv", dtype={"id": str}).drop_duplicates("id").set_index("id")
    rev_path = f"data/processed/{moc}_reviews.csv"
    cur_r = set(pd.read_csv(rev_path, usecols=["id"], dtype=str)["id"]) if os.path.exists(rev_path) else set()

    if prev_l is not None:
        common = cur_l.index.intersection(prev_l.index)
        log_qa(moc, "listings", "gone_since_prev", len(prev_l), (~prev_l.index.isin(cur_l.index)).sum(), "Chỉ báo cáo",
               condition="id có ở mốc trước nhưng biến mất ở mốc này", reason="theo dõi biến động nguồn cung giữa các mốc")
        if "last_scraped" in cur_l.columns and "last_scraped" in prev_l.columns:
            stale = (cur_l.loc[common, "last_scraped"] == prev_l.loc[common, "last_scraped"]) & cur_l.loc[common, "last_scraped"].notna()
            log_qa(moc, "listings", "stale_record_repeated", len(common), stale.sum(), "Chỉ báo cáo",
                   condition="trùng cả id và last_scraped với mốc trước", reason="bản ghi cũ lặp lại giữa các mốc")
        log_qa(moc, "reviews", "repeated_from_prev", len(cur_r), len(cur_r & seen_reviews), "Chỉ báo cáo",
               condition="review_id đã xuất hiện ở mốc trước", reason="tránh cộng dồn trùng review giữa các mốc",
               note="tập review lặp lại 100% từ mốc trước" if cur_r and cur_r <= seen_reviews else "")

    prev_l = cur_l
    seen_reviews |= cur_r

# ============================================================ XUẤT BÁO CÁO QA
pd.DataFrame(QA_LOG).to_csv("reports/qa_report.csv", index=False)
print("\nHoàn tất QA! Đã lưu báo cáo tại reports/qa_report.csv")