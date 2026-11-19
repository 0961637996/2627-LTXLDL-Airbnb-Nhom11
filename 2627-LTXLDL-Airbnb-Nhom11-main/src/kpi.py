import pandas as pd
import numpy as np
import os
import json
import yaml
import argparse

# Các KPI (đề: mỗi KPI có tên -> mục tiêu -> công thức -> đơn vị; ghi rõ tiền tệ). Xuất ra kpi_catalog.csv
def tao_catalog(cur, stakeholder):
    return pd.DataFrame([
        dict(kpi_id="K1", kpi_name="median_price", unit=f"{cur}/đêm", stakeholder=stakeholder,
             objective="Mặt bằng giá để so sánh khu vực. KHÔNG so sánh trực tiếp giữa các mốc vì định nghĩa giá khác nhau.",
             formula="trung vị price của các listing có giá hợp lệ (không thiếu giá, không bị cờ ngoại lai). avg_price là thống kê đi kèm."),
        dict(kpi_id="K2", kpi_name="unavailable_days_pct", unit="%", stakeholder=stakeholder,
             objective="Áp lực cầu: ngày không còn trống càng nhiều thì thị trường càng kín. Gồm cả ngày đã đặt lẫn chủ nhà tự chặn lịch.",
             formula="100 x tổng unavailable_days / tổng total_days trong calendar 365 ngày (bỏ các dòng bị cờ ngày/cửa sổ/mồ côi)."),
        dict(kpi_id="K3", kpi_name="revpar_proxy", unit=f"{cur}/đêm", stakeholder=stakeholder,
             objective="Doanh thu ước tính mỗi listing-đêm. Chỉ là ước lượng cận trên vì ngày không trống chưa chắc là ngày đã đặt.",
             formula="median_price x unavailable_days_pct / 100. Rỗng khi mốc không có giá."),
        dict(kpi_id="K4", kpi_name="avg_reviews_per_listing", unit="review/listing", stakeholder=stakeholder,
             objective="Mức độ hoạt động lịch sử của listing.",
             formula="trung bình number_of_reviews của listing (toàn thời gian, không phải số review trong mẫu)."),
        dict(kpi_id="K5", kpi_name="entire_home_pct", unit="%", stakeholder=stakeholder,
             objective="Tỷ trọng căn hộ nguyên căn - nhóm tác động nhiều nhất đến nhà ở dài hạn.",
             formula="100 x số listing room_type='Entire home/apt' / tổng listing."),
        dict(kpi_id="K6", kpi_name="zero_review_pct", unit="%", stakeholder=stakeholder,
             objective="Tỷ lệ listing chưa từng có đánh giá (rủi ro chất lượng, listing 'chết').",
             formula="100 x số listing có number_of_reviews = 0 / tổng listing."),
        dict(kpi_id="K7", kpi_name="data_error_pct", unit="%", stakeholder=stakeholder,
             objective="Độ tin cậy của KPI giá: tỷ lệ listing bị loại khỏi tính giá.",
             formula="100 x số listing thiếu giá hoặc bị cờ giá ngoại lai/không hợp lệ / tổng listing."),
        dict(kpi_id="K8", kpi_name="negative_share", unit="%", stakeholder=stakeholder,
             objective="Điểm đau của khách theo khu vực và theo quý, đo bằng cảm xúc đánh giá do LLM trích xuất.",
             formula="100 x số đánh giá sentiment='negative' / số đánh giá có sentiment hợp lệ (positive|mixed|negative)."),
    ])


def tinh_kpi(df):
    """Tính các KPI K1-K7 cho một nhóm listing (toàn thành phố hoặc một khu vực)."""
    kpi = {"n_listings": len(df)}

    # Giá: chỉ dùng listing có giá hợp lệ. Nếu không có giá nào thì để RỖNG (không phải 0!)
    df_valid_price = df[(df["flag_missing_price"] == 0) & (df["flag_price_outlier"] == 0)]
    kpi["n_price"] = len(df_valid_price)
    kpi["avg_price"] = round(df_valid_price["price"].mean(), 2) if len(df_valid_price) else np.nan
    kpi["median_price"] = round(df_valid_price["price"].median(), 2) if len(df_valid_price) else np.nan

    # Tỷ lệ ngày không còn trống (gồm cả ngày đã đặt và ngày chủ nhà chặn lịch)
    total_days = df["total_days"].sum()
    unavailable = (df["unavailable_days"].sum() / total_days * 100) if total_days > 0 else np.nan
    kpi["unavailable_days_pct"] = round(unavailable, 2)

    # RevPAR ước tính = giá trung vị x tỷ lệ ngày không trống
    kpi["revpar_proxy"] = (round(kpi["median_price"] * unavailable / 100, 2)
                           if pd.notna(kpi["median_price"]) and pd.notna(unavailable) else np.nan)

    # Số review thực tế của listing (number_of_reviews), không phải số review trong mẫu 5000
    kpi["avg_reviews_per_listing"] = round(pd.to_numeric(df["number_of_reviews"], errors="coerce").mean(), 2)

    n = len(df)
    kpi["entire_home_pct"] = round((df["room_type"] == "Entire home/apt").sum() / n * 100, 2) if n else np.nan
    kpi["zero_review_pct"] = round(df["flag_zero_reviews"].sum() / n * 100, 2) if n else np.nan
    kpi["data_error_pct"] = round(((df["flag_missing_price"] == 1) | (df["flag_price_outlier"] == 1)).sum() / n * 100, 2) if n else np.nan
    return kpi


def tinh_llm(cfg, snapshots, area_col, min_n):
    """K8: tỷ lệ đánh giá tiêu cực (từ kết quả LLM) theo khu vực và theo quý. Trả None nếu chưa có kết quả LLM."""
    path = (cfg.get("llm") or {}).get("cache", "data/processed/llm_cache.jsonl")
    if not os.path.exists(path):
        print(f"  Không thấy {path}: bỏ qua K8 (chạy llm.py trước, hoặc dùng --skip-llm).")
        return None
    if path.endswith(".csv"):
        res = pd.read_csv(path, dtype=str)
    else:
        recs = []
        for line in open(path, encoding="utf-8"):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                   # dòng hỏng do ngắt giữa chừng
        res = pd.DataFrame(recs)
    if "review_id" not in res.columns and "id" in res.columns:
        res = res.rename(columns={"id": "review_id"})
    if not {"review_id", "sentiment"} <= set(res.columns):
        raise SystemExit(f"{path}: cần cột review_id và sentiment, hiện có {list(res.columns)}")
    res["review_id"] = res["review_id"].astype(str)
    res["sentiment"] = res["sentiment"].astype(str).str.strip().str.lower()
    res = res.drop_duplicates("review_id", keep="last")
    n_all = len(res)
    res = res[res["sentiment"].isin(["positive", "mixed", "negative"])]

    last = snapshots[-1]
    rv = pd.read_csv(f"data/processed/{last}_reviews.csv", usecols=["id", "listing_id", "date"], dtype={"id": str, "listing_id": str})
    rv = rv.rename(columns={"id": "review_id"}).drop_duplicates("review_id")
    ms = pd.read_csv(f"data/processed/{last}_master_clean.csv", dtype={"id": str})
    area_cols = [c for c in ("analysis_area", area_col) if c in ms.columns]
    df = res.merge(rv, on="review_id", how="inner").merge(ms[["id"] + area_cols], left_on="listing_id", right_on="id", how="left")
    df["quarter"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("Q").astype(str)
    print(f"  K8: {len(df)}/{n_all} đánh giá dùng được")

    rows = []
    def them(level, area, period, g):
        rows.append(dict(kpi_id="K8", kpi_name="negative_share", level=level, area=area, period=period,
                         value=round((g["sentiment"] == "negative").mean() * 100, 2), n=len(g), unit="%",
                         reliable=len(g) >= min_n))
    them("city", "ALL", "ALL", df)
    for c in area_cols:
        for a, g in df.groupby(c):
            them("neighbourhood" if c == area_col else c, a, "ALL", g)
    for q, g in df[df["quarter"] != "NaT"].groupby("quarter"):
        them("quarter", "ALL", q, g)
    return pd.DataFrame(rows)


def main():
    # 1. Đọc cấu hình
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/madrid.yaml")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    snapshots = [str(x) for x in cfg["snapshots"]]
    cur = str(cfg.get("currency", "")).upper() or "currency"
    kcfg = cfg.get("kpi") or {}
    area_col = kcfg.get("area_col", "neighbourhood_cleansed")
    min_n_listings = kcfg.get("min_n_listings", 10)        # khu vực < N listing: reliable = False
    min_n_reviews = kcfg.get("min_n_reviews", 30)
    stakeholder = kcfg.get("stakeholder", "Chính quyền thành phố (cơ quan quản lý du lịch / nhà ở)")
    all_kpis, area_kpis = [], []

    print("Bắt đầu phân tích KPI từ dữ liệu Master Clean...\n")

    # 2. Tính toán cho từng mốc thời gian
    for moc in snapshots:
        file_path = f"data/processed/{moc}_master_clean.csv"

        if not os.path.exists(file_path):
            print(f"  [CẢNH BÁO] Không tìm thấy file {file_path}. Bỏ qua mốc này.")
            continue

        print(f"  -> Đang phân tích mốc: {moc}")
        df = pd.read_csv(file_path, dtype={"id": str})
        if len(df) == 0:
            continue

        # Toàn thành phố (KPI theo thời gian)
        all_kpis.append({"snapshot": moc, **tinh_kpi(df)})

        # Theo khu vực (đề: ít nhất một KPI theo neighbourhood_cleansed)
        for level, col in (("neighbourhood", area_col), ("analysis_area", "analysis_area")):
            if col in df.columns:
                for a, g in df.groupby(col):
                    area_kpis.append({"snapshot": moc, "level": level, "area": a, **tinh_kpi(g),
                                      "reliable": len(g) >= min_n_listings})

    # 3. Tổng hợp và xuất báo cáo
    tao_catalog(cur, stakeholder).to_csv("data/processed/kpi_catalog.csv", index=False)
    if all_kpis:
        df_kpi = pd.DataFrame(all_kpis)
        df_kpi.to_csv("data/processed/kpi_summary.csv", index=False)
        pd.DataFrame(area_kpis).to_csv("data/processed/kpi_by_area.csv", index=False)

        print("\nHoàn tất! Đã lưu: kpi_summary.csv (theo mốc), kpi_by_area.csv (theo khu vực), kpi_catalog.csv")
        print(f"(tiền tệ: {cur}; ô trống = không có dữ liệu hợp lệ, ví dụ mốc không có giá)")
        print("\n--- KẾT QUẢ PHÂN TÍCH BIẾN ĐỘNG THEO MỐC ---")
        print(df_kpi.to_string(index=False))  # In bảng thẳng hàng, không cần tabulate

        # K8: KPI dùng kết quả LLM
        if args.skip_llm:
            print("\n--skip-llm: bỏ qua K8.")
        else:
            llm = tinh_llm(cfg, snapshots, area_col, min_n_reviews)
            if llm is not None:
                llm.to_csv("data/processed/kpi_llm.csv", index=False)
                print(f"-> kpi_llm.csv: {len(llm)} dòng")
    else:
        print("\nKhông có dữ liệu KPI nào được tạo ra.")


if __name__ == "__main__":
    main()