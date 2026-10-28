import pandas as pd
import os
import yaml
import argparse

def calculate_kpi_for_snapshot(moc):
    """Hàm tính toán bộ KPI cho một mốc thời gian cụ thể"""
    kpi_dict = {"snapshot": moc}
    
    # 1. ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH
    path_list = f"data/processed/{moc}_listings.csv"
    path_cal = f"data/processed/{moc}_calendar.csv"
    path_rev = f"data/processed/{moc}_reviews.csv"
    
    # --- A. KPI TỪ BẢNG LISTINGS ---
    if os.path.exists(path_list):
        df_l = pd.read_csv(path_list)
        kpi_dict["total_active_listings"] = len(df_l)
        
        if "price" in df_l.columns:
            # Tính giá trung bình và trung vị (Median phản ánh thực tế tốt hơn trung bình cộng)
            kpi_dict["average_price"] = round(df_l["price"].mean(), 2)
            kpi_dict["median_price"] = round(df_l["price"].median(), 2)
            
        if "room_type" in df_l.columns:
            # Đếm số lượng phòng Entire home/apt
            kpi_dict["entire_home_count"] = (df_l["room_type"] == "Entire home/apt").sum()
            # Tỷ lệ % phòng nguyên căn
            kpi_dict["entire_home_pct"] = round((kpi_dict["entire_home_count"] / len(df_l)) * 100, 2)
    else:
        print(f"[Cảnh báo] Không tìm thấy file {path_list}")
        
    # --- B. KPI TỪ BẢNG CALENDAR ---
    if os.path.exists(path_cal):
        df_c = pd.read_csv(path_cal)
        if "available" in df_c.columns and not df_c.empty:
            # available = 1 (trống, chưa ai đặt), 0 (đã bị đặt hoặc chủ nhà khóa)
            # Tỷ lệ lấp đầy (Occupancy Rate) = Số ngày không trống / Tổng số ngày
            availability_rate = df_c["available"].mean()
            kpi_dict["occupancy_rate_pct"] = round((1 - availability_rate) * 100, 2)
    
    # --- C. KPI TỪ BẢNG REVIEWS ---
    if os.path.exists(path_rev):
        df_r = pd.read_csv(path_rev, dtype={"id": str})
        kpi_dict["total_reviews"] = len(df_r)
        
        # Số review trung bình mỗi phòng
        if "total_active_listings" in kpi_dict and kpi_dict["total_active_listings"] > 0:
            kpi_dict["reviews_per_listing"] = round(len(df_r) / kpi_dict["total_active_listings"], 2)

    return kpi_dict


def main():
    # Đọc tham số cấu hình
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/madrid.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    snapshots = [str(x) for x in cfg["snapshots"]]
    
    print("Bắt đầu tính toán KPI...")
    all_kpis = []
    
    # Lặp qua từng mốc để tính KPI
    for moc in snapshots:
        print(f"Đang xử lý KPI cho mốc: {moc}")
        kpi_result = calculate_kpi_for_snapshot(moc)
        all_kpis.append(kpi_result)
        
    # Gom tất cả vào 1 bảng DataFrame
    df_kpi = pd.DataFrame(all_kpis)
    
    # Xuất ra file tổng hợp
    out_path = "data/processed/kpi_summary.csv"
    df_kpi.to_csv(out_path, index=False)
    
    print(f"\nHoàn tất! Đã lưu báo cáo KPI tại: {out_path}")
    print("\n--- XEM TRƯỚC BẢNG KPI ---")
    print(df_kpi)

if __name__ == "__main__":
    main()