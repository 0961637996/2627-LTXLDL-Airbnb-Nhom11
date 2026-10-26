import os
import urllib.request

os.makedirs("data/raw", exist_ok=True)

danh_sach_moc = ["2025-09-14", "2025-12-14", "2026-03-24", "2026-06-20"]
cac_tep_csv = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

print("Đang tải ...")

for moc in danh_sach_moc:
    for tep in cac_tep_csv:
        url = f"https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/{moc}/data/{tep}"
        ten_file_luu = f"data/raw/{moc}_{tep}"
        
        if not os.path.exists(ten_file_luu):
                urllib.request.urlretrieve(url, ten_file_luu)
        else:
            print(f"Bỏ qua,{moc}-{tep} đã có trong máy.")

print("\nHoàn tất tải 12 tệp dữ liệu!")
