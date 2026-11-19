import argparse
import os
import urllib.request
import yaml

os.makedirs("data/raw", exist_ok=True)

# ĐỌC CẤU HÌNH TỪ FILE CONFIG ĐƯỢC TRUYỀN VÀO TỪ TERMINAL BẰNG THAM SỐ --config, MẶC ĐỊNH SẼ LÀ madrid.yaml
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
args = parser.parse_args()

# Lấy cấu hình từ file YAML
with open(args.config, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

data_root = config["data_root"]
danh_sach_moc = [str(x) for x in config["snapshots"]]
cac_tep_csv = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

print("Đang tải ...")

for moc in danh_sach_moc:
    
    url2 = f"https://data.insideairbnb.com/{data_root}/{moc}/visualisations/neighbourhoods.geojson"
    ten_file_luu_2 = f"data/raw/{moc}_visualisations_neighbourhoods.geojson"

    if not os.path.exists(ten_file_luu_2):
        print(f"Đang tải geojson của mốc {moc} ...")
        urllib.request.urlretrieve(url2, ten_file_luu_2)
    else:
        print(f"Bỏ qua,geojson của mốc {moc} đã có trong máy")

    for tep in cac_tep_csv:
        url1 = f"https://data.insideairbnb.com/{data_root}/{moc}/data/{tep}"
        ten_file_luu_1 = f"data/raw/{moc}_{tep}"

        if not os.path.exists(ten_file_luu_1):
            print(f"Đang tải {moc}_{tep}...")
            urllib.request.urlretrieve(url1, ten_file_luu_1)
        else:
            print(f"Bỏ qua, {moc}-{tep} đã có trong máy.")

print("\nHoàn tất tải 16 tệp dữ liệu!")