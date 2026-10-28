import argparse
import os
import urllib.request
import yaml

os.makedirs("data/raw", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/madrid.yaml")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Lấy các trường cấu hình từ file YAML
data_root = config["data_root"]
danh_sach_moc = config["snapshots"]
cac_tep_csv = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

print("Đang tải ...")

for moc in danh_sach_moc:
    for tep in cac_tep_csv:
        # Bỏ chữ 'spain/' ở URL vì biến data_root đã chứa sẵn chuỗi này
        url = f"https://data.insideairbnb.com/{data_root}/{moc}/data/{tep}"
        ten_file_luu = f"data/raw/{moc}_{tep}"
        
        if not os.path.exists(ten_file_luu):
            print(f"Đang tải {moc}_{tep}...")
            urllib.request.urlretrieve(url, ten_file_luu)
        else:
            print(f"Bỏ qua, {moc}-{tep} đã có trong máy.")

print("\nHoàn tất tải 12 tệp dữ liệu!")