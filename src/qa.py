import pandas as pd
import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config",type = str,default="configs/madrid.yaml")
args = parser.parse_args()

with open(args.config,"r",encoding="utf-8") as file:
     config = yaml.safe_load(file)

danh_sach_moc=[str(a) for a in config["snapshots"]]

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/samples", exist_ok=True)

df1000 = pd.read_csv(
    "data/raw/phan_cong_1000_listing.csv",
    dtype={"listing_id":"str"},
)
df1000 = df1000.rename(columns={"listing_id":"id"})
id_can_tim = set(df1000["id"].dropna())

df5000 = pd.read_csv(
    "data/raw/phan_cong_5000_review.csv",
    dtype={"listing_id":"str","review_id":"str"},
)
df5000 = df5000.rename(columns={"review_id":"id"})

for moc in danh_sach_moc:
    first_chunk_listings=True
    for chunk in pd.read_csv(
        f"data/raw/{moc}_listings.csv.gz",
        chunksize = 100_000,
        dtype={"id":"str"}
    ):
        ket_qua1=chunk.merge(
            df1000,
            on="id",
            how ="inner"
        )
        if not ket_qua1.empty:
            ket_qua1.to_csv(
                f"data/processed/{moc}_listings.csv",
                mode = "w" if first_chunk_listings else "a",
                header = first_chunk_listings,
                index= False
            )
            first_chunk_listings = False

for moc in danh_sach_moc:
    first_chunk_calendar=True
    if moc == "2025-09-14":
        cols = ["listing_id","date","available","price"]
    else:
        cols = ["listing_id","date","available"]
    for chunk in pd.read_csv(
        f"data/raw/{moc}_calendar.csv.gz",
        chunksize=100_000,
        usecols=cols,
        dtype ={"listing_id":"str"},
    ):
        ket_qua2=chunk[chunk["listing_id"].isin(id_can_tim)]
        if not ket_qua2.empty:
            ket_qua2.to_csv(
                f"data/processed/{moc}_calendar.csv",
                mode ="w" if first_chunk_calendar else "a",
                header = first_chunk_calendar,
                index = False,
            )
            first_chunk_calendar = False

for moc in danh_sach_moc:
    first_chunk_reviews=True
    for chunk in pd.read_csv(
        f"data/raw/{moc}_reviews.csv.gz",
        chunksize=100_000,
        dtype ={"id":"str","listing_id":"str"},
    ):
        ket_qua3 = chunk.merge(
            df5000,
            on="id",
            how="inner",
        )
        if not ket_qua3.empty:
            ket_qua3.to_csv(
                f"data/processed/{moc}_reviews.csv",
                mode ="w" if first_chunk_reviews else "a",
                header = first_chunk_reviews,
                index = False,
            )
            first_chunk_reviews = False
    if moc == danh_sach_moc[0]:
        read = pd.read_csv(
            f"data/processed/{moc}_reviews.csv",
            dtype = str,
        )
        read1= read[read["mau_llm"]=="x"]
        read2= read[read["nhan_tay"]=="x"]
        read1.to_csv(
            "data/samples/llm_reviews.csv",
            index= False
        )
        read2.to_csv(
            "data/samples/hand_labeled_reviews.csv",
            index= False
        )


for moc in danh_sach_moc:
    print(moc)
    check =  pd.read_csv(f"data/processed/{moc}_listings.csv")

    room_type = check["room_type"].value_counts()
    apt = room_type.get('Entire home/apt', 0)
    pri = room_type.get('Private room', 0)
    print(f"Số lượng apt là {apt}, số lượng pri là {pri}")

    so_0_review = (check["number_of_reviews"].fillna(-1).astype(float)==0).sum()
    print(f"số lượng review = 0 là {so_0_review}")

    vung = check["analysis_area"].value_counts()
    v1 = vung.get("A01",0)
    v2 = vung.get("A02",0)
    v3 = vung.get("A03",0)
    v4 = vung.get("A04",0)
    v5 = vung.get("A05",0)
    v6 = vung.get("A06",0)
    v7 = vung.get("A07",0)
    v8 = vung.get("A08",0)

    print(f"{v1},{v2},{v3},{v4},{v5},{v6},{v7},{v8}")
    
    if moc == "2026-06-20":
        thieu_gia = check["price"].isna().sum()
        print(f"Số lượng thiếu price ở mốc cuối là: {thieu_gia}")