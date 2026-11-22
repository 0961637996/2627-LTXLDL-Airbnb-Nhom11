"""kpi.py - Tính KPI cho bên đặt hàng và ghi vào data/processed/kpi_*.csv.

Đọc : data/processed/<mốc>_listings.csv và <mốc>_calendar.csv   (đã qua qa.py)
      data/processed/llm_results.csv                            (do llm.py tạo; chưa có thì bỏ qua K6, K7)
Ghi  : data/processed/kpi_values.csv        mỗi dòng = 1 KPI x 1 mốc x 1 khu vực
       data/processed/kpi_definitions.csv   tên -> mục tiêu -> công thức -> đơn vị -> mức đúng chuẩn
Chạy : python src/kpi.py --config configs/madrid.yaml

Quy ước chung của các hàm f_*:
  - nhận một bảng (DataFrame), trả về (giá trị, n) hoặc (giá trị, n, chữ)
  - n = số dòng THỰC SỰ tham gia phép tính (sau khi bỏ NaN), không phải 1.000
  - tỷ lệ tính theo phần trăm (0-100)

Cấu hình (khối `kpi:` trong madrid.yaml, thiếu khoá nào thì dùng giá trị mặc định):
  area_cols         các cột khu vực để tách KPI       (mặc định: analysis_area, neighbourhood_cleansed)
  min_n             dưới ngưỡng này thì đánh dấu low_n (mặc định: 30, theo bài 13)
  host_threshold    ngưỡng "host nhiều căn"            (mặc định: 5)
  commercial_nights ngưỡng "mở bán thương mại"         (mặc định: 183 đêm)
  occupancy_days    chỉ tính mức khai thác trên N ngày đầu sau mỗi mốc (mặc định: để trống = cả 365 ngày)
  use_license       bật K8 (số đăng ký)                (mặc định: false)
"""
import argparse
import ast
import json
import os

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

PROCESSED = "data/processed"

# Các cột số lấy thẳng từ file gốc (không có quy tắc QA riêng). load_listings ép chúng về số MỘT lần,
# nên các hàm f_* không phải lặp lại việc ép kiểu. KPI mới cần cột số nào thì thêm vào đây.
NUMERIC_COLS = ["calculated_host_listings_count", "availability_365"]

# Giá trị của cột license được coi là "không có số đăng ký" (so sánh sau khi bỏ khoảng trắng, viết thường).
# Hãy chạy value_counts() trên cột license rồi chỉnh danh sách này cho khớp với dữ liệu thật của Madrid.
LICENSE_KHONG_HOP_LE = {"", "exempt", "n/a", "na", "none", "nan", "null"}


# ============================================================ HÀM PHỤ
def as_bool(series):
    """Đổi cột cờ về True/False thật (cờ có thể đang là chuỗi 'True'/'False')."""
    return series.astype(str).str.strip().str.lower().eq("true")


def flag(df, name):
    """Lấy cột cờ; nếu qa.py chưa tạo cột đó thì coi như không dòng nào bị gắn cờ."""
    if name in df.columns:
        return as_bool(df[name])
    return pd.Series(False, index=df.index)


def to_number(series):
    """Ép một cột về số; ký tự bẩn như '$1,200.00' hoặc chữ hỏng thành NaN."""
    if is_numeric_dtype(series):
        return series
    return pd.to_numeric(series.astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")


def parse_aspects(cell):
    """Đổi một ô aspects_* của llm_results.csv thành list tên khía cạnh (viết thường).

    Chấp nhận 3 cách lưu: danh sách JSON '["noise", "host"]', danh sách Python "['noise', 'host']"
    (khi ghi list ra CSV bằng pandas), hoặc chuỗi nối bằng ';' như 'noise;host'. Ô trống -> [].
    """
    if isinstance(cell, list):
        items = cell
    elif pd.isna(cell):
        return []
    else:
        text = str(cell).strip()
        if text.startswith("["):
            try:
                items = json.loads(text)
            except ValueError:
                try:
                    items = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    items = []
        else:
            items = text.replace(",", ";").split(";")
    return [str(a).strip().lower() for a in items if str(a).strip()]


# ============================================================ ĐỌC DỮ LIỆU
def load_listings(moc):
    """Đọc listings đã qua QA. Thêm cột `price_clean` = giá sau khi loại dòng bị gắn cờ.

    Cột price phải đã là số (qa.py chịu trách nhiệm); các cột trong NUMERIC_COLS được ép về số ở đây.
    """
    df = pd.read_csv(f"{PROCESSED}/{moc}_listings.csv", dtype={"id": str})
    df = df.drop_duplicates("id")  # mỗi listing chỉ tính một lần
    if not is_numeric_dtype(df["price"]):
        # qa.py phải đổi giá về số; nếu chưa thì báo lỗi thay vì lặng lẽ biến giá hỏng thành NaN
        raise ValueError(f"[kpi] cột price của mốc {moc} chưa là số, kiểm tra lại bước qa.py")
    bad_price = flag(df, "flag_invalid_price") | flag(df, "flag_outlier_price")
    df["price_clean"] = df["price"].where(~bad_price)  # giữ giá khi KHÔNG bị gắn cờ, còn lại thành NaN
    for col in NUMERIC_COLS:  # cột số khác: ép một lần ở đây (chữ hỏng thành NaN)
        if col in df.columns:
            df[col] = to_number(df[col])
    return df


def load_calendar(moc, days=None):
    """Đọc calendar (1 = còn trống, 0 = không còn trống). Nếu `days` có giá trị thì chỉ giữ N ngày đầu sau mốc.

    Vì sao cần `days`: các đêm ở xa thường còn trống chỉ vì chưa ai đặt, nên tính trên cả 365 ngày
    làm tỷ lệ bị thấp và khó so sánh giữa các mốc. Giới hạn 30 hoặc 90 ngày đầu sẽ công bằng hơn.
    """
    cols = ["listing_id", "available"] + (["date"] if days else [])
    cal = pd.read_csv(f"{PROCESSED}/{moc}_calendar.csv", usecols=cols, dtype={"listing_id": str})
    if not is_numeric_dtype(cal["available"]):  # qa.py chưa đổi t/f thành 1/0 thì đổi ở đây
        cal["available"] = cal["available"].astype(str).str.lower().map({"t": 1, "f": 0, "1": 1, "0": 0})
    cal = cal.dropna(subset=["available"])
    if days:
        start = pd.Timestamp(moc)
        dates = pd.to_datetime(cal["date"])
        cal = cal[(dates >= start) & (dates < start + pd.Timedelta(days=days))]
    return cal


def load_llm_reviews(last_listings, area_cols):
    """Đọc llm_results.csv và gắn khu vực của listing vào từng review. Trả về None nếu chưa dùng được.

    llm_results.csv cần các cột: listing_id, sentiment, aspects_negative (theo schema ở bài 11).
    """
    path = f"{PROCESSED}/llm_results.csv"
    if not os.path.exists(path):
        print("[kpi] chưa có llm_results.csv, bỏ qua K6, K7 (chạy llm.py trước)")
        return None
    llm = pd.read_csv(path, dtype={"review_id": str, "listing_id": str})
    missing = {"listing_id", "llm_sentiment", "llm_aspects_negative"} - set(llm.columns)
    if missing:
        print(f"[kpi] llm_results.csv thiếu cột {sorted(missing)}, bỏ qua K6, K7")
        return None
    llm["llm_sentiment"] = llm["llm_sentiment"].astype(str).str.strip().str.lower()
    llm = llm[llm["llm_sentiment"].isin(["positive", "mixed", "negative"])]  # bỏ dòng LLM trả sai định dạng
    cols = [c for c in area_cols if c in last_listings.columns]
    return llm.join(last_listings.set_index("id")[cols], on="listing_id")


# ============================================================ TÍNH THEO KHU VỰC
def by_level(df, area_cols, func, **kwargs):
    """Chạy hàm KPI `func` cho toàn thành phố, rồi cho từng khu vực theo mỗi cột trong area_cols.

    func(sub_df, **kwargs) trả về (giá trị, n) hoặc (giá trị, n, chữ).
    Trả về list các bộ (level, area, value, n, text).
    """
    groups = [("all", "ALL", df)]
    for col in area_cols:
        if col in df.columns:
            groups += [(col, str(area), sub) for area, sub in df.groupby(col)]
    out = []
    for level, area, sub in groups:
        res = func(sub, **kwargs)
        out.append((level, area, res[0], res[1], res[2] if len(res) > 2 else ""))
    return out


# ============================================================ TÍNH CỘT CHO TỪNG LISTING
def add_occupancy(listings, calendar):
    """Thêm vào bảng `listings` hai cột `occupancy` và `revpar`, rồi trả về bảng đó.

    - occupancy của một listing = 1 - trung bình `available` của listing đó trong calendar
      (`available` là "còn trống" nên tỷ lệ không trống là 1 TRỪ trung bình)
    - revpar của một listing = price_clean * occupancy (giá niêm yết x tỷ lệ đêm không trống)
    Listing không có dòng nào trong calendar hoặc không có giá hợp lệ sẽ có NaN ở cột tương ứng.
    """
    occupancy = 1 - calendar.groupby("listing_id")["available"].mean()  # Series: index = listing_id
    listings["occupancy"] = listings["id"].map(occupancy)
    listings["revpar"] = listings["price_clean"] * listings["occupancy"]
    return listings


# ============================================================ CÁC HÀM KPI (mỗi hàm một KPI)
def f_median_price(d):
    """K1. Giá niêm yết trung vị (proxy ADR). Đơn vị: EUR/đêm."""
    s = d["price_clean"].dropna()
    return s.median(), len(s)


def f_occupancy(d):
    """K2. Tỷ lệ đêm không còn trống (proxy occupancy, cận trên). Đơn vị: %."""
    s = d["occupancy"].dropna()
    return s.mean() * 100, len(s)


def f_revpar(d):
    """K3. RevPAR ước tính (trung vị theo listing). Đơn vị: EUR/đêm."""
    s = d["revpar"].dropna()
    return s.median(), len(s)


def f_host_chuyen_nghiep(d, nguong=5):
    """K4. Tỷ lệ listing thuộc host có từ `nguong` listing trở lên. Đơn vị: %.

    Bỏ NaN TRƯỚC khi so sánh: NaN >= 5 cho False và sẽ bị tính nhầm là "không chuyên nghiệp".
    Đơn vị đếm là listing (không phải host).
    """
    s = d["calculated_host_listings_count"].dropna()
    return (s >= nguong).mean() * 100, len(s)


def f_entire_home_thuong_mai(d, nguong=183):
    """K5. Tỷ lệ entire home mở bán thương mại (xấp xỉ FREH). Đơn vị: %.

    Mẫu số là các listing có room_type == "Entire home/apt" và có availability_365.
    """
    entire = d[d["room_type"] == "Entire home/apt"]
    s = entire["availability_365"].dropna()
    return (s >= nguong).mean() * 100, len(s)


def f_negative(d):
    """K6. Tỷ lệ review tiêu cực (LLM). Đơn vị: %. `d` là bảng review đã gắn khu vực."""
    return d["llm_sentiment"].eq("negative").mean() * 100, len(d)


def f_top_complaint(d):
    """K7. Khía cạnh bị chê nhiều nhất (LLM). Trả về (value, n, text).

    value = % review nhắc tới khía cạnh đứng đầu trong aspects_negative; text = tên khía cạnh đó.
    """
    counts = {}
    for cell in d["llm_aspects_negative"]:
        for aspect in set(parse_aspects(cell)):  # set: một review chỉ tính một lần cho mỗi khía cạnh
            counts[aspect] = counts.get(aspect, 0) + 1
    if not counts:
        return float("nan"), len(d), ""
    top = max(counts, key=counts.get)
    return counts[top] / len(d) * 100, len(d), top


def f_license(d):
    """K8 (chỉ chạy khi bật use_license). Tỷ lệ listing có số đăng ký. Đơn vị: %.

    Mẫu số là TẤT CẢ listing; listing để trống license được tính là chưa đăng ký.
    "Có đăng ký" = license không rỗng và không thuộc LICENSE_KHONG_HOP_LE (xem đầu file).
    """
    text = d["license"].fillna("").astype(str).str.strip().str.lower()
    return (~text.isin(LICENSE_KHONG_HOP_LE)).mean() * 100, len(d)


# ============================================================ ĐỊNH NGHĨA KPI (nhóm sửa cột `target` cho đúng bên đặt hàng)
def kpi_definitions(currency, host_threshold, commercial_nights, occupancy_days=None):
    cols = ["kpi_id", "name", "target", "formula", "unit", "fidelity", "source", "uses_llm"]
    window = f"{occupancy_days} ngày đầu sau mỗi mốc" if occupancy_days else "cả 365 ngày của calendar"
    rows = [
        ("K1", "Giá niêm yết trung vị", "Theo dõi mặt bằng giá, phát hiện khu vực tăng giá nhanh",
         "median(price) của listing có giá hợp lệ, không ngoại lai", f"{currency}/đêm",
         "xấp xỉ (giá niêm yết, không phải giá thực trả)", "listings", False),
        ("K2", "Tỷ lệ đêm không còn trống", "Ước lượng mức độ khai thác chỗ ở",
         f"trung bình trên listing của (1 - tỷ lệ đêm available), tính trên {window}", "%",
         "cận trên (gồm cả đêm host tự chặn lịch)", "calendar", False),
        ("K3", "RevPAR ước tính", "So sánh sức sinh lời giữa các khu vực và các mốc",
         "median(giá niêm yết x tỷ lệ đêm không trống) theo listing", f"{currency}/đêm",
         "xấp xỉ (kế thừa sai lệch của K1 và K2)", "listings + calendar", False),
        ("K4", "Tỷ lệ listing thuộc host nhiều căn", "Đo mức độ thương mại hoá",
         f"% listing có calculated_host_listings_count >= {host_threshold}", "%",
         "đúng nghĩa, trong phạm vi thành phố", "listings", False),
        ("K5", "Tỷ lệ entire home mở bán thương mại", "Đo khả năng căn hộ rời khỏi thị trường nhà ở dài hạn",
         f"% entire home có availability_365 >= {commercial_nights}", "%",
         "xấp xỉ thấp (không biết số đêm đã thuê)", "listings", False),
        ("K6", "Tỷ lệ review tiêu cực", "Đo mức hài lòng của khách theo khu vực",
         "% review do LLM gán sentiment = negative", "%",
         "đúng ý tưởng, mẫu 1.000 review", "llm_results", True),
        ("K7", "Khía cạnh bị chê nhiều nhất", "Chỉ ra vấn đề cần cải thiện ở từng khu vực",
         "khía cạnh xuất hiện nhiều nhất trong aspects_negative; value = % review nhắc tới nó", "%",
         "đúng ý tưởng, mẫu 1.000 review", "llm_results", True),
        ("K8", "Tỷ lệ listing có số đăng ký", "Đo mức tuân thủ quy định đăng ký",
         "% listing có license không rỗng và không thuộc nhóm 'không hợp lệ' (vd Exempt)", "%",
         "tuỳ chất lượng cột license", "listings", False),
    ]
    return pd.DataFrame(rows, columns=cols)


# ============================================================ GOM VÀ GHI KẾT QUẢ
def add_rows(rows, kpi_id, snapshot, results):
    """Chuyển kết quả của by_level thành các dòng dạng dài rồi thêm vào danh sách `rows`."""
    for level, area, value, n, text in results:
        rows.append({"kpi_id": kpi_id, "snapshot": snapshot, "level": level, "area": area,
                     "value": value, "value_text": text, "n": n})


def run_kpi(rows, skipped, kpi_id, snapshot, df, area_cols, func, kwargs=None):
    """Tính một KPI; thiếu cột thì ghi vào `skipped` thay vì làm dừng cả file."""
    try:
        add_rows(rows, kpi_id, snapshot, by_level(df, area_cols, func, **(kwargs or {})))
    except KeyError as err:
        skipped.setdefault(f"{kpi_id} (mốc {snapshot})", f"thiếu cột {err}")


def build_values(rows, definitions, min_n):
    """Biến danh sách dòng thành bảng kpi_values (thêm tên, đơn vị, cờ low_n)."""
    values = pd.DataFrame(rows).merge(definitions[["kpi_id", "name", "unit"]], on="kpi_id")
    values["value"] = values["value"].round(2)
    values["low_n"] = values["n"] < min_n  # nhóm quá ít mẫu: vẫn ghi nhưng đánh dấu
    return values[["kpi_id", "name", "snapshot", "level", "area", "value", "value_text", "n", "unit", "low_n"]]


def write_outputs(values, definitions):
    """Ghi kpi_values.csv và kpi_definitions.csv (chỉ gồm các KPI đã có giá trị)."""
    os.makedirs(PROCESSED, exist_ok=True)
    values.to_csv(f"{PROCESSED}/kpi_values.csv", index=False)
    definitions[definitions["kpi_id"].isin(values["kpi_id"])].to_csv(f"{PROCESSED}/kpi_definitions.csv", index=False)


# ============================================================ CHẠY CHÍNH
def main(config):
    snapshots = [str(x) for x in config["snapshots"]]  # YAML có thể biến ngày thành kiểu date
    currency = config.get("currency", "")
    kcfg = config.get("kpi", {})
    area_cols = kcfg.get("area_cols", ["analysis_area", "neighbourhood_cleansed"])
    min_n = kcfg.get("min_n", 30)
    host_threshold = kcfg.get("host_threshold", 5)
    commercial_nights = kcfg.get("commercial_nights", 183)
    occupancy_days = kcfg.get("occupancy_days")  # None = dùng cả calendar
    use_license = kcfg.get("use_license", False)

    definitions = kpi_definitions(currency, host_threshold, commercial_nights, occupancy_days)
    per_listing_kpis = [
        ("K1", f_median_price, {}),
        ("K2", f_occupancy, {}),
        ("K3", f_revpar, {}),
        ("K4", f_host_chuyen_nghiep, {"nguong": host_threshold}),
        ("K5", f_entire_home_thuong_mai, {"nguong": commercial_nights}),
    ]
    if use_license:
        per_listing_kpis.append(("K8", f_license, {}))

    rows, skipped = [], {}

    # ---- K1..K5, K8: tính cho từng mốc
    last_listings = None
    for moc in snapshots:
        listings = load_listings(moc)
        calendar = load_calendar(moc, occupancy_days)
        listings = add_occupancy(listings, calendar)
        for kpi_id, func, kwargs in per_listing_kpis:
            run_kpi(rows, skipped, kpi_id, moc, listings, area_cols, func, kwargs)
        last_listings = listings
        print(f"[kpi] xong mốc {moc}")

    # ---- K6, K7: dùng kết quả LLM (mẫu review không gắn với mốc nào nên snapshot = 'all')
    llm = load_llm_reviews(last_listings, area_cols)
    if llm is not None:
        run_kpi(rows, skipped, "K6", "all", llm, area_cols, f_negative)
        run_kpi(rows, skipped, "K7", "all", llm, area_cols, f_top_complaint)

    # ---- báo cáo KPI nào bị bỏ qua và vì sao
    for kpi_id, reason in skipped.items():
        print(f"[kpi] bỏ qua {kpi_id}: {reason}")
    if not rows:
        print("[kpi] chưa có KPI nào tính được, chưa ghi file")
        return

    values = build_values(rows, definitions, min_n)
    write_outputs(values, definitions)
    print(f"\n[kpi] đã ghi {len(values)} dòng vào {PROCESSED}/kpi_values.csv")
    city = values[values["level"] == "all"].pivot(index="kpi_id", columns="snapshot", values="value")
    print("Giá trị toàn thành phố:\n", city.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/madrid.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    main(config)