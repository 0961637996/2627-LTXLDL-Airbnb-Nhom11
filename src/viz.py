"""viz.py - Sinh hình vào figures/ theo 3 góc độ của đề: KHÔNG GIAN, THỜI GIAN, LLM.

Đọc : data/processed/kpi_values.csv                            (do kpi.py tạo; cột kpi_id, snapshot, level, area, value, value_text, n, unit, low_n)
      data/processed/<mốc>_listings.csv, <mốc>_calendar.csv    (do qa.py tạo)
      data/processed/llm_results.csv                           (do llm.py tạo; chưa có thì bỏ qua các hình cần review gốc)
      data/raw/<mốc>_visualisations_neighbourhoods.geojson     (do download.py tải; chưa có thì bỏ qua bản đồ)
Ghi  : figures/*.png
Chạy : python src/viz.py --config configs/madrid.yaml

Danh sách hình (mỗi hình gắn với KPI nào và vì sao chọn dạng biểu đồ đó)
  KHÔNG GIAN
    space_choropleth_kpi        K1-K5 theo quận trên bản đồ tô màu: thấy ngay cụm khu vực giá cao / khai thác cao
    space_rank_price            K1 theo quận, dạng lollipop ngang: tên quận dài nên đọc ngang dễ hơn cột đứng
    space_scatter_price_occ     K1 (x) vs K2 (y), màu = K3: RevPAR là tích của giá và mức khai thác, scatter cho
                                thấy khu vực kiếm tiền nhờ giá cao hay nhờ kín phòng
    space_k1_grouped_bar        K1 theo 8 vùng, nhóm theo mốc, vạch nét đứt = toàn thành phố
    space_k4_stacked            K4 theo vùng, cột chồng 100% (host nhiều căn / host còn lại)
    space_k8_license            K8 (chỉ khi bật use_license) theo vùng, thanh tiến độ tới 100%
  THỜI GIAN
    time_k2_multiline           K2 đường đa tuyến: ALL đậm, 8 vùng mảnh
    time_kpi_trend              K1-K5 qua các mốc, mỗi KPI một ô
    time_area_heatmap           vùng x mốc cho K2, K4, K5 (heatmap, có số trong ô)
    time_k5_neigh_heatmap       K5 quận x mốc (chỉ quận đủ mẫu)
    time_slope_k4_k5            độ dốc K4, K5 giữa mốc đầu và mốc cuối
    time_calendar_curve         tỷ lệ đêm không trống theo ngày lịch, so sánh các mốc chụp
  LLM
    llm_k6_rank                 K6 xếp hạng vùng / quận, vạch tham chiếu = toàn thành phố
    llm_k7_top_complaint        K7 khía cạnh bị chê nhiều nhất ở từng vùng (màu = khía cạnh)
    llm_sentiment_by_area       cơ cấu positive / mixed / negative theo vùng
    llm_aspects                 khía cạnh được khen / bị chê nhiều nhất
    llm_aspect_by_area          khía cạnh bị chê x vùng
    llm_negative_map            bản đồ K6 theo quận
    llm_language                ngôn ngữ review và % tiêu cực (nếu có cột language)

Lưu ý khi đọc hình:
  - K6, K7 chỉ có một giá trị cho cả mẫu review (snapshot = 'all'), nên KHÔNG có trục thời gian; hình LLM so sánh theo khu vực.
  - K2 tính trên cửa sổ `kpi.occupancy_days` (để trống = cả 365 ngày calendar). Với cửa sổ 365 ngày, các mốc đều phủ đủ một
    năm nên chênh lệch giữa các mốc KHÔNG phải mùa vụ; muốn xem mùa vụ hãy xem time_calendar_curve.
  - Giá (K1, K3) ở các mốc trước có thể thiếu nhiều (đề bài): mốc có n < min_coverage x n lớn nhất bị bỏ khỏi hình thời gian.

Cấu hình (khối `viz:` trong yaml, thiếu thì dùng mặc định; min_n và area_col lấy từ khối `kpi:`):
  dpi (200) | top_n (12) | min_coverage (0.5) | k6_threshold (không đặt: chỉ vẽ vạch toàn thành phố)
"""
import argparse
import ast
import json
import os

import matplotlib

matplotlib.use("Agg")  # chạy được trên máy không có màn hình
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon

PROCESSED, RAW, FIG = "data/processed", "data/raw", "figures"
SENTIMENTS = ["positive", "mixed", "negative"]
SENT_COLOR = {"positive": "#2a9d8f", "mixed": "#e9c46a", "negative": "#e76f51"}
SENT_LABEL = {"positive": "Tích cực", "mixed": "Lẫn lộn", "negative": "Tiêu cực"}
SNAP_COLORS = ["#a9b4c2", "#6f8fb0", "#2f6590", "#0f2f4f"]  # nhạt -> đậm theo thời gian
BLUE, LIGHT, RED = "#2f6590", "#d7dde5", "#e76f51"
NOTE_LOW_N = "Gạch chéo / chấm rỗng / dấu *: n nhỏ hơn ngưỡng, chỉ để tham khảo"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})


# ============================================================ HÀM PHỤ
def save(fig, name, dpi, created):
    os.makedirs(FIG, exist_ok=True)
    path = f"{FIG}/{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    created.append(path)
    print(f"[viz] đã vẽ {path}")


def parse_aspects(cell):
    """Ô aspects_* của llm_results.csv -> list tên khía cạnh (JSON, list Python hoặc 'a;b'). Giống kpi.py."""
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
            items = text.split(";")
    return [str(a).strip().lower() for a in items if str(a).strip()]


def fmt(v, unit):
    return f"{v:.1f}%" if "%" in str(unit) else f"{v:,.0f}"


def meta(values, kpi_id):
    row = values[values["kpi_id"] == kpi_id].iloc[0]
    return row["name"], row["unit"]


def has(values, kpi_id, level="all"):
    return not values[(values["kpi_id"] == kpi_id) & (values["level"] == level)].dropna(subset=["value"]).empty


def kdf(values, kpi_id, level, snaps=None):
    """Các dòng KPI có giá trị, thêm cột `low` (bool) từ low_n."""
    d = values[(values["kpi_id"] == kpi_id) & (values["level"] == level)].dropna(subset=["value"]).copy()
    if snaps is not None:
        d = d[d["snapshot"].isin(snaps)]
    d["low"] = d["low_n"].astype(str).str.lower().eq("true")
    return d


def kpi_table(values, kpi_id, snapshot, level):
    """Bảng (index = khu vực) gồm value, n, low, value_text của một KPI ở một mốc."""
    d = kdf(values, kpi_id, level)
    return d[d["snapshot"] == snapshot].set_index("area")[["value", "n", "low", "value_text"]]


def city_value(values, kpi_id, snapshot):
    t = kpi_table(values, kpi_id, snapshot, "all")
    return None if t.empty else float(t["value"].iloc[0])


def area_sort_key(a):
    return (len(str(a)), str(a))


def valid_snapshots(values, kpi_id, snapshots, min_cov):
    """Mốc đủ dữ liệu cho KPI: n toàn thành phố >= min_cov x n lớn nhất qua các mốc."""
    city = values[(values["kpi_id"] == kpi_id) & (values["level"] == "all") & values["snapshot"].isin(snapshots)]
    n = city.set_index("snapshot")["n"].reindex(snapshots).fillna(0)
    ref = n.max()
    return [s for s in snapshots if ref > 0 and n[s] >= min_cov * ref]


def snap_color(snapshots, s):
    return SNAP_COLORS[snapshots.index(s) % len(SNAP_COLORS)]


def window_note(days):
    return f"K2 tính trên {days} ngày đầu sau mỗi mốc" if days else "K2 tính trên cả 365 ngày của calendar"


def spread(ys, gap):
    """Đẩy nhãn ra xa nhau (nhãn biểu đồ slope) để không chồng chữ. ys: dict tên -> y."""
    order = sorted(ys, key=ys.get)
    out, last = {}, -np.inf
    for k in order:
        out[k] = max(ys[k], last + gap)
        last = out[k]
    shift = (np.mean(list(out.values())) - np.mean(list(ys.values()))) if ys else 0
    return {k: v - shift for k, v in out.items()}


def heatmap(fig, ax, mat, title, unit, cmap="YlGnBu", low=None):
    """Heatmap có số trong từng ô; ô thiếu mẫu có dấu *."""
    im = ax.imshow(mat.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(c)[:7] for c in mat.columns], fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.grid(False)
    lo, hi = np.nanmin(mat.values), np.nanmax(mat.values)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                star = "*" if low is not None and bool(low.values[i, j]) else ""
                ax.text(j, i, f"{v:.1f}{star}", ha="center", va="center", fontsize=8,
                        color="white" if (v - lo) / (hi - lo + 1e-9) > 0.6 else "#111")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label(unit)


def load_geo(snapshots):
    """Đọc geojson ranh giới khu vực (mốc mới nhất có file). Trả về dict tên -> list vòng ngoài."""
    for moc in reversed(snapshots):
        path = f"{RAW}/{moc}_visualisations_neighbourhoods.geojson"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                geo = json.load(f)
            polys = {}
            for ft in geo["features"]:
                props, geom = ft["properties"], ft["geometry"]
                name = props.get("neighbourhood") or props.get("neighborhood")
                if geom["type"] == "Polygon":
                    parts = [geom["coordinates"]]
                elif geom["type"] == "MultiPolygon":
                    parts = geom["coordinates"]
                else:
                    continue
                polys[str(name)] = [np.array(p[0])[:, :2] for p in parts]  # chỉ vẽ vòng ngoài
            return polys
    return None


def draw_choropleth(fig, ax, polys, tab, title, unit, cmap="YlOrRd", label=True):
    """Tô màu từng quận theo KPI. Quận ít mẫu gạch chéo; quận không có dữ liệu màu xám."""
    ref = tab.loc[~tab["low"], "value"]
    ref = ref if len(ref) >= 3 else tab["value"]
    lo, hi = float(ref.min()), float(ref.max())
    norm = Normalize(lo, hi if hi > lo else lo + 1)
    cm = plt.get_cmap(cmap)
    xs, ys = [], []
    for name, rings in polys.items():
        for ring in rings:
            xs.extend(ring[:, 0])
            ys.extend(ring[:, 1])
            if name in tab.index:
                row = tab.loc[name]
                low = bool(row["low"])
                patch = Polygon(ring, closed=True, facecolor=cm(norm(row["value"])),
                                edgecolor="#555555" if low else "white", linewidth=0.5 if low else 0.7,
                                hatch="////" if low else None)
            else:
                patch = Polygon(ring, closed=True, facecolor="#e6e6e6", edgecolor="white", linewidth=0.7)
            ax.add_patch(patch)
        if label and name in tab.index:
            big = max(rings, key=len)
            ax.text(big[:, 0].mean(), big[:, 1].mean(), name[:13], ha="center", va="center", fontsize=5, color="#111",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ys))))
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(ScalarMappable(norm, cm), ax=ax, shrink=0.7, pad=0.01).set_label(unit)


# ============================================================ GÓC ĐỘ 1: KHÔNG GIAN
def fig_choropleth(values, snap, geo_level, polys, dpi, created):
    kpis = [k for k in ("K1", "K2", "K3", "K4", "K5") if not kpi_table(values, k, snap, geo_level).empty]
    if not kpis or polys is None:
        raise RuntimeError("thiếu geojson hoặc KPI theo quận")
    cmaps = {"K1": "YlOrRd", "K2": "YlGnBu", "K3": "PuRd", "K4": "Oranges", "K5": "Purples"}
    cols = 3 if len(kpis) > 4 else 2
    rows = (len(kpis) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 6.0 * rows), squeeze=False)
    for ax in axes.ravel()[len(kpis):]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), kpis):
        name, unit = meta(values, k)
        draw_choropleth(fig, ax, polys, kpi_table(values, k, snap, geo_level), f"{k} · {name}", unit, cmaps[k])
    fig.suptitle(f"Phân bố không gian của các KPI theo quận (mốc {snap})", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.0, f"{NOTE_LOW_N}. Xám: không có dữ liệu.", ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "space_choropleth_kpi", dpi, created)


def fig_rank_lollipop(values, snap, geo_level, top_n, dpi, created):
    t = kpi_table(values, "K1", snap, geo_level).sort_values("value", ascending=False)
    if t.empty:
        raise RuntimeError("không có K1 theo quận")
    name, unit = meta(values, "K1")
    gap = len(t) > 2 * top_n
    shown = pd.concat([t.head(top_n), t.tail(top_n)]) if gap else t
    shown = shown.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(shown) + 2))
    y = np.arange(len(shown))
    ax.hlines(y, 0, shown["value"], color=[LIGHT if l else BLUE for l in shown["low"]], lw=2)
    ax.scatter(shown["value"], y, color=[("#9aa5b1" if l else BLUE) for l in shown["low"]], s=55, zorder=3,
               edgecolor="#333", linewidth=0.5)
    for yi, (_, r) in zip(y, shown.iterrows()):
        ax.text(r["value"], yi, f"   {fmt(r['value'], unit)} (n={int(r['n'])})", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(shown.index, fontsize=8)
    cv = city_value(values, "K1", snap)
    if cv is not None:
        ax.axvline(cv, ls="--", c=RED, lw=1)
        ax.text(cv, len(shown) - 0.3, f" Toàn TP {fmt(cv, unit)}", color=RED, fontsize=8, va="bottom")
    ax.set_xlim(0, shown["value"].max() * 1.3)
    ax.set_xlabel(f"{name} ({unit})")
    ax.set_title(f"Giá niêm yết trung vị theo quận, mốc {snap}" + (f" (top {top_n} và bottom {top_n})" if gap else ""))
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.01, "Chấm xám: n nhỏ hơn ngưỡng. Giá niêm yết, không phải giá khách thực trả.", fontsize=8, color="#555")
    save(fig, "space_rank_price", dpi, created)


def fig_scatter_price_occ(values, snap, geo_level, dpi, created):
    panels = []
    for lv, label in (("analysis_area", "Vùng phân tích"), (geo_level, "Quận")):
        p, o, r = (kpi_table(values, k, snap, lv) for k in ("K1", "K2", "K3"))
        idx = p.index.intersection(o.index)
        if len(idx) >= 3:
            panels.append((label, idx, p, o, r.reindex(idx)["value"]))
    if not panels:
        raise RuntimeError("quá ít khu vực có đủ K1 và K2")
    pn, pu = meta(values, "K1")
    on, _ = meta(values, "K2")
    fig, axes = plt.subplots(1, len(panels), figsize=(8.5 * len(panels), 6.8), squeeze=False)
    for ax, (label, idx, p, o, r) in zip(axes.ravel(), panels):
        size = (30 + 14 * np.sqrt(p.loc[idx, "n"].astype(float))).values
        low = (p.loc[idx, "low"] | o.loc[idx, "low"]).values
        use_r = r.notna().any()
        sc = ax.scatter(p.loc[idx, "value"], o.loc[idx, "value"], s=size, c=r.values if use_r else BLUE,
                        cmap="viridis", alpha=0.85, edgecolor="#333", linewidth=0.6)
        if low.any():  # chấm n nhỏ: viền nét đứt
            ax.scatter(p.loc[idx[low], "value"], o.loc[idx[low], "value"], s=size[low] + 25, facecolor="none",
                       edgecolor="#888", linewidth=1, linestyle="--")
        for x, y, nm, lw in zip(p.loc[idx, "value"], o.loc[idx, "value"], idx, low):
            ax.annotate(nm, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7,
                        style="italic" if lw else "normal", color="#777" if lw else "#111")
        ax.axvline(p.loc[idx, "value"].median(), ls="--", c="#999", lw=0.8)
        ax.axhline(o.loc[idx, "value"].median(), ls="--", c="#999", lw=0.8)
        ax.set_xlabel(f"{pn} ({pu})")
        ax.set_ylabel(f"{on} (%)")
        ax.set_title(f"{label}, mốc {snap}")
        if use_r:
            fig.colorbar(sc, ax=ax, shrink=0.8).set_label("K3 · RevPAR ước tính")
    fig.suptitle("Giá hay độ kín phòng tạo ra RevPAR?", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, "Kích thước chấm ~ số listing (n); viền nét đứt / chữ nghiêng: n nhỏ hơn ngưỡng. Nét đứt: trung vị giữa các "
             "khu vực. K3 là trung vị của (giá x tỷ lệ đêm không trống) theo listing nên chỉ xấp xỉ K1 x K2.", fontsize=8, color="#555")
    save(fig, "space_scatter_price_occ", dpi, created)


def fig_k1_grouped(values, snapshots, min_cov, dpi, created):
    ok = valid_snapshots(values, "K1", snapshots, min_cov)
    d = kdf(values, "K1", "analysis_area", ok)
    if d.empty:
        raise RuntimeError("không có K1 theo vùng")
    name, unit = meta(values, "K1")
    areas = sorted(d["area"].unique(), key=area_sort_key)
    w = 0.8 / len(ok)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, s in enumerate(ok):
        ds = d[d["snapshot"] == s].set_index("area").reindex(areas)
        x = np.arange(len(areas)) + (i - (len(ok) - 1) / 2) * w
        ax.bar(x, ds["value"], w, color=snap_color(snapshots, s), label=s, edgecolor="white")
        for xi, (_, r) in zip(x, ds.iterrows()):
            if pd.notna(r["value"]):
                ax.text(xi, r["value"], f"{r['value']:.0f}" + ("*" if r["low"] else ""), ha="center", va="bottom", fontsize=7)
        cv = city_value(values, "K1", s)
        if cv is not None:
            ax.axhline(cv, color=snap_color(snapshots, s), ls="--", lw=1)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas)
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_title("Giá niêm yết trung vị theo vùng và mốc chụp (nét đứt = toàn thành phố của mốc đó)")
    ax.legend(title="Mốc", frameon=False, ncol=len(ok))
    ax.grid(axis="x", visible=False)
    drop = [s for s in snapshots if s not in ok]
    fig.text(0.01, -0.01, "* n nhỏ hơn ngưỡng." + (f" Mốc bị bỏ vì thiếu giá: {drop}." if drop else ""), fontsize=8, color="#555")
    save(fig, "space_k1_grouped_bar", dpi, created)


def fig_k4_stacked(values, snap, dpi, created):
    t = kpi_table(values, "K4", snap, "analysis_area").sort_values("value", ascending=False)
    if t.empty:
        raise RuntimeError("không có K4 theo vùng")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(t))
    ax.bar(x, t["value"], color="#0f2f4f", label="Host nhiều căn (chuyên nghiệp)")
    ax.bar(x, 100 - t["value"], bottom=t["value"], color=LIGHT, label="Host ít căn / cá nhân")
    for xi, (_, r) in zip(x, t.iterrows()):
        ax.text(xi, r["value"] / 2, f"{r['value']:.0f}%" + ("*" if r["low"] else ""), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(xi, 101, f"n={int(r['n'])}", ha="center", fontsize=7, color="#555")
    cv = city_value(values, "K4", snap)
    if cv is not None:
        ax.axhline(cv, ls="--", c=RED, lw=1.2)
        ax.text(len(t) - 0.5, cv + 1, f"Toàn TP {cv:.0f}%", color=RED, ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(t.index)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% listing")
    ax.set_title(f"Cơ cấu host theo vùng (mốc {snap}); vùng sắp theo mức thương mại hoá giảm dần")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "space_k4_stacked", dpi, created)


def fig_k8(values, snap, dpi, created):
    t = kpi_table(values, "K8", snap, "analysis_area").sort_values("value")
    if t.empty:
        raise RuntimeError("không có K8 (bật kpi.use_license trong yaml rồi chạy lại kpi.py)")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 2))
    y = np.arange(len(t))
    ax.barh(y, 100, color="#eceff3", height=0.65)  # đường đua tới 100%
    ax.barh(y, t["value"], color=BLUE, height=0.4)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(max(r["value"], 1) + 1, yi, f"{r['value']:.0f}%" + ("*" if r["low"] else "") + f"  (n={int(r['n'])})",
                va="center", fontsize=8)
    cv = city_value(values, "K8", snap)
    if cv is not None:
        ax.axvline(cv, ls="--", c=RED, lw=1.2)
        ax.text(cv, len(t) - 0.4, f" Toàn TP {cv:.0f}%", color=RED, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% listing có số đăng ký (vạch đích = 100%)")
    ax.set_title(f"Mức tuân thủ đăng ký theo vùng, mốc {snap}")
    ax.grid(axis="y", visible=False)
    save(fig, "space_k8_license", dpi, created)


# ============================================================ GÓC ĐỘ 2: THỜI GIAN
def line_panel(ax, values, k, ok, snapshots, areas, cmap, unit):
    for i, a in enumerate(areas):
        d = values[(values["kpi_id"] == k) & (values["level"] == "analysis_area") & (values["area"] == a)
                   & values["snapshot"].isin(ok)].sort_values("snapshot")
        ax.plot(pd.to_datetime(d["snapshot"]), d["value"], color=cmap(i % 10), lw=1.1, alpha=0.6, marker="o", ms=3)
    c = values[(values["kpi_id"] == k) & (values["level"] == "all") & values["snapshot"].isin(ok)].sort_values("snapshot")
    x = pd.to_datetime(c["snapshot"])
    ax.plot(x, c["value"], color="black", lw=2.6, marker="o", ms=6, zorder=5)
    for xi, v in zip(x, c["value"]):
        ax.annotate(fmt(v, unit), (xi, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(pd.to_datetime(snapshots))
    ax.set_xticklabels([s[:7] for s in snapshots], fontsize=8)
    ax.set_xlim(pd.to_datetime(snapshots[0]) - pd.Timedelta(days=25), pd.to_datetime(snapshots[-1]) + pd.Timedelta(days=25))


def fig_k2_multiline(values, snapshots, min_cov, window, dpi, created):
    if not has(values, "K2"):
        raise RuntimeError("không có K2")
    name, unit = meta(values, "K2")
    areas = sorted(values.loc[values["level"] == "analysis_area", "area"].unique(), key=area_sort_key)
    cmap = plt.get_cmap("tab10")
    ok = valid_snapshots(values, "K2", snapshots, min_cov)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    line_panel(ax, values, "K2", ok, snapshots, areas, cmap, unit)
    ax.plot([], [], color="black", lw=2.6, marker="o", label="Toàn thành phố")
    for i, a in enumerate(areas):
        ax.plot([], [], color=cmap(i % 10), lw=1.2, label=a)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="best")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_title("Tỷ lệ đêm không còn trống qua các mốc chụp")
    fig.text(0.01, -0.01, f"{window}. Là cận trên của mức khai thác (gồm cả đêm host tự chặn lịch).", fontsize=8, color="#555")
    save(fig, "time_k2_multiline", dpi, created)


def fig_trend(values, snapshots, min_cov, window, dpi, created):
    kpis = [k for k in ("K1", "K2", "K3", "K4", "K5") if has(values, k)]
    if not kpis:
        raise RuntimeError("không có KPI nào theo mốc")
    areas = sorted(values.loc[values["level"] == "analysis_area", "area"].unique(), key=area_sort_key)
    cmap = plt.get_cmap("tab10")
    n_col = 3
    n_row = (len(kpis) + 1 + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(14, 3.9 * n_row), squeeze=False)
    flat = axes.ravel()
    dropped = []
    for ax, k in zip(flat, kpis):
        name, unit = meta(values, k)
        ok = valid_snapshots(values, k, snapshots, min_cov)
        if len(ok) < len(snapshots):
            dropped.append(f"{k}: {len(ok)}/{len(snapshots)} mốc")
        line_panel(ax, values, k, ok, snapshots, areas, cmap, unit)
        ax.set_title(f"{k} · {name} ({unit})")
    leg = flat[len(kpis)]
    leg.axis("off")
    leg.plot([], [], color="black", lw=2.6, marker="o", label="Toàn thành phố")
    for i, a in enumerate(areas):
        leg.plot([], [], color=cmap(i % 10), lw=1.2, label=a)
    leg.legend(loc="center", ncol=2, frameon=False, title="Vùng phân tích")
    for ax in flat[len(kpis) + 1:]:
        ax.axis("off")
    fig.suptitle("KPI qua các mốc chụp", fontsize=14, fontweight="bold", y=1.0)
    note = f"{window}."
    if dropped:
        note += " Mốc thiếu dữ liệu (n < %d%% mốc đầy đủ nhất) bị bỏ: %s." % (min_cov * 100, "; ".join(dropped))
    fig.text(0.01, -0.01, note, fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "time_kpi_trend", dpi, created)


def fig_time_heatmap(values, snapshots, window, dpi, created):
    kpis = [k for k in ("K2", "K4", "K5") if has(values, k, "analysis_area")]
    if not kpis:
        raise RuntimeError("không có KPI theo vùng")
    fig, axes = plt.subplots(1, len(kpis), figsize=(5.4 * len(kpis), 4.8), squeeze=False)
    cmaps = {"K2": "YlGnBu", "K4": "Oranges", "K5": "Purples"}
    for ax, k in zip(axes.ravel(), kpis):
        name, unit = meta(values, k)
        d = kdf(values, k, "analysis_area", snapshots)
        cols = [s for s in snapshots if s in set(d["snapshot"])]
        mat = d.pivot(index="area", columns="snapshot", values="value")[cols]
        low = d.pivot(index="area", columns="snapshot", values="low")[cols].fillna(False)
        order = sorted(mat.index, key=area_sort_key)
        heatmap(fig, ax, mat.loc[order], f"{k} · {name}", unit, cmaps[k], low.loc[order])
    fig.suptitle("Vùng phân tích x mốc chụp", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, f"{window}. {NOTE_LOW_N}.", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "time_area_heatmap", dpi, created)


def fig_k5_neigh_heatmap(values, snapshots, geo_level, top_n, dpi, created):
    d = kdf(values, "K5", geo_level, snapshots)
    d = d[~d["low"]]
    cols = [s for s in snapshots if s in set(d["snapshot"])]
    if d.empty or not cols:
        raise RuntimeError("không có quận đủ mẫu cho K5")
    mat = d.pivot(index="area", columns="snapshot", values="value")[cols].dropna(how="all")
    mat = mat.sort_values(cols[-1], ascending=False).head(2 * top_n)
    name, unit = meta(values, "K5")
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.34 * len(mat) + 2))
    heatmap(fig, ax, mat, f"K5 · {name}: quận x mốc (chỉ quận đủ mẫu)", unit, "Purples")
    save(fig, "time_k5_neigh_heatmap", dpi, created)


def fig_slope(values, snapshots, min_cov, dpi, created):
    kp = [k for k in ("K4", "K5") if has(values, k, "analysis_area")]
    if not kp:
        raise RuntimeError("không có K4/K5 theo vùng")
    fig, axes = plt.subplots(1, len(kp), figsize=(6.5 * len(kp), 6.2), squeeze=False)
    for ax, k in zip(axes.ravel(), kp):
        ok = valid_snapshots(values, k, snapshots, min_cov)
        if len(ok) < 2:
            ax.axis("off")
            continue
        a, b = ok[0], ok[-1]
        d = kdf(values, k, "analysis_area", [a, b])
        pv = d.pivot(index="area", columns="snapshot", values="value")[[a, b]].dropna()
        name, unit = meta(values, k)
        gap = (pv.values.max() - pv.values.min()) / 22 + 1e-6
        left, right = spread(pv[a].to_dict(), gap), spread(pv[b].to_dict(), gap)
        for area, row in pv.iterrows():
            ax.plot([0, 1], [row[a], row[b]], marker="o", color=RED if row[b] > row[a] else BLUE, lw=1.8)
            ax.text(-0.04, left[area], f"{area}  {row[a]:.1f}", ha="right", va="center", fontsize=8)
            ax.text(1.04, right[area], f"{row[b]:.1f}  {area}", ha="left", va="center", fontsize=8)
        ca, cb = city_value(values, k, a), city_value(values, k, b)
        if ca is not None and cb is not None:
            ax.plot([0, 1], [ca, cb], color="black", lw=2.8, marker="o", zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([a, b])
        ax.set_xlim(-0.45, 1.45)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_title(f"{k} · {name} ({unit})")
    fig.suptitle("Thay đổi giữa mốc đầu và mốc cuối (đỏ: tăng, xanh: giảm, đen: toàn thành phố)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "time_slope_k4_k5", dpi, created)


def fig_calendar_curve(snapshots, dpi, created):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    drawn = 0
    for moc in snapshots:
        path = f"{PROCESSED}/{moc}_calendar.csv"
        if not os.path.exists(path):
            continue
        cal = pd.read_csv(path, usecols=["date", "available"]).dropna()
        if cal.empty:
            continue
        cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
        daily = (1 - cal.groupby("date")["available"].mean()) * 100
        color = snap_color(snapshots, moc)
        ax.plot(daily.index, daily.rolling(7, min_periods=3, center=True).mean(), color=color, lw=2, label=f"Mốc {moc}")
        ax.axvline(pd.Timestamp(moc), color=color, ls=":", lw=1)
        drawn += 1
    if not drawn:
        raise RuntimeError("không có calendar đã xử lý")
    ax.set_ylabel("% listing không còn trống trong đêm đó (TB trượt 7 ngày)")
    ax.set_xlabel("Ngày lưu trú trên lịch")
    ax.set_title("Đường cong đặt chỗ: cùng một đêm, nhìn từ các thời điểm chụp khác nhau")
    ax.legend(title="Chụp lúc", frameon=False)
    fig.text(0.01, -0.01, "Nét chấm đứng: ngày chụp. Gồm cả đêm host tự chặn lịch nên là cận trên của mức khai thác. "
             "Đây là hình để đọc mùa vụ: mọi mốc cùng được xem trên một trục ngày.", fontsize=8, color="#555")
    save(fig, "time_calendar_curve", dpi, created)


# ============================================================ GÓC ĐỘ 3: LLM
def fig_k6_rank(values, geo_level, top_n, threshold, dpi, created):
    levels = [(lv, lb) for lv, lb in (("analysis_area", "Vùng phân tích"), (geo_level, "Quận (top rủi ro)"))
              if not kpi_table(values, "K6", "all", lv).empty]
    if not levels:
        raise RuntimeError("không có K6 (chạy llm.py rồi kpi.py)")
    name, unit = meta(values, "K6")
    cv = city_value(values, "K6", "all")
    fig, axes = plt.subplots(1, len(levels), figsize=(7.5 * len(levels), 5.8), squeeze=False)
    for ax, (lv, lb) in zip(axes.ravel(), levels):
        t = kpi_table(values, "K6", "all", lv).sort_values("value", ascending=False).head(top_n).iloc[::-1]
        bars = ax.barh(t.index, t["value"], color=[LIGHT if l else RED for l in t["low"]], edgecolor="#bbb")
        for b, (_, r) in zip(bars, t.iterrows()):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {r['value']:.1f}%  (n={int(r['n'])})", va="center", fontsize=8)
        if cv is not None:
            ax.axvline(cv, ls="--", c="black", lw=1)
            ax.text(cv, len(t) - 0.4, f" Toàn TP {cv:.1f}%", fontsize=8)
        if threshold is not None:
            ax.axvline(threshold, ls="--", c="#d62828", lw=1.2)
            ax.text(threshold, -0.9, f" Ngưỡng cảnh báo {threshold:g}%", color="#d62828", fontsize=8)
        ax.set_xlim(0, max(t["value"].max(), threshold or 0, cv or 0) * 1.3)
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_title(lb)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Tỷ lệ review tiêu cực (LLM) theo khu vực", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, "Mẫu review không gắn với mốc nào nên không có trục thời gian. Cột xám: n nhỏ hơn ngưỡng.", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "llm_k6_rank", dpi, created)


def fig_k7(values, dpi, created):
    t = kpi_table(values, "K7", "all", "analysis_area")
    t = t[t["value_text"].astype(str).str.len() > 0].sort_values("value", ascending=False)
    if t.empty:
        raise RuntimeError("không có K7 (chạy llm.py rồi kpi.py)")
    aspects = sorted(t["value_text"].unique())
    cmap = plt.get_cmap("tab10")
    col = {a: cmap(i % 10) for i, a in enumerate(aspects)}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(t))
    ax.bar(x, t["value"], color=[col[a] for a in t["value_text"]], edgecolor="white")
    for xi, (_, r) in zip(x, t.iterrows()):
        ax.text(xi, r["value"], f"{r['value_text']}\n{r['value']:.0f}%" + ("*" if r["low"] else ""), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\nn={int(n)}" for a, n in zip(t.index, t["n"])], fontsize=8)
    ax.set_ylim(0, t["value"].max() * 1.3)
    ax.set_ylabel("% review nhắc tới khía cạnh đứng đầu")
    cs = kpi_table(values, "K7", "all", "all")
    sub = f" (toàn TP: {cs['value_text'].iloc[0]}, {cs['value'].iloc[0]:.0f}%)" if not cs.empty else ""
    ax.set_title(f"Khía cạnh bị chê nhiều nhất theo vùng{sub}")
    ax.legend(handles=[Patch(color=col[a], label=a) for a in aspects], frameon=False, title="Khía cạnh", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.01, "Cột = % review của vùng nhắc tới khía cạnh đứng đầu; * = n nhỏ hơn ngưỡng.", fontsize=8, color="#555")
    save(fig, "llm_k7_top_complaint", dpi, created)


def load_llm(snapshots, area_col):
    path = f"{PROCESSED}/llm_results.csv"
    if not os.path.exists(path):
        raise RuntimeError("chưa có llm_results.csv (chạy llm.py trước)")
    llm = pd.read_csv(path, dtype={"review_id": str, "listing_id": str})
    llm["sentiment"] = llm["sentiment"].astype(str).str.strip().str.lower()
    llm = llm[llm["sentiment"].isin(SENTIMENTS)].copy()
    want = {"id", "analysis_area", area_col}
    lst = pd.read_csv(f"{PROCESSED}/{snapshots[-1]}_listings.csv", usecols=lambda c: c in want,
                      dtype={"id": str}).drop_duplicates("id").set_index("id")
    return llm.drop(columns=[c for c in lst.columns if c in llm.columns]).join(lst, on="listing_id")


def fig_sentiment_area(llm, min_n, dpi, created):
    d = llm.dropna(subset=["analysis_area"])
    t = pd.crosstab(d["analysis_area"], d["sentiment"]).reindex(columns=SENTIMENTS, fill_value=0)
    n = t.sum(axis=1)
    pct = (t.div(n, axis=0) * 100).sort_values("negative")
    pct.loc["Toàn thành phố"] = llm["sentiment"].value_counts(normalize=True).reindex(SENTIMENTS).fillna(0) * 100
    n.loc["Toàn thành phố"] = len(llm)
    pct = pct.loc[["Toàn thành phố"] + [a for a in pct.index if a != "Toàn thành phố"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(pct) + 2))
    left = np.zeros(len(pct))
    for s in SENTIMENTS:
        bars = ax.barh(pct.index, pct[s], left=left, color=SENT_COLOR[s], label=SENT_LABEL[s], edgecolor="white")
        for b, v, l in zip(bars, pct[s], left):
            if v >= 6:
                ax.text(l + v / 2, b.get_y() + b.get_height() / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        left += pct[s].values
    for i, a in enumerate(pct.index):
        ax.text(101, i, f"n={int(n[a])}" + ("  (ít mẫu)" if n[a] < min_n else ""), va="center", fontsize=8, color="#555")
    ax.set_xlim(0, 118)
    ax.set_xlabel("% review")
    ax.set_title("Cảm xúc của khách (LLM) theo vùng; vùng sắp theo % tiêu cực")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.12), frameon=False)
    save(fig, "llm_sentiment_by_area", dpi, created)


def count_aspects(series):
    c = {}
    for cell in series:
        for a in set(parse_aspects(cell)):
            c[a] = c.get(a, 0) + 1
    return pd.Series(c, dtype=float)


def fig_aspects(llm, top_n, dpi, created):
    neg = count_aspects(llm["aspects_negative"]) / len(llm) * 100 if "aspects_negative" in llm else pd.Series(dtype=float)
    pos = count_aspects(llm["aspects_positive"]) / len(llm) * 100 if "aspects_positive" in llm else pd.Series(dtype=float)
    if neg.empty and pos.empty:
        raise RuntimeError("llm_results.csv không có khía cạnh nào")
    top = neg.add(pos, fill_value=0).sort_values(ascending=False).head(top_n).index[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(top) + 2))
    if not pos.empty:
        ax.barh(top, pos.reindex(top).fillna(0), color=SENT_COLOR["positive"], label="Được khen")
    if not neg.empty:
        ax.barh(top, -neg.reindex(top).fillna(0), color=SENT_COLOR["negative"], label="Bị chê")
    for i, a in enumerate(top):
        if not pos.empty and pos.get(a, 0) > 0:
            ax.text(pos[a], i, f" {pos[a]:.1f}%", va="center", fontsize=8)
        if not neg.empty and neg.get(a, 0) > 0:
            ax.text(-neg[a], i, f"{neg[a]:.1f}% ", va="center", ha="right", fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.0f}" for t in ticks])
    ax.set_xlabel(f"% review nhắc tới khía cạnh (mẫu {len(llm)} review)")
    ax.set_title("Khía cạnh được khen và bị chê nhiều nhất (LLM)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "llm_aspects", dpi, created)


def fig_aspect_area(llm, top_n, dpi, created):
    if "aspects_negative" not in llm:
        raise RuntimeError("không có aspects_negative")
    d = llm.dropna(subset=["analysis_area"])
    overall = count_aspects(d["aspects_negative"]).sort_values(ascending=False).head(min(top_n, 8))
    if overall.empty:
        raise RuntimeError("không có khía cạnh bị chê")
    areas = sorted(d["analysis_area"].unique(), key=area_sort_key)
    mat = pd.DataFrame(index=overall.index, columns=areas, dtype=float)
    ns = {}
    for a in areas:
        sub = d[d["analysis_area"] == a]
        ns[a] = len(sub)
        c = count_aspects(sub["aspects_negative"])
        mat[a] = [c.get(x, 0) / len(sub) * 100 for x in overall.index]
    labels = [f"{a}\nn={ns[a]}" for a in areas]
    fig, ax = plt.subplots(figsize=(1.1 * len(areas) + 3, 0.6 * len(mat) + 2.5))
    heatmap(fig, ax, mat, "Khía cạnh bị chê theo vùng (LLM)", "% review trong vùng nhắc tới", "Reds")
    ax.set_xticklabels(labels, fontsize=8)
    save(fig, "llm_aspect_by_area", dpi, created)


def fig_negative_map(values, geo_level, polys, dpi, created):
    t = kpi_table(values, "K6", "all", geo_level)
    if t.empty or polys is None:
        raise RuntimeError("thiếu K6 theo quận hoặc geojson")
    name, unit = meta(values, "K6")
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_choropleth(fig, ax, polys, t, f"K6 · {name}", unit, cmap="Reds")
    fig.text(0.5, 0.0, f"{NOTE_LOW_N}. Mẫu LLM chia cho nhiều quận nên n thường nhỏ.", ha="center", fontsize=8, color="#555")
    save(fig, "llm_negative_map", dpi, created)


def fig_language(llm, top_n, dpi, created):
    if "language" not in llm.columns:
        raise RuntimeError("llm_results.csv không có cột language")
    d = llm.dropna(subset=["language"]).copy()
    d["language"] = d["language"].astype(str).str.lower()
    top = d["language"].value_counts().head(top_n)
    neg = d[d["language"].isin(top.index)].groupby("language")["sentiment"].apply(lambda s: s.eq("negative").mean() * 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(top) + 2.5), sharey=True)
    order = top.index[::-1]
    axes[0].barh(order, top.reindex(order), color=BLUE)
    axes[0].set_title("Số review theo ngôn ngữ")
    axes[1].barh(order, neg.reindex(order), color=SENT_COLOR["negative"])
    axes[1].set_title("% review tiêu cực theo ngôn ngữ")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    for i, l in enumerate(order):
        axes[0].text(top[l], i, f" {top[l]}", va="center", fontsize=8)
        axes[1].text(neg[l], i, f" {neg[l]:.0f}%", va="center", fontsize=8)
    save(fig, "llm_language", dpi, created)


# ============================================================ CHẠY CHÍNH
def main(config):
    kcfg, vcfg = config.get("kpi") or {}, config.get("viz") or {}
    dpi, top_n, min_cov = vcfg.get("dpi", 200), vcfg.get("top_n", 12), vcfg.get("min_coverage", 0.5)
    threshold = vcfg.get("k6_threshold")
    min_n = kcfg.get("min_n", 30)
    geo_level = kcfg.get("area_col", "neighbourhood_cleansed")
    window = window_note(kcfg.get("occupancy_days"))

    path = f"{PROCESSED}/kpi_values.csv"
    if not os.path.exists(path):
        raise SystemExit("[viz] chưa có kpi_values.csv, hãy chạy kpi.py trước")
    values = pd.read_csv(path, dtype={"snapshot": str, "area": str})

    cfg_snaps = [str(x) for x in config["snapshots"]]
    present = set(values.loc[values["snapshot"] != "all", "snapshot"])
    snapshots = [s for s in cfg_snaps if s in present]
    missing = [s for s in cfg_snaps if s not in present]
    if missing:
        print(f"[viz] CẢNH BÁO: kpi_values.csv không có mốc {missing}; hình thời gian chỉ vẽ {snapshots}. "
              f"Kiểm tra lại qa.py và kpi.py cho mốc đó.")
    if not snapshots:
        raise SystemExit("[viz] kpi_values.csv không có mốc nào trùng với config")
    snap = snapshots[-1]  # hình không gian dùng mốc cuối: giá ở mốc cuối đáng tin nhất (đề bài)

    polys = load_geo(cfg_snaps)
    if polys is None:
        print("[viz] chưa có geojson trong data/raw, bỏ qua các bản đồ (chạy download.py)")

    created = []
    steps = [
        # không gian
        ("space_choropleth_kpi", lambda: fig_choropleth(values, snap, geo_level, polys, dpi, created)),
        ("space_rank_price", lambda: fig_rank_lollipop(values, snap, geo_level, top_n, dpi, created)),
        ("space_scatter_price_occ", lambda: fig_scatter_price_occ(values, snap, geo_level, dpi, created)),
        ("space_k1_grouped_bar", lambda: fig_k1_grouped(values, snapshots, min_cov, dpi, created)),
        ("space_k4_stacked", lambda: fig_k4_stacked(values, snap, dpi, created)),
        ("space_k8_license", lambda: fig_k8(values, snap, dpi, created)),
        # thời gian
        ("time_k2_multiline", lambda: fig_k2_multiline(values, snapshots, min_cov, window, dpi, created)),
        ("time_kpi_trend", lambda: fig_trend(values, snapshots, min_cov, window, dpi, created)),
        ("time_area_heatmap", lambda: fig_time_heatmap(values, snapshots, window, dpi, created)),
        ("time_k5_neigh_heatmap", lambda: fig_k5_neigh_heatmap(values, snapshots, geo_level, top_n, dpi, created)),
        ("time_slope_k4_k5", lambda: fig_slope(values, snapshots, min_cov, dpi, created)),
        ("time_calendar_curve", lambda: fig_calendar_curve(cfg_snaps, dpi, created)),
        # LLM: K6, K7 đọc từ kpi_values.csv
        ("llm_k6_rank", lambda: fig_k6_rank(values, geo_level, top_n, threshold, dpi, created)),
        ("llm_k7_top_complaint", lambda: fig_k7(values, dpi, created)),
        ("llm_negative_map", lambda: fig_negative_map(values, geo_level, polys, dpi, created)),
    ]
    try:  # các hình cần review gốc đã gán nhãn
        llm = load_llm(snapshots, geo_level)
        steps += [
            ("llm_sentiment_by_area", lambda: fig_sentiment_area(llm, min_n, dpi, created)),
            ("llm_aspects", lambda: fig_aspects(llm, top_n, dpi, created)),
            ("llm_aspect_by_area", lambda: fig_aspect_area(llm, top_n, dpi, created)),
            ("llm_language", lambda: fig_language(llm, top_n, dpi, created)),
        ]
    except Exception as err:  # noqa: BLE001 - thiếu llm_results không được làm dừng các hình khác
        print(f"[viz] bỏ qua các hình LLM cần review gốc: {err}")

    for name, fn in steps:
        try:
            fn()
        except Exception as err:  # noqa: BLE001 - một hình lỗi không làm hỏng cả bộ
            plt.close("all")
            print(f"[viz] bỏ qua {name}: {err}")
    print(f"\n[viz] đã tạo {len(created)} hình trong {FIG}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/madrid.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        main(yaml.safe_load(f))