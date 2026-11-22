"""viz.py - Vẽ 5 biểu đồ cốt lõi phủ trọn 8 KPI (K1-K8) và 3 góc độ: Không gian, Thời gian, LLM.

Đọc : data/processed/kpi_values.csv
      data/raw/<mốc>_visualisations_neighbourhoods.geojson (tùy chọn, cho bản đồ)
Ghi : figures/fig1..fig5.png
Chạy: python src/viz.py --config configs/madrid.yaml
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon

PROCESSED, RAW, FIG = "data/processed", "data/raw", "figures"
BLUE, RED, GRAY = "#2f6590", "#e76f51", "#cfd6df"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})


# ============================================================ HÀM PHỤ
def save_fig(fig, name, dpi):
    os.makedirs(FIG, exist_ok=True)
    path = f"{FIG}/{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[viz] đã vẽ {path}")


def kpi_table(values, kpi_id, snapshot, level):
    d = values[
        (values["kpi_id"] == kpi_id)
        & (values["snapshot"] == snapshot)
        & (values["level"] == level)
    ].dropna(subset=["value"]).copy()
    d["low"] = d["low_n"].astype(str).str.lower().eq("true")
    return d.set_index("area")[["value", "n", "low", "value_text"]]


def city_value(values, kpi_id, snapshot):
    t = kpi_table(values, kpi_id, snapshot, "all")
    return None if t.empty else float(t["value"].iloc[0])


def valid_snapshots(values, kpi_id, snapshots, min_cov=0.5):
    city = values[
        (values["kpi_id"] == kpi_id)
        & (values["level"] == "all")
        & values["snapshot"].isin(snapshots)
    ]
    n = city.set_index("snapshot")["n"].reindex(snapshots).fillna(0)
    ref = n.max()
    return [s for s in snapshots if ref > 0 and n[s] >= min_cov * ref]


def load_geo(snapshots):
    for moc in reversed(snapshots):
        path = f"{RAW}/{moc}_visualisations_neighbourhoods.geojson"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                geo = json.load(f)
            polys = {}
            for ft in geo["features"]:
                name = ft["properties"].get("neighbourhood") or ft["properties"].get("neighborhood")
                geom = ft["geometry"]
                if geom["type"] == "Polygon":
                    parts = [geom["coordinates"]]
                elif geom["type"] == "MultiPolygon":
                    parts = geom["coordinates"]
                else:
                    continue
                polys[str(name)] = [np.array(p[0])[:, :2] for p in parts]
            return polys
    return None


def draw_choropleth(fig, ax, polys, tab, title, unit, cmap="YlOrRd"):
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
                patch = Polygon(
                    ring, closed=True, facecolor=cm(norm(row["value"])),
                    edgecolor="#555" if low else "white", linewidth=0.6,
                    hatch="////" if low else None,
                )
            else:
                patch = Polygon(ring, closed=True, facecolor="#e6e6e6", edgecolor="white", linewidth=0.6)
            ax.add_patch(patch)
        if name in tab.index:
            big = max(rings, key=len)
            ax.text(
                big[:, 0].mean(), big[:, 1].mean(), name[:12],
                ha="center", va="center", fontsize=5.5, color="#111",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
            )
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ys))))
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(ScalarMappable(norm, cm), ax=ax, shrink=0.72, pad=0.01).set_label(unit)


# ============================================================ 5 BIỂU ĐỒ CHÍNH
def fig1_space_price_occ_revpar(values, snap, dpi):
    """Hình 1 (Không gian - K1, K2, K3): Giá vs Độ kín phòng, màu = RevPAR."""
    p = kpi_table(values, "K1", snap, "analysis_area")
    o = kpi_table(values, "K2", snap, "analysis_area")
    r = kpi_table(values, "K3", snap, "analysis_area")
    idx = p.index.intersection(o.index)
    if idx.empty:
        raise RuntimeError("thiếu K1/K2 theo analysis_area")

    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = 60 + 25 * np.sqrt(p.loc[idx, "n"].astype(float))
    sc = ax.scatter(
        p.loc[idx, "value"], o.loc[idx, "value"],
        s=sizes, c=r.reindex(idx)["value"], cmap="viridis",
        alpha=0.9, edgecolor="#222", linewidth=0.8, zorder=3,
    )
    for area in idx:
        x, y = p.loc[area, "value"], o.loc[area, "value"]
        rv = r.loc[area, "value"] if area in r.index else np.nan
        lbl = f"{area}\n({rv:.0f}€)" if pd.notna(rv) else area
        ax.annotate(lbl, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8.5, fontweight="bold")

    cv_p, cv_o = city_value(values, "K1", snap), city_value(values, "K2", snap)
    if cv_p is not None:
        ax.axvline(cv_p, ls="--", c=RED, lw=1.1, label=f"Giá toàn TP: {cv_p:.0f} EUR")
    if cv_o is not None:
        ax.axhline(cv_o, ls=":", c=BLUE, lw=1.1, label=f"Độ kín toàn TP: {cv_o:.1f}%")

    fig.colorbar(sc, ax=ax, shrink=0.85).set_label("K3 · RevPAR ước tính (EUR/đêm)")
    ax.set_xlabel("K1 · Giá niêm yết trung vị (EUR/đêm)")
    ax.set_ylabel("K2 · Tỷ lệ đêm không còn trống - 90 ngày (%)")
    ax.set_title(f"Phân vị giá, độ kín phòng và RevPAR theo 8 vùng (mốc {snap})")
    ax.legend(frameon=False, loc="best")
    save_fig(fig, "fig1_space_price_occ_revpar", dpi)


def fig2_space_commercial_license(values, snap, dpi):
    """Hình 2 (Không gian - K4, K5, K8): Mức độ thương mại hóa & tuân thủ đăng ký."""
    kpis = [k for k in ("K4", "K5", "K8") if not kpi_table(values, k, snap, "analysis_area").empty]
    if not kpis:
        raise RuntimeError("thiếu K4, K5, K8")

    titles = {
        "K4": "K4 · Host nhiều căn (>=5)",
        "K5": "K5 · Entire home thương mại (>=183 đêm)",
        "K8": "K8 · Có số đăng ký (License)",
    }
    colors = {"K4": "#0f2f4f", "K5": "#5e60ce", "K8": "#2a9d8f"}

    fig, axes = plt.subplots(1, len(kpis), figsize=(5.2 * len(kpis), 5), sharey=True)
    if len(kpis) == 1:
        axes = [axes]

    areas = sorted(kpi_table(values, kpis[0], snap, "analysis_area").index)
    for ax, k in zip(axes, kpis):
        t = kpi_table(values, k, snap, "analysis_area").reindex(areas)
        bars = ax.barh(areas, t["value"], color=colors[k], edgecolor="white")
        for b, (_, row) in zip(bars, t.iterrows()):
            if pd.notna(row["value"]):
                star = "*" if row["low"] else ""
                ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                        f"{row['value']:.1f}%{star}", va="center", fontsize=8.5)
        cv = city_value(values, k, snap)
        if cv is not None:
            ax.axvline(cv, ls="--", c=RED, lw=1.2)
            ax.text(cv, len(areas) - 0.3, f" TP: {cv:.1f}%", color=RED, fontsize=8, va="bottom")
        ax.set_xlim(0, 112)
        ax.set_xlabel("% listing")
        ax.set_title(titles[k])
        ax.grid(axis="y", visible=False)

    fig.suptitle(f"Mức độ thương mại hóa và tuân thủ đăng ký theo vùng (mốc {snap})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig2_space_commercial_license", dpi)


def fig3_time_kpi_trends(values, snapshots, min_cov, dpi):
    """Hình 3 (Thời gian - K1..K5): Xu hướng 5 KPI qua các mốc chụp."""
    kpis = [k for k in ("K1", "K2", "K3", "K4", "K5") if not values[values["kpi_id"] == k].empty]
    areas = sorted(values.loc[values["level"] == "analysis_area", "area"].dropna().unique())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat = axes.ravel()
    dropped = []

    for ax, k in zip(flat, kpis):
        row_meta = values[values["kpi_id"] == k].iloc[0]
        ok = valid_snapshots(values, k, snapshots, min_cov)
        if len(ok) < len(snapshots):
            dropped.append(f"{k}: {len(ok)}/{len(snapshots)} mốc")

        for i, a in enumerate(areas):
            d = values[
                (values["kpi_id"] == k) & (values["level"] == "analysis_area")
                & (values["area"] == a) & values["snapshot"].isin(ok)
            ].sort_values("snapshot")
            ax.plot(pd.to_datetime(d["snapshot"]), d["value"], color=cmap(i % 10), lw=1.1, alpha=0.55, marker="o", ms=3)

        c = values[
            (values["kpi_id"] == k) & (values["level"] == "all") & values["snapshot"].isin(ok)
        ].sort_values("snapshot")
        x = pd.to_datetime(c["snapshot"])
        ax.plot(x, c["value"], color="black", lw=2.5, marker="o", ms=6, zorder=5)
        for xi, v in zip(x, c["value"]):
            lbl = f"{v:.1f}%" if "%" in str(row_meta["unit"]) else f"{v:.0f}"
            ax.annotate(lbl, (xi, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(pd.to_datetime(snapshots))
        ax.set_xticklabels([s[:7] for s in snapshots], fontsize=8)
        ax.set_title(f"{k} · {row_meta['name']} ({row_meta['unit']})")

    # Ô thứ 6 làm chú giải
    leg = flat[5]
    leg.axis("off")
    leg.plot([], [], color="black", lw=2.5, marker="o", label="Toàn thành phố")
    for i, a in enumerate(areas):
        leg.plot([], [], color=cmap(i % 10), lw=1.4, label=a)
    leg.legend(loc="center", ncol=2, frameon=False, title="Khu vực", fontsize=9.5)

    fig.suptitle("Diễn biến các chỉ số KPI qua 4 mốc thời gian", fontsize=14, fontweight="bold")
    if dropped:
        fig.text(0.01, -0.01, f"Lưu ý: Bỏ mốc thiếu dữ liệu (n < {int(min_cov*100)}% mốc đầy đủ nhất): {', '.join(dropped)}.",
                 fontsize=8.5, color="#555")
    fig.tight_layout()
    save_fig(fig, "fig3_time_kpi_trends", dpi)


def fig4_llm_k6_k7(values, dpi):
    """Hình 4 (LLM - K6, K7): Tỷ lệ review tiêu cực và khía cạnh bị chê nhiều nhất theo vùng."""
    t6 = kpi_table(values, "K6", "all", "analysis_area").sort_values("value", ascending=True)
    t7 = kpi_table(values, "K7", "all", "analysis_area")
    t7 = t7[t7["value_text"].astype(str).str.len() > 0].sort_values("value", ascending=False)
    if t6.empty and t7.empty:
        raise RuntimeError("chưa có K6, K7 trong kpi_values.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Trái: K6
    bars = ax1.barh(t6.index, t6["value"], color=[GRAY if l else RED for l in t6["low"]], edgecolor="#999")
    for b, (_, r) in zip(bars, t6.iterrows()):
        ax1.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2,
                 f"{r['value']:.1f}% (n={int(r['n'])})", va="center", fontsize=8.5)
    cv6 = city_value(values, "K6", "all")
    if cv6 is not None:
        ax1.axvline(cv6, ls="--", c="black", lw=1.1)
        ax1.text(cv6, len(t6) - 0.3, f" Toàn TP: {cv6:.1f}%", fontsize=8.5, va="bottom")
    ax1.set_xlim(0, max(t6["value"].max(), cv6 or 0) * 1.32)
    ax1.set_xlabel("% review tiêu cực")
    ax1.set_title("K6 · Tỷ lệ review tiêu cực theo vùng (LLM)")
    ax1.grid(axis="y", visible=False)

    # Phải: K7
    aspects = sorted(t7["value_text"].unique())
    cmap = plt.get_cmap("tab10")
    col = {a: cmap(i % 10) for i, a in enumerate(aspects)}
    x = np.arange(len(t7))
    ax2.bar(x, t7["value"], color=[col[a] for a in t7["value_text"]], edgecolor="white")
    for xi, (_, r) in zip(x, t7.iterrows()):
        star = "*" if r["low"] else ""
        ax2.text(xi, r["value"] + 0.5, f"{r['value_text']}\n{r['value']:.1f}%{star}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{a}\n(n={int(n)})" for a, n in zip(t7.index, t7["n"])], fontsize=8.5)
    ax2.set_ylim(0, t7["value"].max() * 1.32)
    ax2.set_ylabel("% review nhắc tới khía cạnh đứng đầu")
    ax2.set_title("K7 · Khía cạnh bị chê nhiều nhất theo vùng (LLM)")
    ax2.legend(handles=[Patch(color=col[a], label=a) for a in aspects], frameon=False, title="Khía cạnh")
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Phân tích đánh giá khách hàng bằng LLM (mẫu 1.000 review)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig4_llm_k6_k7", dpi)


def fig5_space_map_k1_k6(values, snap, geo_level, polys, dpi):
    """Hình 5 (Bản đồ không gian - K1 & K6 theo quận)."""
    if polys is None:
        raise RuntimeError("chưa có file geojson trong data/raw/")
    t1 = kpi_table(values, "K1", snap, geo_level)
    t6 = kpi_table(values, "K6", "all", geo_level)
    if t1.empty and t6.empty:
        raise RuntimeError("thiếu dữ liệu theo quận")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    if not t1.empty:
        draw_choropleth(fig, ax1, polys, t1, f"K1 · Giá niêm yết trung vị (mốc {snap})", "EUR/đêm", "YlOrRd")
    else:
        ax1.axis("off")
    if not t6.empty:
        draw_choropleth(fig, ax2, polys, t6, "K6 · Tỷ lệ review tiêu cực (LLM)", "%", "Reds")
    else:
        ax2.axis("off")

    fig.suptitle("Bản đồ phân bố giá và tỷ lệ phàn nàn theo quận", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.0, "Gạch chéo: n nhỏ hơn ngưỡng; Xám: không có dữ liệu.", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    save_fig(fig, "fig5_space_map_k1_k6", dpi)


# ============================================================ CHẠY CHÍNH
def main(config):
    kcfg, vcfg = config.get("kpi") or {}, config.get("viz") or {}
    dpi = vcfg.get("dpi", 200)
    min_cov = vcfg.get("min_coverage", 0.5)
    geo_level = kcfg.get("area_col", "neighbourhood_cleansed")

    path = f"{PROCESSED}/kpi_values.csv"
    if not os.path.exists(path):
        raise SystemExit("[viz] chưa có kpi_values.csv, hãy chạy kpi.py trước")
    values = pd.read_csv(path, dtype={"snapshot": str, "area": str})

    cfg_snaps = [str(x) for x in config["snapshots"]]
    present = set(values.loc[values["snapshot"] != "all", "snapshot"])
    snapshots = [s for s in cfg_snaps if s in present]
    if not snapshots:
        raise SystemExit("[viz] kpi_values.csv không có mốc nào trùng với config")
    snap = snapshots[-1]

    polys = load_geo(cfg_snaps)

    steps = [
        ("fig1_space_price_occ_revpar", lambda: fig1_space_price_occ_revpar(values, snap, dpi)),
        ("fig2_space_commercial_license", lambda: fig2_space_commercial_license(values, snap, dpi)),
        ("fig3_time_kpi_trends", lambda: fig3_time_kpi_trends(values, snapshots, min_cov, dpi)),
        ("fig4_llm_k6_k7", lambda: fig4_llm_k6_k7(values, dpi)),
        ("fig5_space_map_k1_k6", lambda: fig5_space_map_k1_k6(values, snap, geo_level, polys, dpi)),
    ]

    for name, fn in steps:
        try:
            fn()
        except Exception as err:
            plt.close("all")
            print(f"[viz] bỏ qua {name}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/madrid.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        main(yaml.safe_load(f))