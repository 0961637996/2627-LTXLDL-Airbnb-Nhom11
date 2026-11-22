# 2627-LTXLDL-Airbnb-Nhom11

## 1. Cài đặt môi trường

1. **Tạo và kích hoạt môi trường ảo** *(khuyến nghị dùng đúng phiên bản Python trong `.python-version`)*:
   ```bash
   python -m venv .venv
   # Windows (PowerShell/CMD):
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình API Key** *(chỉ cần thiết nếu muốn chạy lại bước `src/llm.py`)*:
   Tạo file `.env` ở thư mục gốc và thêm khóa Gemini API:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## 2. Hướng dẫn chạy Pipeline

### Chạy nhanh toàn bộ quy trình (Khuyên dùng)
Khi đã có sẵn dữ liệu thô trong `data/raw/` và kết quả gán nhãn `data/processed/llm_results.csv`, chạy lệnh sau để tái lập toàn bộ kết quả QA, Baseline, KPI, Evaluate và Biểu đồ:
```bash
python run_pipeline.py --config configs/madrid.yaml --skip-download --skip-llm
```

### Chạy đầy đủ từ đầu (gồm tải dữ liệu & gọi lại API Gemini)
```bash
python run_pipeline.py --config configs/madrid.yaml
```

### Chạy lẻ từng bước (nếu cần)
```bash
python src/download.py --config configs/madrid.yaml
python src/qa.py --config configs/madrid.yaml
python src/baseline.py
python src/llm.py
python src/kpi.py --config configs/madrid.yaml
python src/evaluate.py
python src/viz.py --config configs/madrid.yaml
```

## 3. Đầu ra chính của hệ thống

* **`data/processed/kpi_values.csv` & `kpi_definitions.csv`**: Bảng giá trị và định nghĩa chi tiết của 8 chỉ số KPI (`K1`–`K8`).
* **`figures/`**: 5 biểu đồ độ phân giải 200 DPI phủ trọn 3 góc độ phân tích:
  1. `fig1_space_price_occ_revpar.png` (Không gian: K1, K2, K3)
  2. `fig2_space_commercial_license.png` (Không gian: K4, K5, K8)
  3. `fig3_time_kpi_trends.png` (Thời gian: K1–K5 qua 4 mốc)
  4. `fig4_llm_k6_k7.png` (LLM: K6, K7 theo 8 vùng)
  5. `fig5_space_map_k1_k6.png` (Bản đồ không gian cấp quận: K1 & K6)