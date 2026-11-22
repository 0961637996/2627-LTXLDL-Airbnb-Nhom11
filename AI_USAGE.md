### 1. Module `src/download.py` (Thu thập dữ liệu)
* **Công cụ AI sử dụng:** Gemini.
* **Nội dung hỗ trợ:**
  * Gợi ý cách tách tham số cấu hình ra khỏi mã nguồn bằng `argparse` và `PyYAML` (đọc `data_root` và `snapshots` từ `configs/madrid.yaml`).
  * Hỗ trợ xây dựng cơ chế kiểm tra tệp tồn tại (`os.path.exists`) để bỏ qua các tệp đã tải, tránh tải trùng lặp.
* **Phần tự kiểm chứng và chỉnh sửa của nhóm:**
  * Tự đối chiếu cấu trúc URL thực tế của Inside Airbnb cho dữ liệu Madrid (`visualisations/neighbourhoods.geojson` và 3 tệp `.csv.gz`).
  * Chạy kiểm tra thực tế để xác nhận tải đủ 16 tệp vào thư mục `data/raw/` và bổ sung tùy chọn `--skip-download` ở `run_pipeline.py` nhằm đảm bảo khả năng chạy lại trên dữ liệu cục bộ.

---

### 2. Module `src/baseline.py` (Phân tích cảm xúc cơ sở bằng từ điển & luật)
* **Công cụ AI sử dụng:** Gemini.
* **Nội dung hỗ trợ:**
  * Hỗ trợ tổng hợp và mở rộng bộ từ điển song ngữ Anh – Tây Ban Nha (`ASPECT_KEYWORDS`, `POS_WORDS`, `NEG_WORDS`) bám sát 6 khía cạnh của bài toán (`location`, `host`, `amenities`, `cleanliness`, `value`, `noise`).
  * Gợi ý biểu thức chính quy (Regex) để tách câu theo liên từ đối lập (*but, however, pero, aunque...*) và nhận diện cấu trúc phủ định (`NEGATION_PATTERNS`).
* **Phần tự kiểm chứng và chỉnh sửa của nhóm:**
  * Tự điều chỉnh logic xử lý đặc thù cho khía cạnh `noise` (tách riêng `NOISE_POS`, `NOISE_NEG` và cụm phủ định như *no noise / sin ruido*) và quy tắc ưu tiên nhãn tiêu cực (`asp_pos -= asp_neg`).
  * Tự bổ sung cơ chế tự động nhận diện cột định danh (`id` hoặc `review_id`) để đồng bộ với đầu ra của `qa.py`.
  * Chạy kiểm thử trên toàn bộ tập `data/samples/llm_reviews.csv`, rà soát phân bố nhãn `base_sentiment`, `base_language` và đối chiếu kết quả trên tập 100 nhãn tay ở bước `evaluate.py`.

  ---

### 6. Module `src/kpi.py` (Tính toán thông số từ dữ liệu đã làm sạch)
* **Công cụ AI sử dụng:** Claude
* **Nội dung hỗ trợ:**
  * Hỗ trợ viết khung sườn code , chỉ rõ các phần todo (bộ 8 kpi ,... )
  * Hỗ trợ làm phần todo và kiểm tra phần todo sau khi đã hoàn thành
* **Phần tự kiểm chứng và chỉnh sửa của nhóm:**
  * Tra cứu các thông tin về công thức tính bộ chỉ số kpi
  * Chỉnh sửa tên cho khớp nhãn của đầu vào để lấy dữ liệu từ file llm
  * Chạy kiểm thử file và tính tay trên bộ mẫu bé xem có khớp dữ liệu không

---

### 7. Module `src/viz.py` (Trực quan hóa dữ liệu kpi)
* **Công cụ AI sử dụng:** claude, gemini
* **Nội dung hỗ trợ:**
  * Giúp thu thập thông tin , các cách sử dụng biểu đồ phân tích dữ liệu thực tế từ những nguồn thực tế ,
  dựa trên những yêu cầu của mình để lập ra promt hoàn chỉnh cho claude
  * Hỗ trợ code biểu đồ
* **Phần tự kiểm chứng và chỉnh sửa của nhóm:**
  * Kiểm chứng dựa vào thị giác , thật sự dễ nhìn và dễ hiểu , chỉnh sửa code qua nhiều lần biểu đồ không như mong muốn

