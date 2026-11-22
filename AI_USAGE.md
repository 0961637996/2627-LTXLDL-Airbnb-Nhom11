## Phần đóng góp của Cấn Anh Sơn (llm, evaluate, hand_labeled_reviews, llm_results)

### 1. Các công cụ đã sử dụng và mục đích

- **Gemini:**
  - Em sử dụng Gemini để hỗ trợ viết, chỉnh sửa và gỡ lỗi mã nguồn trong `src/llm.py` và `src/evaluate.py`. Cụ thể, `src/llm.py` phụ trách gọi API với đầu ra JSON theo cấu trúc Pydantic, lưu kết quả định kỳ và tiếp tục chạy từ phần đã dừng (resume); còn `src/evaluate.py` dùng để tính các chỉ số Accuracy, Exact Match, Precision, Recall, F1 và Confusion Matrix.
  - Gemini cũng hỗ trợ em điều chỉnh prompt trích xuất đánh giá từ phiên bản v1 sang v2, đồng thời kiểm tra khả năng tương thích giữa `src/baseline.py` và `src/evaluate.py`.
  - Trong quá trình làm việc, Gemini còn hỗ trợ xử lý một số xung đột Git thông qua các lệnh `git stash`, `git pull`, `git push`, cũng như xây dựng và căn chỉnh bố cục báo cáo LaTeX trên Overleaf (`main.tex`, `content.tex`, `cover.tex`).

- **Gemini API (tích hợp trong `src/llm.py`):**
  - Em sử dụng model `emini-3.5-flash-lite` để xử lý tự động 1.000 đánh giá trong `data/samples/llm_reviews.csv`. Kết quả được lưu vào `data/processed/llm_results.csv`.

---

### 2. Các prompt và công việc chính đã thực hiện

1. **Thiết kế và tinh chỉnh prompt trích xuất ABSA (`src/llm.py`):**
   - Em thử nghiệm prompt trên 10 đánh giá mẫu, sau đó điều chỉnh từ phiên bản v1 sang v2. Trong quá trình này, em yêu cầu mô hình trả về JSON theo schema Pydantic (`ReviewAnalysis`) để đầu ra có cấu trúc thống nhất. Em cũng bổ sung quy tắc không suy diễn nhãn `host` hoặc `value` chỉ từ tên chủ nhà hay những lời cảm ơn chung chung.
   - **Tệp áp dụng:** `src/llm.py` (hàm `analyze_review`), dùng để tạo dữ liệu trong `data/processed/llm_results.csv`.

2. **Xây dựng bộ đánh giá để so sánh Baseline và LLM (`src/evaluate.py`):**
   - Em xây dựng hàm tách các nhãn đa nhãn (`to_set`), sau đó tính Exact Match và Micro Precision/Recall/F1 cho hai trường `aspects_positive` và `aspects_negative`. Ngoài ra, chương trình còn tính Sentiment Accuracy, Language Accuracy và xuất kết quả tổng hợp ra hai tệp `reports/eval_summary.csv`, `reports/eval_merged.csv`.
   - **Tệp áp dụng:** `src/evaluate.py`.

3. **Kiểm tra và sửa lỗi tương thích giữa Baseline và Evaluate (`src/baseline.py`, `src/evaluate.py`):**
   - Em nhờ AI rà soát mã nguồn `baseline.py` để thống nhất định dạng đầu ra với `evaluate.py`. Các chỉnh sửa gồm chuyển dữ liệu từ định dạng `json.dumps` sang chuỗi phân tách bằng dấu phẩy, điều chỉnh tên cột chưa thống nhất giữa `id` và `review_id`, đồng thời sửa một số lỗi cú pháp.
   - **Tệp áp dụng:** `src/baseline.py`, `src/evaluate.py`.

---

### 3. Những hạn chế và sai sót của AI (kèm ví dụ thực tế)

**A. Sai sót của Gemini khi hỗ trợ lập trình:**

- **Đề xuất thao tác chưa cần thiết do chưa kiểm tra kỹ mã nguồn:** Khi em tìm hiểu nguyên nhân `evaluate.py` chỉ chấm được 83/100 dòng, trong khi `llm_results.csv` có 199 dòng mang giá trị `"error"`, Gemini đề xuất xóa trắng các ô `"error"` rồi mới chạy lại. Tuy nhiên, sau khi kiểm tra `src/llm.py`, em nhận thấy điều kiện bỏ qua chỉ áp dụng với những dòng đã có sentiment hợp lệ thuộc `ALLOWED_SENTIMENTS = {"positive", "negative", "mixed"}`. Vì vậy, các dòng có giá trị `"error"` vốn vẫn được đưa vào xử lý lại, không cần xóa thủ công như đề xuất ban đầu.

**B. Sai sót của mô hình LLM trong pipeline (`src/llm.py`):**

- **Bỏ sót nhãn `amenities` do hiểu phạm vi khía cạnh này quá hẹp:** Trong 100 mẫu đánh giá, có 23/32 nhãn bị bỏ sót ở `aspects_positive` và 3/3 nhãn bị bỏ sót ở `aspects_negative` thuộc nhóm `amenities`. Chẳng hạn, với `review_id = 1225294395646169956` (*"Size of the place is huge"*) và `1011488363493274111` (*"no windows out onto the street"*), LLM không nhận diện được nhãn này vì chủ yếu hiểu `amenities` là các thiết bị cụ thể, chưa bao gồm những đặc điểm như diện tích hay kiến trúc phòng.

### 4. Cách nhóm kiểm chứng kết quả


- **Kiểm tra thủ công những mẫu có kết quả khác biệt:** Nhóm mở `reports/eval_merged.csv` để đọc lại nội dung đánh giá gốc tại các dòng mà nhãn của người gán và kết quả từ LLM không trùng khớp. Việc kiểm tra tập trung vào 3 trường hợp lệch Sentiment, 5 trường hợp lệch `aspects_negative` và những trường hợp phát sinh trong quá trình thử nghiệm prompt từ v1 sang v2. Qua đó, nhóm xem xét liệu sai lệch đến từ mô hình hay do nhãn thủ công còn thiếu sót.

- **Kiểm tra tính toàn vẹn của dữ liệu và chạy lại chương trình sau khi sửa lỗi:**
  - Nhóm xác nhận chương trình đã đối chiếu đủ 100/100 mẫu, khắc phục tình trạng trước đó chỉ chấm được 83/100 mẫu do các dòng có giá trị `"error"`.
  - Sau khi đổi tên cột, điều chỉnh thứ tự và sắp xếp dữ liệu, nhóm kiểm tra lại các tệp CSV để bảo đảm những chuỗi `review_id` dài không bị làm tròn hoặc thay đổi. Việc đọc và ghi dữ liệu được thực hiện với `dtype=str` và `encoding="utf-8-sig"`.

## Phần đóng góp của Nguyễn Đức Quý
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
## Phần đóng góp của Nguyễn Đình Thọ
### 1. Module `src/kpi.py` (Tính toán thông số từ dữ liệu đã làm sạch)
* **Công cụ AI sử dụng:** Claude
* **Nội dung hỗ trợ:**
  * Hỗ trợ viết khung sườn code , chỉ rõ các phần todo (bộ 8 kpi ,... )
  * Hỗ trợ làm phần todo và kiểm tra phần todo sau khi đã hoàn thành
* **Phần tự kiểm chứng và chỉnh sửa của nhóm:**
  * Tra cứu các thông tin về công thức tính bộ chỉ số kpi
  * Chỉnh sửa tên cho khớp nhãn của đầu vào để lấy dữ liệu từ file llm
  * Chạy kiểm thử file và tính tay trên bộ mẫu bé xem có khớp dữ liệu không

---

### 2. Module `src/viz.py` (Trực quan hóa dữ liệu kpi)
* **Công cụ AI sử dụng:** claude, gemini
* **Nội dung hỗ trợ:**
  * Giúp thu thập thông tin , các cách sử dụng biểu đồ phân tích dữ liệu thực tế từ những nguồn thực tế ,
  dựa trên những yêu cầu của mình để lập ra promt hoàn chỉnh cho claude
  * Hỗ trợ code biểu đồ
* **Phần tự kiểm chứng và chỉnh sửa của nhóm:**
  * Kiểm chứng dựa vào thị giác , thật sự dễ nhìn và dễ hiểu , chỉnh sửa code qua nhiều lần biểu đồ không như mong muốn

---

## Phần đóng góp của Hà Doãn Trọng 
### 1. Module src/qa.py (Lọc dữ liệu phân công & Kiểm soát chất lượng QA)
* *Công cụ AI sử dụng:* Gemini
* *Nội dung hỗ trợ:*
  * Gợi ý cấu trúc hàm log_qa ghi nhật ký quy tắc theo từng mốc (snapshot), bảng (table), điều kiện, lý do, hành động và biến động chỉ số (kpi_before / kpi_after).
  * Hỗ trợ viết logic đọc dữ liệu lớn theo từng lô (chunksize=100_000) cho calendar và reviews, làm sạch thẻ HTML, chuẩn hóa khoảng trắng và gắn cờ kiểm tra trùng/lệch giữa các mốc.
* *Phần tự kiểm chứng và chỉnh sửa của nhóm:*
  * Tự chỉnh đường dẫn đọc file phân công về configs/phan_cong_1000_listing.csv và configs`/phan_cong_5000_review.csv`, đồng bộ ngưỡng giá (min_price, max_price), danh mục vùng/loại phòng và mẫu bình luận tự động từ configs/madrid.yaml.
  * Tự điều chỉnh logic tách mẫu llm_reviews.csv và hand_labeled_reviews.csv ở mốc đầu tiên để giữ nguyên tệp nhãn tay nếu đã tồn tại, đồng thời kiểm tra đầu ra reports/qa_report.csv và các tệp trong data/processed/ sau khi chạy.
