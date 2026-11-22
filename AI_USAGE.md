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
