import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/madrid.yaml", help="File cấu hình YAML")
parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước tải dữ liệu")
parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua bước gọi API Gemini")
args = parser.parse_args()

py = sys.executable

steps = [
    [py, "src/download.py", "--config", args.config],
    [py, "src/qa.py", "--config", args.config],
    [py, "src/baseline.py"],
    [py, "src/llm.py"],
    [py, "src/kpi.py", "--config", args.config],
    [py, "src/evaluate.py"],
    [py, "src/viz.py", "--config", args.config],
]

for cmd in steps:
    if args.skip_download and "src/download.py" in cmd:
        print("\n>>> Bỏ qua: src/download.py")
        continue
    if args.skip_llm and "src/llm.py" in cmd:
        print("\n>>> Bỏ qua: src/llm.py")
        continue
    print(f"\n>>> Đang chạy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print("\n>>> Đã chạy xong toàn bộ pipeline!")