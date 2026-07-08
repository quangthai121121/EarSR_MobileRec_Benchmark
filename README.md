# EarSR-MobileRec Benchmark — All Lightweight SR + Mobile Recognition

## Reproducibility / Seed 42

Project đã cố định seed mặc định là **42** trong config:

```yaml
project:
  seed: 42
  deterministic: true
```

Các bước đã được set seed:

- Downsample dataset: `--seed 42` trong `scripts/run_00_downsample_10p.sh`.
- Recognition training: Python `random`, NumPy, PyTorch CPU/GPU, DataLoader shuffle và DataLoader workers đều dùng seed 42.
- External SR command: `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, và biến `{seed}` có thể dùng trong `command_template` nếu repo SR hỗ trợ tham số seed.

Ví dụ nếu repo SR có argument seed, bạn có thể sửa command như sau:

```yaml
command_template: "python {repo_dir}/test.py --scale {scale} --model_path {checkpoint} --input {input_dir} --output {output_dir} --seed {seed}"
```

Lưu ý: seed 42 giúp kết quả lặp lại tốt trên cùng máy/cùng phiên bản thư viện. Một số phép toán CUDA hoặc khác phiên bản GPU/driver vẫn có thể tạo sai khác rất nhỏ. Nếu cần reproducibility nghiêm ngặt nhất, có thể đặt `num_workers: 0` trong config, đổi `amp: false`, và chạy lại từ đầu sau khi xóa folder kết quả cũ.


Project này được thiết kế cho đúng mục tiêu thí nghiệm:

```text
Dataset ảnh tai đã split
        ↓
Downsample, ví dụ còn 10% kích thước gốc
        ↓
Chạy nhiều lightweight SR models trên ảnh downsample
        ↓
Đưa ảnh downsample và ảnh downsample+SR vào cùng mobile recognition models
        ↓
Xuất CSV so sánh Accuracy / Precision / Recall / F1-score
```

Mục tiêu chính là kiểm tra:

```text
Ảnh downsample + SR có accuracy cao hơn ảnh downsample hay không?
```

Không chọn SR model theo PSNR/SSIM trước. Với hướng này, model SR tốt là model làm **recognition accuracy / macro-F1 tăng**.

---

## 1. Models có trong project

### Lightweight SR models

Project đã khai báo đầy đủ các SR model bạn yêu cầu trong:

```text
configs/benchmark_10p_all_sr_mobile_rec.yaml
```

Danh sách SR pipeline:

```text
lr_10p              = ảnh downsample 10%, baseline chính
bicubic_10p_x4      = optional baseline, không phải learned SR
span_10p_x4         = SPAN
safmn_10p_x4        = SAFMN
rfdn_10p_x4         = RFDN
efdn_10p_x4         = EFDN
asid_10p_x4         = ASID
catanet_10p_x4      = CATANet
lkfn_10p_x4         = LKFN/LKDN family
seemore_10p_x4      = SeemoRe
```

Nếu bật toàn bộ learned SR thì sẽ có:

```text
1 tập ảnh downsample baseline
8 tập ảnh downsample + SR
= 9 pipeline ảnh đưa vào recognition
```

Nếu chỉ muốn đúng 6 tập như ví dụ trước, bật 5 SR model bất kỳ và giữ `lr_10p`.

### Mobile / edge recognition models

Project đã khai báo các recognition backbone:

```text
MobileNetV4
RepViT
MobileOne
FastViT
EfficientFormerV2
GhostNetV2
MobileViTv2
```

Các model recognition được gọi qua `timm`. Script sẽ tự chọn tên model đầu tiên tồn tại trong phiên bản `timm` đang cài.

---

## 2. Cấu trúc dataset đầu vào

Bạn chuẩn bị dataset đã split sẵn dạng ImageFolder:

```text
data/EarVN1.0_split/
  train/
    subject_001/*.jpg
    subject_002/*.jpg
  val/
    subject_001/*.jpg
    subject_002/*.jpg
  test/
    subject_001/*.jpg
    subject_002/*.jpg
```

Mỗi folder subject là một class recognition.

---

## 3. Cài đặt môi trường

```bash
cd EarSR_MobileRec_Benchmark
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra các model recognition có trong `timm`:

```bash
bash scripts/run_00_check_timm.sh
```

Nếu model nào không xuất hiện, mở file config và sửa candidates:

```text
configs/benchmark_10p_all_sr_mobile_rec.yaml
```

---

## 4. Bước 1 — Downsample dataset 10%

Chạy:

```bash
bash scripts/run_00_downsample_10p.sh
```

Script này tạo:

```text
outputs/lr/earvn_10p/
  train/<subject>/*.png
  val/<subject>/*.png
  test/<subject>/*.png
```

Đây là tập **ảnh downsample 10%**. Nó là baseline chính trong so sánh.

Đồng thời script tạo report:

```text
outputs/reports/downsample_10p_report.csv
```

Report này có các cột:

```text
relative_path, lr_path, hr_path, orig_w, orig_h, hr_w, hr_h, lr_w, lr_h, percent, degradation
```

### Nếu muốn downsample tỉ lệ khác

Ví dụ 20%:

```bash
python -m src.downsample_dataset \
  --input_root data/EarVN1.0_split \
  --output_lr_root outputs/lr/earvn_20p \
  --output_hr_root outputs/hr_mod/earvn_20p \
  --percent 20 \
  --interpolation bicubic \
  --output_format png \
  --degradation bicubic \
  --report_csv outputs/reports/downsample_20p_report.csv
```

Lưu ý: nếu dùng SR ×4, ảnh 10% sau SR ×4 sẽ thành khoảng 40% kích thước gốc. Recognition vẫn resize mọi ảnh về 224×224 nên phép so sánh giữa các pipeline vẫn công bằng. Nếu muốn SR ×4 khôi phục gần kích thước gốc, dùng downsample 25% hoặc `--scale 4`.

---

## 5. Bước 2 — Clone repo SR

Chạy:

```bash
bash scripts/clone_sr_repos.sh
```

Script sẽ clone vào `external/`:

```text
external/SPAN
external/SAFMN
external/RFDN
external/EFDN
external/ASID
external/CATANet
external/LKDN
external/seemoredetails
```

Sau đó bạn cần tải checkpoint theo README của từng repo.

Quan trọng: mỗi repo SR có script inference khác nhau. Vì vậy project dùng cơ chế adapter bằng `command_template` trong config:

```text
configs/benchmark_10p_all_sr_mobile_rec.yaml
```

Ví dụ:

```yaml
- name: span_10p_x4
  type: external
  enabled: false
  repo_dir: external/SPAN
  checkpoint: external/SPAN/checkpoints/span_x4.pth
  command_template: "python {repo_dir}/main_test.py --scale {scale} --model_path {checkpoint} --input {input_dir} --output {output_dir}"
```

Bạn cần sửa `checkpoint` và `command_template` cho đúng repo thực tế. Các biến có thể dùng:

```text
{scale}       scale SR, mặc định 4
{checkpoint}  đường dẫn checkpoint
{input_dir}   input SR, ở đây là outputs/lr/earvn_10p
{output_dir}  output SR, ví dụ outputs/sr_10p/span_10p_x4
{repo_dir}    folder repo external
```

Mặc định các external SR để `enabled: false` để tránh lỗi khi bạn chưa tải checkpoint. Muốn chạy model nào thì đổi:

```yaml
enabled: true
```

---

## 6. Bước 3 — Chạy SR cho tất cả model đã bật

```bash
bash scripts/run_02_sr_10p_all_enabled.sh
```

Output SR sẽ nằm ở:

```text
outputs/sr_10p/span_10p_x4/
outputs/sr_10p/safmn_10p_x4/
outputs/sr_10p/rfdn_10p_x4/
outputs/sr_10p/efdn_10p_x4/
outputs/sr_10p/asid_10p_x4/
outputs/sr_10p/catanet_10p_x4/
outputs/sr_10p/lkfn_10p_x4/
outputs/sr_10p/seemore_10p_x4/
```

Mỗi output phải giữ cấu trúc:

```text
train/<subject>/*.png
val/<subject>/*.png
test/<subject>/*.png
```

Nếu repo SR xuất ảnh ra cấu trúc khác, cần sửa command hoặc copy lại output để giữ đúng cấu trúc ImageFolder.

---

## 7. Bước 4 — Kiểm tra số ảnh trong các pipeline

Sau khi chạy SR, nên kiểm tra các pipeline có đủ train/val/test chưa:

```bash
bash scripts/run_05_verify_pipelines.sh
```

Script sẽ tạo:

```text
outputs/results_10p_all_sr_mobile_rec/pipeline_image_counts.csv
```

Nếu một SR repo xuất thiếu split hoặc thiếu ảnh, cần sửa output trước khi chạy recognition.

---

## 8. Bước 5 — Chạy recognition benchmark

Chạy toàn bộ pipeline ảnh và toàn bộ recognition models:

```bash
bash scripts/run_03_recognition_10p_all.sh
```

Lệnh bên trong:

```bash
python -m src.run_benchmark --config configs/benchmark_10p_all_sr_mobile_rec.yaml
```

Nó sẽ train/test matched pipeline:

```text
Train recognition trên lr_10p train → test trên lr_10p test
Train recognition trên span_10p_x4 train → test trên span_10p_x4 test
Train recognition trên safmn_10p_x4 train → test trên safmn_10p_x4 test
...
```

Mỗi kết quả riêng nằm ở:

```text
outputs/results_10p_all_sr_mobile_rec/<pipeline>/<recognition_model>/metrics.csv
outputs/results_10p_all_sr_mobile_rec/<pipeline>/<recognition_model>/history.csv
outputs/results_10p_all_sr_mobile_rec/<pipeline>/<recognition_model>/confusion_matrix.csv
```

---

## 9. Bước 6 — Tổng hợp CSV so sánh với ảnh downsample

Chạy:

```bash
bash scripts/run_04_summary_vs_lr10p.sh
```

Output chính:

```text
outputs/results_10p_all_sr_mobile_rec/final_comparison_vs_lr_10p.csv
```

File này là bảng bạn cần nhất. Nó có dạng:

```text
pipeline,accuracy,precision_macro,recall_macro,f1_macro,delta_accuracy,delta_precision_macro,delta_recall_macro,delta_f1_macro,better_than_baseline_acc,better_than_baseline_f1
lr_10p,...
span_10p_x4,...
safmn_10p_x4,...
rfdn_10p_x4,...
efdn_10p_x4,...
asid_10p_x4,...
catanet_10p_x4,...
lkfn_10p_x4,...
seemore_10p_x4,...
```

Trong đó:

```text
delta_accuracy = accuracy của pipeline - accuracy của lr_10p
delta_f1_macro = f1_macro của pipeline - f1_macro của lr_10p
```

Nếu `better_than_baseline_acc = True`, nghĩa là ảnh sau SR cho accuracy cao hơn ảnh downsample.

Ngoài ra còn có:

```text
outputs/results_10p_all_sr_mobile_rec/summary_all.csv
outputs/results_10p_all_sr_mobile_rec/summary_delta_vs_lr_10p.csv
outputs/results_10p_all_sr_mobile_rec/summary_avg_by_pipeline_vs_lr_10p.csv
```

---

## 10. Chạy full workflow

Sau khi đã sửa command/checkpoint và bật các SR model muốn chạy:

```bash
bash scripts/run_10_full_10p_all_sr_workflow.sh
```

Workflow này chạy:

```text
1. Downsample 10%
2. SR bằng các method enabled=true
3. Train/test recognition
4. Xuất CSV tổng hợp so với lr_10p
```

---

## 11. Chạy nhanh một vài model để test trước

Chỉ chạy 1 SR method:

```bash
python -m src.run_sr --config configs/benchmark_10p_all_sr_mobile_rec.yaml --method span_10p_x4
```

Chỉ benchmark một vài recognition families:

```bash
python -m src.run_benchmark \
  --config configs/benchmark_10p_all_sr_mobile_rec.yaml \
  --families MobileNetV4 RepViT \
  --epochs 5
```

Chỉ benchmark một vài pipeline:

```bash
python -m src.run_benchmark \
  --config configs/benchmark_10p_all_sr_mobile_rec.yaml \
  --pipelines lr_10p span_10p_x4 safmn_10p_x4 \
  --epochs 5
```

---

## 12. Cách đọc kết quả

Bảng cuối cùng cần nhìn:

```text
final_comparison_vs_lr_10p.csv
```

Tiêu chí chọn SR model tốt:

```text
1. delta_accuracy > 0
2. delta_f1_macro > 0
3. Tăng ở đa số recognition models trong summary_delta_vs_lr_10p.csv
4. Nếu có đo thêm tốc độ: params/FLOPs/latency/FPS vẫn phù hợp mobile/real-time
```

Câu kết luận khi có kết quả:

```text
Among the evaluated lightweight super-resolution methods, <MODEL> achieved the highest average recognition accuracy and macro-F1, improving over the downsampled baseline by <X> and <Y>, respectively.
```

Tiếng Việt:

```text
Trong các phương pháp lightweight SR được đánh giá, <MODEL> cho accuracy và macro-F1 trung bình cao nhất, tăng lần lượt <X> và <Y> so với ảnh downsample baseline.
```
