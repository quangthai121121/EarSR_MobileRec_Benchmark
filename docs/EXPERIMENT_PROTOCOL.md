# Experiment protocol

## Mục tiêu

Tìm lightweight SR model nào làm ảnh sau SR có recognition accuracy cao hơn ảnh downsample baseline.

Trong bản workflow 10%, baseline chính là:

```text
lr_10p = ảnh gốc downsample còn 10% kích thước width/height
```

Các SR pipeline:

```text
span_10p_x4
safmn_10p_x4
rfdn_10p_x4
efdn_10p_x4
asid_10p_x4
catanet_10p_x4
lkfn_10p_x4
seemore_10p_x4
```

## Protocol A: matched training

Với mỗi pipeline ảnh, train recognition riêng trên train split của pipeline đó, validate trên val split, test trên test split.

Ví dụ:

```text
Train MobileNetV4 trên lr_10p/train  → test lr_10p/test
Train MobileNetV4 trên SPAN/train    → test SPAN/test
Train MobileNetV4 trên ASID/train    → test ASID/test
```

Câu hỏi trả lời:

```text
Nếu triển khai nguyên pipeline Downsample/SR + Recognition, pipeline nào tốt nhất?
```

## Metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Delta Accuracy so với lr_10p
- Delta Macro-F1 so với lr_10p

## Điều kiện claim SR có ích

So sánh **từng cặp** trên cùng recognition backbone — **không** lấy trung bình qua các model:

```text
span_r10_x4 + MobileNetV4  vs  lr_r10 + MobileNetV4
span_r10_x4 + RepViT       vs  lr_r10 + RepViT
...
```

Một SR method được xem là có ích cho một backbone nếu trên cặp đó:

1. Accuracy > baseline (`lr_r10`) cùng backbone.
2. Macro-F1 > baseline cùng backbone.

Claim tổng thể (optional): SR có ích nếu tăng trên **đa số** backbone (đếm số cặp thắng), không dùng mean Acc/F1.

## Protocol B: fixed recognizer, tùy chọn

Train recognizer trên lr_10p train một lần, sau đó test trên:

```text
lr_10p/test
span_10p_x4/test
safmn_10p_x4/test
...
```

Protocol này đo domain shift do SR tạo ra.


## Seed 42

Tất cả thí nghiệm dùng `project.seed: 42`. Downsample, DataLoader shuffle, augmentation random horizontal flip, và training đều được seed để mỗi lần chạy trên cùng môi trường cho kết quả lặp lại tốt nhất.
