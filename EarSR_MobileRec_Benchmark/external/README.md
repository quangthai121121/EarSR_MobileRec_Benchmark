# external/

Folder này để chứa các repo SR bên ngoài.

Chạy:

```bash
bash scripts/clone_sr_repos.sh
```

Repo mặc định:

```text
SPAN
SAFMN
RFDN
EFDN
ASID
CATANet
LKDN     # dùng thay LKFN nếu bạn chưa có repo LKFN cụ thể
seemoredetails
```

Sau khi clone, cần tải checkpoint của từng repo và sửa `command_template` trong:

```text
configs/benchmark_10p_all_sr_mobile_rec.yaml
```

Mỗi repo có CLI inference khác nhau nên project không hard-code cố định một command cho tất cả.
