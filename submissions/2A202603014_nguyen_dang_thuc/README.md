# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận

Học viên: **Nguyễn Đăng Thực** — MHV **2A202603014**

## Nội dung thư mục

| Đường dẫn | Nội dung |
|---|---|
| `report.md` | Báo cáo kết luận |
| `results.xlsx` | Bảng tổng hợp: Backbones, Training, Inference, Final, PerClass, Latency, Summary (+ Bonus_Shift) |
| `curves/` | Đường cong huấn luyện, mỗi `exp_id` một ảnh (`B0x_*`, `T0x_*`, `F01_*`) |
| `predictions/` | File dự đoán test của chung kết `F01` và mốc `T00` (3 seed), bản chưa hiệu chuẩn `F01_uncal_*`, dự đoán val của mọi lần chạy |
| `figures/` | EDA, kiểm tra pipeline, đánh đổi độ chính xác–độ trễ, reliability diagram, ma trận nhầm lẫn, ảnh đoán sai, Grad-CAM |
| `eval_results/` | Đầu ra nguyên bản của `eval.py score` và `eval.py grade` |
| `logs/` | `decisions.json` (mọi lựa chọn kèm lý do, chỉ dựa trên val), các bảng csv trung gian, kết quả test tự viết |
| `code/` | Toàn bộ code và notebook |

## Notebook chạy lại

Notebook nguồn: `code/lab_day2.ipynb`. Trên Kaggle em chạy bằng một script nhỏ: script chép code vào `/kaggle/working/repo/code`, rồi chạy notebook bằng nbclient và lưu bản đã chạy.

- Phần 1, Bước 0 và Bước 1: https://www.kaggle.com/code/nguyendangthuc11/lab-day2-deepweeds-2a202603014
  Phần này dừng ở ô hiển thị bảng sau Bước 1 vì lỗi tên cột. Bản đã chạy: `code/lab_day2_executed_part1.ipynb`.
- Phần 2, chạy tiếp đến hết: https://www.kaggle.com/code/nguyendangthuc11/lab-day2-deepweeds-2a202603014-resume
  Phần này gắn output của phần 1 làm input và dùng lại B01–B05, không train lại. Bản đã chạy: `code/lab_day2_executed_part2.ipynb`.

Thời gian chạy: phần 1 khoảng 11 phút, phần 2 khoảng 33 phút, trên một Tesla T4.

## Môi trường đã dùng

Kaggle Notebook, GPU Tesla T4, Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29,
pandas 2.3.3, fvcore (pip mới nhất lúc chạy), openpyxl. Trọng số tiền huấn luyện tải từ Hugging Face Hub qua timm
(tag ghi trong sheet `Backbones`).

## Cách chạy lại

Mọi thí nghiệm đi qua một hàm `train.run(Config(...))`. Notebook `code/lab_day2.ipynb` gọi lần lượt:

1. Tải dữ liệu (Zenodo, kiểm tra MD5; CSV fold 0 từ GitHub tác giả), `check_split`, EDA, các kiểm tra pipeline, `python -m unittest test_own`.
2. `experiments.step1(lab)` — 5 backbone, công thức nền T00, seed 0.
3. `experiments.step2(lab, backbone)` — T00 ba seed (đo nhiễu), T01–T06 mỗi lần khác T00 một yếu tố, T07 kết hợp.
4. `experiments.step3(lab)` — các cách suy luận trên val, đo độ trễ.
5. `experiments.step4(lab)` — chung kết F01 và mốc T00, 3 seed (0, 1, 2), test chạy một lần mỗi seed; `eval.py score` / `grade`.
6. `bonus.shift_eval`, `bonus.gradcam_errors`, rồi `make_results.main(out)` ghi `results.xlsx`.

Đặt `LAB_SMOKE=1` để chạy bản rút gọn kiểm tra code (1 epoch, 24 ảnh train mỗi lớp).

Chạy một thí nghiệm lẻ từ dòng lệnh (đứng trong `code/`, cần `eval.py` của repo ở thư mục cha):

```bash
python train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=../data/images labels_dir=../data/labels
```

Kiểm tra tự viết (CPU là đủ): `cd code && python -m unittest test_own -v`.

Seed dùng: 0 cho Bước 1–3; 0, 1, 2 cho T00 (đo nhiễu và mốc) và chung kết F01.
Ngân sách: ảnh train 128×128, 4 epoch, batch 128 (đặt trong `experiments.FAST`). Lý do và ảnh hưởng ghi ở report mục 2.4 và 8.
Checkpoint không commit (`.gitignore` chặn `*.pt`); nằm trong output của kernel Kaggle ở trên (`ckpt/`).
