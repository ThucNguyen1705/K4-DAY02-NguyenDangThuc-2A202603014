## Backbones

| exp_id | backbone | tag trọng số | #tham số (M) | GMAC | độ phân giải | epoch | seed | best epoch | macro-F1 val | top-1 val | F1 Chinee apple val | F1 Snake weed val | thời gian train/epoch (s) | độ trễ batch-1 p50 (ms) | độ trễ batch-1 p95 (ms) | thông lượng batch-32 (ảnh/s) | ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B01 | resnet50 | resnet50.a1_in1k | 23.5265 | 1.3419 | 128 | 4 | 0 | 4 | 0.5226 | 0.6778 | 0.3908 | 0.4242 | 13.4018 | 6.1106 | 6.6017 | 828.6985 | công thức nền T00, 1 seed; độ trễ FP32 đo ở Bước 3 (warmup 10, synchronize, 60 lần) |
| B02 | convnext_tiny | convnext_tiny.in12k_ft_in1k | 27.8270 | 1.4595 | 128 | 4 | 0 | 4 | 0.9412 | 0.9554 | 0.8707 | 0.8905 | 23.6124 | 5.7163 | 6.0587 | 625.4172 | công thức nền T00, 1 seed; độ trễ FP32 đo ở Bước 3 (warmup 10, synchronize, 60 lần) |
| B03 | deit_small_patch16_224 | deit_small_patch16_224.fb_in1k | 21.6184 | 1.4022 | 128 | 4 | 0 | 4 | 0.8934 | 0.9212 | 0.8018 | 0.8165 | 11.7537 | 4.7684 | 5.2230 | 771.6456 | công thức nền T00, 1 seed; độ trễ FP32 đo ở Bước 3 (warmup 10, synchronize, 60 lần) |
| B04 | efficientnet_b0 | efficientnet_b0.ra_in1k | 4.0191 | 0.1304 | 128 | 4 | 0 | 4 | 0.5851 | 0.6895 | 0.4537 | 0.4390 | 20.3751 | 7.9075 | 11.0973 | 2535.4406 | công thức nền T00, 1 seed; độ trễ FP32 đo ở Bước 3 (warmup 10, synchronize, 60 lần) |
| B05 | mobilenetv3_large_100 | mobilenetv3_large_100.ra_in1k | 4.2136 | 0.0750 | 128 | 4 | 0 | 4 | 0.5656 | 0.6815 | 0.4365 | 0.4021 | 15.3843 | 6.2659 | 6.5959 | 4554.2976 | công thức nền T00, 1 seed; độ trễ FP32 đo ở Bước 3 (warmup 10, synchronize, 60 lần) |

## Training

| exp_id | backbone | trục (A–G) | khác T00 ở điểm nào | seed | best epoch | macro-F1 val | top-1 val | Δ macro-F1 so với T00 | std nhiễu T00 (3 seed) | F1 Chinee apple val | F1 Snake weed val | F1 lớp thấp nhất val | F1 Negatives val | thời gian train/epoch (s) | ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T00 | convnext_tiny | - | công thức nền T00 | 0 | 4 | 0.9412 | 0.9554 | 0.0000 | 0.0037 | 0.8707 | 0.8905 | 0.8707 | 0.9723 | 23.6124 | công thức nền; std qua 3 seed = 0.0037 |
| T00 | convnext_tiny | - | công thức nền T00 | 1 | 4 | 0.9486 | 0.9603 | 0.0074 | 0.0037 | 0.9095 | 0.9027 | 0.9027 | 0.9739 | 24.4204 | công thức nền; std qua 3 seed = 0.0037 |
| T00 | convnext_tiny | - | công thức nền T00 | 2 | 4 | 0.9446 | 0.9572 | 0.0034 | 0.0037 | 0.8879 | 0.9023 | 0.8879 | 0.9718 | 18.6332 | công thức nền; std qua 3 seed = 0.0037 |
| T01 | convnext_tiny | A | init=frozen (chỉ train head) | 0 | 4 | 0.7476 | 0.8066 | -0.1936 | 0.0037 | 0.6667 | 0.6571 | 0.6571 | 0.8623 | 7.3256 | Δ vượt std nhiễu |
| T02 | convnext_tiny | B | aug=trivial (+TrivialAugmentWide) | 0 | 3 | 0.9484 | 0.9592 | 0.0072 | 0.0037 | 0.9241 | 0.8835 | 0.8835 | 0.9722 | 18.8424 | Δ vượt std nhiễu |
| T03 | convnext_tiny | B | mix=cutmix (alpha=1) | 0 | 4 | 0.9443 | 0.9572 | 0.0031 | 0.0037 | 0.8829 | 0.8939 | 0.8829 | 0.9725 | 18.8702 | không phân biệt được với T00 (|Δ| ≤ std) |
| T04 | convnext_tiny | C | loss=label smoothing eps=0.1 | 0 | 4 | 0.9459 | 0.9580 | 0.0047 | 0.0037 | 0.8868 | 0.8950 | 0.8868 | 0.9721 | 18.7518 | Δ vượt std nhiễu |
| T05 | convnext_tiny | C | loss=CE trọng số 1/n_c | 0 | 4 | 0.9327 | 0.9452 | -0.0085 | 0.0037 | 0.8658 | 0.8810 | 0.8658 | 0.9615 | 18.6635 | Δ vượt std nhiễu |
| T06 | convnext_tiny | F | EMA decay=0.99 | 0 | 3 | 0.9433 | 0.9560 | 0.0020 | 0.0037 | 0.8848 | 0.9027 | 0.8848 | 0.9710 | 18.8470 | không phân biệt được với T00 (|Δ| ≤ std) |
| T07 | convnext_tiny | kết hợp | kết hợp T02 + T04 | 0 | 3 | 0.9453 | 0.9574 | 0.0041 | 0.0037 | 0.8864 | 0.8906 | 0.8864 | 0.9719 | 18.8024 | Δ vượt std nhiễu |

## Inference

| exp_id | phương pháp | mô hình/checkpoint | K (view hoặc mô hình) | macro-F1 val | top-1 val | ECE val | NLL val | F1 Chinee apple val | F1 Snake weed val | p50 batch-1 (ms) | p95 batch-1 (ms) | p99 batch-1 (ms) | thông lượng batch-32 (ảnh/s) | chi phí tương đối so với I00 | ECE val trước TS | ECE val sau TS (khớp chéo 2 nửa) | T | ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I00 | 1 view (resize 146 + center crop 128) | T02 seed0 (convnext_tiny) | 1 | 0.9481 | 0.9589 | 0.0125 | 0.1398 | 0.9217 | 0.8835 | 5.8994 | 6.3850 | 6.7383 | 635.7935 | 1.0000 |  |  |  |  |
| I01 | TTA lật ngang, gộp xác suất | T02 seed0 (convnext_tiny) | 2 | 0.9460 | 0.9583 | 0.0143 | 0.1356 | 0.9103 | 0.8744 | 11.0673 | 12.2432 | 12.5834 | 298.1873 | 1.8760 |  |  |  |  |
| I01L | TTA lật ngang, gộp logit | T02 seed0 (convnext_tiny) | 2 | 0.9460 | 0.9583 | 0.0132 | 0.1345 | 0.9103 | 0.8744 | 11.0673 | 12.2432 | 12.5834 | 298.1873 | 1.8760 |  |  |  |  |
| I02 | TTA 5 crop 128 từ ảnh 146, gộp xác suất | T02 seed0 (convnext_tiny) | 5 | 0.9509 | 0.9614 | 0.0195 | 0.1292 | 0.9204 | 0.8862 | 27.4647 | 28.5729 | 29.4624 | 126.2971 | 4.6555 |  |  |  |  |
| I02L | TTA 5 crop 128 từ ảnh 146, gộp logit | T02 seed0 (convnext_tiny) | 5 | 0.9509 | 0.9614 | 0.0132 | 0.1256 | 0.9204 | 0.8841 | 27.4647 | 28.5729 | 29.4624 | 126.2971 | 4.6555 |  |  |  |  |
| I02F | TTA 10 crop (5 crop + lật), gộp xác suất | T02 seed0 (convnext_tiny) | 10 | 0.9527 | 0.9634 | 0.0236 | 0.1265 | 0.9207 | 0.8949 | 55.5610 | 59.3777 | 60.6155 | 63.0688 | 9.4181 |  |  |  |  |
| I04_R160 | độ phân giải kiểm tra 160 (resize 183 + crop 160) | T02 seed0 (convnext_tiny) | 1 | 0.9545 | 0.9640 | 0.0120 | 0.1157 | 0.9174 | 0.8932 | 5.5809 | 5.9512 | 6.6259 | 409.7590 | 0.9460 |  |  |  |  |
| I04_R192 | độ phân giải kiểm tra 192 (resize 219 + crop 192) | T02 seed0 (convnext_tiny) | 1 | 0.9494 | 0.9589 | 0.0132 | 0.1302 | 0.9159 | 0.9091 | 5.6118 | 5.9431 | 6.1458 | 280.0290 | 0.9513 |  |  |  |  |
| I04_R224 | độ phân giải kiểm tra 224 (resize 256 + crop 224) | T02 seed0 (convnext_tiny) | 1 | 0.9351 | 0.9463 | 0.0101 | 0.1668 | 0.8847 | 0.8789 | 5.8183 | 6.0661 | 6.8669 | 205.7141 | 0.9863 |  |  |  |  |
| I05a | ensemble 3 backbone tốt nhất Bước 1 (TB xác suất) | B02:convnext_tiny + B03:deit_small_patch16_224 + B04:efficientnet_b0 | 3 | 0.9240 | 0.9426 | 0.1025 | 0.2675 | 0.8492 | 0.8623 | 18.4494 | 19.7020 | 20.2230 | 306.4337 | 3.1274 |  |  |  |  |
| I05b | ensemble T00 ba seed (TB xác suất) | T00 seed0,1,2 | 3 | 0.9540 | 0.9646 | 0.0095 | 0.1111 | 0.9070 | 0.9100 | 16.7433 | 17.5517 | 18.3622 | 209.9546 | 2.8382 |  |  |  |  |
| I06 | trọng số EMA (T06) | T06 seed0 | 1 | 0.9433 | 0.9560 | 0.0063 | 0.1421 | 0.8848 | 0.9027 | 5.8994 | 6.3850 | 6.7383 | 635.7935 | 1.0000 |  |  |  |  |
| I06_raw | trọng số thường cùng epoch (T06) | T06 seed0 | 1 | 0.9369 | 0.9509 | 0.0093 | 0.1516 | 0.8838 | 0.8862 | 5.8994 | 6.3850 | 6.7383 | 635.7935 | 1.0000 |  |  |  |  |
| I07 | I00 + temperature scaling (T=0.915, khớp trên val) | T02 seed0 | 1 | 0.9481 | 0.9589 | 0.0071 | 0.1388 | 0.9217 | 0.8835 | 5.8994 | 6.3850 | 6.7383 | 635.7935 | 1.0000 | 0.0125 | 0.0100 | 0.9150 |  |
| I08_fp16 | FP16 (model.half()) | T02 seed0 | 1 | 0.9481 | 0.9589 | 0.0124 | 0.1397 | 0.9217 | 0.8835 | 5.4208 | 5.8551 | 6.1762 | 2220.4096 | 0.9189 |  |  |  |  |
| I08_amp | AMP autocast FP16 | T02 seed0 | 1 | 0.9484 | 0.9592 | 0.0127 | 0.1397 | 0.9241 | 0.8835 | 7.7136 | 8.0447 | 8.7998 | 1742.7392 | 1.3075 |  |  |  |  |
| I08_fuse | gộp BN vào conv | T02 seed0 (convnext_tiny) | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | convnext_tiny không có BatchNorm (dùng LayerNorm), không gộp được |
| I08_unfused | chưa gộp BN (đối chứng) | B01 seed0 (resnet50) | 1 | 0.5224 | 0.6775 | 0.0514 | 0.9499 | 0.3977 | 0.4362 | 5.8399 | 6.3751 | 6.6507 | 791.4626 | 0.9899 |  |  |  |  |
| I08_fused | gộp BN vào conv | B01 seed0 (resnet50) | 1 | 0.5224 | 0.6775 | 0.0514 | 0.9499 | 0.3977 | 0.4362 | 4.4861 | 4.6902 | 4.8964 | 836.8107 | 0.7604 |  |  |  | 53 cặp conv-BN, sai số đầu ra lớn nhất 1.62e-05 |

## Final

| exp_id | cấu hình | seed | macro-F1 val | macro-F1 test | top-1 test | balanced acc test | ECE test | file |
|---|---|---|---|---|---|---|---|---|
| F01 | convnext_tiny + công thức T02 {'aug': 'trivial'} + suy luận R160 + TS | 0 | 0.9545 | 0.9544 | 0.9638 | 0.9499 | 0.0057 | F01_seed0_test.csv |
| F01 | convnext_tiny + công thức T02 {'aug': 'trivial'} + suy luận R160 + TS | 1 | 0.9520 | 0.9576 | 0.9644 | 0.9454 | 0.0051 | F01_seed1_test.csv |
| F01 | convnext_tiny + công thức T02 {'aug': 'trivial'} + suy luận R160 + TS | 2 | 0.9466 | 0.9497 | 0.9604 | 0.9341 | 0.0051 | F01_seed2_test.csv |
| F01 | convnext_tiny + công thức T02 {'aug': 'trivial'} + suy luận R160 + TS — mean ± std | 3 seed | 0.9510 ± 0.0040 | 0.9539 ± 0.0040 | 0.9628 ± 0.0022 | 0.9431 ± 0.0081 | 0.0053 ± 0.0003 | eval.py score |
| T00 | convnext_tiny + công thức nền T00 + suy luận I00 (1 view) | 0 | 0.9412 | 0.9567 | 0.9661 | 0.9496 | 0.0071 | T00_seed0_test.csv |
| T00 | convnext_tiny + công thức nền T00 + suy luận I00 (1 view) | 1 | 0.9486 | 0.9529 | 0.9626 | 0.9516 | 0.0063 | T00_seed1_test.csv |
| T00 | convnext_tiny + công thức nền T00 + suy luận I00 (1 view) | 2 | 0.9446 | 0.9466 | 0.9598 | 0.9400 | 0.0070 | T00_seed2_test.csv |
| T00 | convnext_tiny + công thức nền T00 + suy luận I00 (1 view) — mean ± std | 3 seed | 0.9448 ± 0.0037 | 0.9520 ± 0.0051 | 0.9628 ± 0.0031 | 0.9471 ± 0.0062 | 0.0068 ± 0.0004 | eval.py score |

## PerClass

| cấu hình | lớp | số ảnh test | precision | recall | F1 |
|---|---|---|---|---|---|
| chung kết F01 (mean ± std 3 seed) | Chinee apple | 226 | 0.9552 ± 0.0053 | 0.8791 ± 0.0135 | 0.9155 ± 0.0076 |
| chung kết F01 (mean ± std 3 seed) | Lantana | 213 | 0.9758 ± 0.0140 | 0.9390 ± 0.0141 | 0.9569 ± 0.0072 |
| chung kết F01 (mean ± std 3 seed) | Parkinsonia | 207 | 0.9683 ± 0.0070 | 0.9823 ± 0.0028 | 0.9752 ± 0.0027 |
| chung kết F01 (mean ± std 3 seed) | Parthenium | 205 | 0.9844 ± 0.0053 | 0.9268 ± 0.0129 | 0.9547 ± 0.0089 |
| chung kết F01 (mean ± std 3 seed) | Prickly acacia | 213 | 0.9596 ± 0.0074 | 0.9280 ± 0.0072 | 0.9435 ± 0.0069 |
| chung kết F01 (mean ± std 3 seed) | Rubber vine | 202 | 0.9722 ± 0.0177 | 0.9703 ± 0.0099 | 0.9711 ± 0.0048 |
| chung kết F01 (mean ± std 3 seed) | Siam weed | 215 | 0.9795 ± 0.0072 | 0.9628 ± 0.0047 | 0.9711 ± 0.0059 |
| chung kết F01 (mean ± std 3 seed) | Snake weed | 204 | 0.9355 ± 0.0229 | 0.9134 ± 0.0279 | 0.9239 ± 0.0086 |
| chung kết F01 (mean ± std 3 seed) | Negative | 1822 | 0.9600 ± 0.0081 | 0.9866 ± 0.0055 | 0.9731 ± 0.0015 |
| mốc T00+I00 (mean ± std 3 seed) | Chinee apple | 226 | 0.9270 ± 0.0126 | 0.8997 ± 0.0284 | 0.9130 ± 0.0193 |
| mốc T00+I00 (mean ± std 3 seed) | Lantana | 213 | 0.9727 ± 0.0024 | 0.9468 ± 0.0143 | 0.9595 ± 0.0066 |
| mốc T00+I00 (mean ± std 3 seed) | Parkinsonia | 207 | 0.9621 ± 0.0093 | 0.9807 ± 0.0048 | 0.9713 ± 0.0071 |
| mốc T00+I00 (mean ± std 3 seed) | Parthenium | 205 | 0.9914 ± 0.0078 | 0.9350 ± 0.0028 | 0.9624 ± 0.0023 |
| mốc T00+I00 (mean ± std 3 seed) | Prickly acacia | 213 | 0.9193 ± 0.0105 | 0.9609 ± 0.0072 | 0.9396 ± 0.0021 |
| mốc T00+I00 (mean ± std 3 seed) | Rubber vine | 202 | 0.9718 ± 0.0122 | 0.9620 ± 0.0029 | 0.9669 ± 0.0050 |
| mốc T00+I00 (mean ± std 3 seed) | Siam weed | 215 | 0.9749 ± 0.0184 | 0.9535 ± 0.0081 | 0.9640 ± 0.0079 |
| mốc T00+I00 (mean ± std 3 seed) | Snake weed | 204 | 0.9293 ± 0.0068 | 0.9036 ± 0.0311 | 0.9161 ± 0.0193 |
| mốc T00+I00 (mean ± std 3 seed) | Negative | 1822 | 0.9697 ± 0.0032 | 0.9817 ± 0.0052 | 0.9756 ± 0.0014 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Chinee Apple | 226 | 0.9528 | 0.8938 | 0.9224 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Lantana | 213 | 0.9621 | 0.9531 | 0.9575 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Parkinsonia | 207 | 0.9623 | 0.9855 | 0.9737 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Parthenium | 205 | 0.9896 | 0.9317 | 0.9598 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Prickly Acacia | 213 | 0.9659 | 0.9296 | 0.9474 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Rubber Vine | 202 | 0.9519 | 0.9802 | 0.9659 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Siam Weed | 215 | 0.9717 | 0.9581 | 0.9649 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Snake Weed | 204 | 0.9095 | 0.9363 | 0.9227 |
| F01 seed 0 (seed có macro-F1 VAL cao nhất) | Negatives | 1822 | 0.9691 | 0.9808 | 0.9749 |

## Latency

| cấu hình | phương pháp | GPU | torch | dtype | batch | độ phân giải | gộp BN | tính tiền xử lý | p50 (ms) | p95 (ms) | p99 (ms) | số lần đo | ảnh/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T02 (convnext_tiny) I00 fp32 | I00 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 5.8994 | 6.3850 | 6.7383 | 60 | 169.5098 |
| T02 (convnext_tiny) I00 amp | I08_amp | Tesla T4 | 2.11.0+cu128 | amp | 1 | 128.0000 | không | không | 7.7136 | 8.0447 | 8.7998 | 60 | 129.6415 |
| T02 (convnext_tiny) I00 fp16 | I08_fp16 | Tesla T4 | 2.11.0+cu128 | fp16 | 1 | 128.0000 | không | không | 5.4208 | 5.8551 | 6.1762 | 60 | 184.4740 |
| B01 seed0 (resnet50) gộp BN fp32 | I08_fused | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | có | không | 4.4861 | 4.6902 | 4.8964 | 60 | 222.9123 |
| B01 seed0 (resnet50) chưa gộp BN fp32 | I08_unfused | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 5.8399 | 6.3751 | 6.6507 | 60 | 171.2354 |
| T02 (convnext_tiny) I01 K=2 tuần tự | I01 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 11.0673 | 12.2432 | 12.5834 | 60 | 90.3565 |
| T02 (convnext_tiny) I01 K=2 gộp batch | I01_batched | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 5.8481 | 8.7569 | 10.4267 | 60 | 170.9965 |
| T02 (convnext_tiny) I02 K=5 tuần tự | I02 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 27.4647 | 28.5729 | 29.4624 | 60 | 36.4103 |
| T02 (convnext_tiny) I02 K=5 gộp batch | I02_batched | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 7.9899 | 8.6807 | 8.8984 | 60 | 125.1576 |
| T02 (convnext_tiny) I02F K=10 tuần tự | I02F | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 55.5610 | 59.3777 | 60.6155 | 60 | 17.9982 |
| T02 (convnext_tiny) I02F K=10 gộp batch | I02F_batched | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 15.8549 | 16.1384 | 16.2460 | 60 | 63.0721 |
| T02 (convnext_tiny) res 160 | I04_R160 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 160.0000 | không | không | 5.5809 | 5.9512 | 6.6259 | 60 | 179.1820 |
| T02 (convnext_tiny) res 192 | I04_R192 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 192.0000 | không | không | 5.6118 | 5.9431 | 6.1458 | 60 | 178.1965 |
| T02 (convnext_tiny) res 224 | I04_R224 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 224.0000 | không | không | 5.8183 | 6.0661 | 6.8669 | 60 | 171.8716 |
| ensemble convnext_tiny+deit_small_patch16_224+efficientnet_b0 | I05a | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 18.4494 | 19.7020 | 20.2230 | 60 | 54.2022 |
| ensemble T00 x3 seed | I05b | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 16.7433 | 17.5517 | 18.3622 | 60 | 59.7254 |
| T02 (convnext_tiny) I00 fp32 | I00 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 50.3308 | 51.5107 | 52.3042 | 60 | 635.7935 |
| T02 (convnext_tiny) I00 amp | I08_amp | Tesla T4 | 2.11.0+cu128 | amp | 32 | 128.0000 | không | không | 18.3619 | 18.6203 | 18.7227 | 60 | 1742.7392 |
| T02 (convnext_tiny) I00 fp16 | I08_fp16 | Tesla T4 | 2.11.0+cu128 | fp16 | 32 | 128.0000 | không | không | 14.4118 | 14.8211 | 14.9367 | 60 | 2220.4096 |
| B01 seed0 (resnet50) gộp BN fp32 | I08_fused | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | có | không | 38.2404 | 39.0680 | 39.2576 | 60 | 836.8107 |
| B01 seed0 (resnet50) chưa gộp BN fp32 | I08_unfused | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 40.4315 | 41.3193 | 41.4894 | 60 | 791.4626 |
| T02 (convnext_tiny) I01 K=2 tuần tự | I01 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 107.3151 | 108.5823 | 109.0956 | 60 | 298.1873 |
| T02 (convnext_tiny) I01 K=2 gộp batch | I01_batched | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 103.4315 | 105.3959 | 105.6814 | 60 | 309.3834 |
| T02 (convnext_tiny) I02 K=5 tuần tự | I02 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 253.3709 | 258.3549 | 259.1679 | 60 | 126.2971 |
| T02 (convnext_tiny) I02 K=5 gộp batch | I02_batched | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 242.7470 | 243.8038 | 244.0730 | 60 | 131.8245 |
| T02 (convnext_tiny) I02F K=10 tuần tự | I02F | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 507.3824 | 513.6888 | 516.3229 | 60 | 63.0688 |
| T02 (convnext_tiny) I02F K=10 gộp batch | I02F_batched | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 485.8518 | 487.1992 | 487.9512 | 60 | 65.8637 |
| T02 (convnext_tiny) res 160 | I04_R160 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 160.0000 | không | không | 78.0947 | 79.4855 | 80.2505 | 60 | 409.7590 |
| T02 (convnext_tiny) res 192 | I04_R192 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 192.0000 | không | không | 114.2739 | 116.0841 | 116.6357 | 60 | 280.0290 |
| T02 (convnext_tiny) res 224 | I04_R224 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 224.0000 | không | không | 155.5557 | 158.6379 | 158.9575 | 60 | 205.7141 |
| ensemble convnext_tiny+deit_small_patch16_224+efficientnet_b0 | I05a | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 104.4272 | 106.2403 | 106.8563 | 60 | 306.4337 |
| ensemble T00 x3 seed | I05b | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 152.4139 | 154.9768 | 155.7437 | 60 | 209.9546 |
| B01 resnet50 | B01 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 6.1106 | 6.6017 | 6.9156 | 60 | 163.6493 |
| B01 resnet50 | B01 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 38.6148 | 39.3006 | 39.5582 | 60 | 828.6985 |
| B02 convnext_tiny | B02 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 5.7163 | 6.0587 | 6.2149 | 60 | 174.9382 |
| B02 convnext_tiny | B02 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 51.1658 | 52.0893 | 52.2830 | 60 | 625.4172 |
| B03 deit_small_patch16_224 | B03 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 4.7684 | 5.2230 | 5.6996 | 60 | 209.7149 |
| B03 deit_small_patch16_224 | B03 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 41.4698 | 43.2327 | 43.3505 | 60 | 771.6456 |
| B04 efficientnet_b0 | B04 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 7.9075 | 11.0973 | 14.5148 | 60 | 126.4618 |
| B04 efficientnet_b0 | B04 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 12.6211 | 13.0100 | 13.2588 | 60 | 2535.4406 |
| B05 mobilenetv3_large_100 | B05 | Tesla T4 | 2.11.0+cu128 | fp32 | 1 | 128.0000 | không | không | 6.2659 | 6.5959 | 6.8017 | 60 | 159.5931 |
| B05 mobilenetv3_large_100 | B05 | Tesla T4 | 2.11.0+cu128 | fp32 | 32 | 128.0000 | không | không | 7.0263 | 7.6384 | 7.6417 | 60 | 4554.2976 |
| T02 (convnext_tiny) I00 fp32 + tiền xử lý | I00_preproc | Tesla T4 | 2.11.0+cu128 | fp32 | 1 |  | không | có | 7.9159 | 8.8347 | 11.3709 | 60 | 126.3285 |

## Summary

| hạng | exp_id | loại | mô tả | macro-F1 val | top-1 val | #tham số (M) | GMAC | p50 batch-1 (ms) | train s/epoch | chi phí so với I00 | macro-F1 test (mean ± std) | top-1 test (mean ± std) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | I04_R160 | suy luận | độ phân giải kiểm tra 160 (resize 183 + crop 160) | 0.9545 | 0.9640 |  |  | 5.5809 |  | 0.9460 |  |  |
| 2 | I05b | suy luận | ensemble T00 ba seed (TB xác suất) | 0.9540 | 0.9646 |  |  | 16.7433 |  | 2.8382 |  |  |
| 3 | I02F | suy luận | TTA 10 crop (5 crop + lật), gộp xác suất | 0.9527 | 0.9634 |  |  | 55.5610 |  | 9.4181 |  |  |
| 4 | I02L | suy luận | TTA 5 crop 128 từ ảnh 146, gộp logit | 0.9509 | 0.9614 |  |  | 27.4647 |  | 4.6555 |  |  |
| 5 | I02 | suy luận | TTA 5 crop 128 từ ảnh 146, gộp xác suất | 0.9509 | 0.9614 |  |  | 27.4647 |  | 4.6555 |  |  |
| 6 | I04_R192 | suy luận | độ phân giải kiểm tra 192 (resize 219 + crop 192) | 0.9494 | 0.9589 |  |  | 5.6118 |  | 0.9513 |  |  |
| 7 | T02 | công thức (trục B) | aug=trivial (+TrivialAugmentWide) | 0.9484 | 0.9592 |  |  |  | 18.8424 |  |  |  |
| 8 | I08_amp | suy luận | AMP autocast FP16 | 0.9484 | 0.9592 |  |  | 7.7136 |  | 1.3075 |  |  |
| 9 | I08_fp16 | suy luận | FP16 (model.half()) | 0.9481 | 0.9589 |  |  | 5.4208 |  | 0.9189 |  |  |
| 10 | I01 | suy luận | TTA lật ngang, gộp xác suất | 0.9460 | 0.9583 |  |  | 11.0673 |  | 1.8760 |  |  |
| chung kết | F01 | chung kết 3 seed | convnext_tiny + công thức T02 {'aug': 'trivial'} + suy luận R160 + TS — mean ± std | 0.9510 ± 0.0040 | None |  |  |  |  |  | 0.9539 ± 0.0040 | 0.9628 ± 0.0022 |
| chung kết | T00 | chung kết 3 seed | convnext_tiny + công thức nền T00 + suy luận I00 (1 view) — mean ± std | 0.9448 ± 0.0037 | None |  |  |  |  |  | 0.9520 ± 0.0051 | 0.9628 ± 0.0031 |

## Bonus_Shift

| lệch phân phối (val) | macro-F1 val | top-1 val | recall Chinee apple | recall Snake weed | ECE trước TS | ECE sau TS (T từ val sạch) | T |
|---|---|---|---|---|---|---|---|
| sạch | 0.9481 | 0.9589 | 0.9156 | 0.8966 | 0.0125 | 0.0071 | 0.9150 |
| mờ (Gaussian r=2.5) | 0.4433 | 0.6392 | 0.2800 | 0.1182 | 0.1435 | 0.1652 | 0.9150 |
| tối (độ sáng x0.4) | 0.9085 | 0.9240 | 0.9111 | 0.8719 | 0.0151 | 0.0101 | 0.9150 |
| nhiễu Gaussian std=0.08 | 0.7989 | 0.8389 | 0.6400 | 0.7389 | 0.0498 | 0.0611 | 0.9150 |
| JPEG quality 10 | 0.3905 | 0.6198 | 0.2756 | 0.1084 | 0.1582 | 0.1803 | 0.9150 |

## decisions.json

```json
{
  "step1_backbone": {
    "exp_id": "B02",
    "backbone": "convnext_tiny",
    "reason": "macro-F1 val cao nhất (0.9412) trong 5 backbone, 1 seed"
  },
  "step2_noise_std": 0.0037205331719145345,
  "step2_combination": {
    "members": [
      "T02",
      "T04"
    ],
    "cfg": {
      "aug": "trivial",
      "loss": "ls",
      "label_smoothing": 0.1
    },
    "note": "các yếu tố đều có Δ > std nhiễu"
  },
  "step2_recipe": {
    "exp_id": "T02",
    "cfg": {
      "aug": "trivial"
    },
    "backbone": "convnext_tiny",
    "val_macro_f1": 0.948393773619887,
    "delta_vs_T00": 0.007181627455327688,
    "reason": "macro-F1 val cao nhất (0.9484) trong các cấu hình seed 0; Δ so với T00 = +0.0072, std nhiễu T00 = 0.0037 (vượt nhiễu)"
  },
  "step3_method": {
    "method": "R160",
    "desc": "độ phân giải kiểm tra 160 (resize 183 + crop 160)",
    "val_macro_f1": 0.9545048386497371,
    "delta_vs_I00": 0.006408210017816662,
    "p95_b1_ms": 5.951188249844108,
    "temperature_scaling": true,
    "reason": "macro-F1 val cao nhất trong các cách suy luận một mô hình; Δ so với I00 = +0.0064 (std nhiễu seed 0.0037); p95 batch 1 = 6.0 ms. Sau đó áp temperature scaling khớp trên val."
  }
}
```
