# Dự án: Phân tích mô hình LightGCN (RecBole) - Giai đoạn 1 (MovieLens 100k)

Ngày cập nhật: 9 tháng 4, 2026

## 1. Ý tưởng cốt lõi và Kiến trúc

### Khái niệm
LightGCN là mô hình gợi ý dựa trên đồ thị user-item bipartite, tập trung vào lan truyền embedding qua các lớp graph convolution đơn giản hóa. So với các mô hình MF/embedding thuần, LightGCN nhấn mạnh thông tin cấu trúc lân cận trong đồ thị tương tác.

### Mục tiêu
Mục tiêu là học biểu diễn user/item tốt hơn cho bài toán ranking Top-K, sau đó đề xuất danh sách item phù hợp cho từng user trên MovieLens 100k.

### Đầu vào/Đầu ra
- Đầu vào: tương tác user-item (implicit feedback theo pipeline RecBole cho bài toán ranking).
- Đầu ra: điểm xếp hạng và danh sách Top-10 item đề xuất cho mỗi user.

## 2. Thiết lập dữ liệu và huấn luyện

### Bộ dữ liệu
- Dataset: `ml-100k` (MovieLens 100k).
- Quy mô theo log: 944 users, 1683 items, 100000 interactions.
- Độ thưa: ~93.71%.

### Cấu hình huấn luyện chính (suy ra từ log RecBole)
- Model: `LightGCN`
- epochs: `300`
- eval_step: `1`
- stopping_step: `10`
- optimizer: `adam`
- learning_rate: `0.001`
- embedding_size: `64`
- n_layers: `2`
- reg_weight: `1e-05`
- train_batch_size: `2048`
- eval_batch_size: `4096`
- valid_metric: `MRR@10` (maximize)
- seed: `2020`

zGhi chú thực thi:
- Log cho thấy `use_gpu=True` trong cấu hình tổng quát, nhưng `device=cpu` ở runtime của lần chạy này.

## 3. Chỉ số và Kết quả thực nghiệm

### Nguồn kết quả
Dựa trên dòng mới nhất trong `summary.csv`:

- timestamp: `2026-04-08 09:58:09`
- model: `LightGCN`
- dataset: `ml-100k`
- best_epoch: `14`

### Kết quả tốt nhất (best eval result)

| Chỉ số | Kết quả |
|---|---:|
| Recall@10 | 0.1626 |
| MRR@10 | 0.3162 |
| NDCG@10 | 0.1765 |
| Hit@10 | 0.6394 |
| Precision@10 | 0.1205 |

### Diễn giải nhanh chất lượng gợi ý
- `Hit@10 = 0.6394`: khoảng 63.94% user có ít nhất một item đúng trong top-10, cho thấy khả năng "bắt trúng" mục tiêu khá tốt.
- `MRR@10 = 0.3162`: vị trí item đúng đầu tiên thường nằm khá sớm trong top-10, phản ánh chất lượng thứ hạng đầu danh sách tốt.
- `NDCG@10 = 0.1765`: xét thêm yếu tố vị trí và nhiều item đúng, chất lượng ranking ở mức ổn định cho baseline graph-based.
- `Recall@10 = 0.1626` cao hơn `Precision@10 = 0.1205`: đúng với bối cảnh top-10 bị giới hạn nhưng mỗi user có thể có nhiều item liên quan.

## 4. Phân tích hiệu năng mô hình

### Điểm mạnh thể hiện qua metric
- Cân bằng tốt giữa khả năng tìm đúng (`Hit@10`, `Recall@10`) và thứ hạng sớm (`MRR@10`).
- Giá trị `MRR@10` tương đối cao so với mặt bằng baseline, cho thấy item đúng xuất hiện sớm trong danh sách đề xuất.
- Trên dữ liệu thưa (~93.71%), mô hình vẫn giữ được hiệu năng ranking ổn định.

### Hạn chế/điểm cần lưu ý
- Kết quả hiện tại là từ một lần chạy chính (một seed), chưa có thống kê mean ± std qua nhiều seed.
- `summary.csv` đang lưu best validation result; nếu cần báo cáo cuối cùng cho paper/report nghiêm ngặt nên bổ sung rõ test result tương ứng best checkpoint.
- Chưa có benchmark theo nhiều cấu hình (layer depth, reg_weight, learning_rate), nên dư địa tuning vẫn lớn.

### Phân tích tương quan metric
- `MRR@10` cao cùng với `Hit@10` cao: mô hình không chỉ "chạm đúng" mà còn thường đặt item đúng ở vị trí trên cao.
- `Recall@10` vượt `Precision@10`: mô hình thu hồi được lượng item liên quan tốt, nhưng top-10 vẫn chứa một phần item chưa thật sự tối ưu.

## 5. So sánh theo tư duy LightGCN vs LightFM

### Khác biệt phương pháp
- LightFM (baseline trước đó): embedding-based, chủ yếu học từ tương tác (và có thể mở rộng hybrid nếu có metadata).
- LightGCN: graph-based, khai thác trực tiếp cấu trúc đồ thị user-item qua message passing đơn giản hóa.

### Kỳ vọng hành vi trên dữ liệu thưa
- Với dữ liệu sparse, LightGCN thường có lợi thế vì tận dụng tín hiệu lân cận bậc cao (multi-hop) tốt hơn matrix factorization thuần.
- LightFM thường dễ triển khai và chạy nhanh, nhưng có thể kém hơn về khai thác cấu trúc đồ thị nếu chỉ dùng CF thuần.

### Tham chiếu nhanh với report LightFM hiện có
- LightFM (bản tham chiếu): Precision@10 = 0.0737, Recall@10 = 0.1323, MRR@10 = 0.2357, NDCG@10 = 0.1188, Hit@10 = 0.5385.
- LightGCN (run hiện tại): Precision@10 = 0.1205, Recall@10 = 0.1626, MRR@10 = 0.3162, NDCG@10 = 0.1765, Hit@10 = 0.6394.

Nhận định:
- LightGCN đang cho kết quả cao hơn trên tất cả metric chính trong so sánh này.
- Tuy nhiên, cần lưu ý công bằng thực nghiệm: cùng dataset nhưng có thể khác pipeline xử lý/đánh giá chi tiết giữa hai framework, nên kết luận cuối nên đi kèm protocol đồng nhất.

## 6. Phân tích giao thức đánh giá RecBole 

### 6.1 Chiến lược chia dữ liệu (Data Split)
Cấu hình từ log:
- `split: {'RS': [0.8, 0.1, 0.1]}`
- `order: 'RO'`

Ý nghĩa:
- `RS` (Random Split): chia ngẫu nhiên tương tác thành 80% train, 10% validation, 10% test.
- Train: dùng để cập nhật tham số mô hình.
- Validation: dùng để theo dõi trong huấn luyện, chọn epoch tốt nhất và early stopping.
- Test: dùng để đánh giá cuối cùng sau khi đã chọn mô hình.
- `order='RO'`: random order, tức thứ tự tương tác được xem theo kiểu ngẫu nhiên cho quy trình chia/đánh giá tương ứng cấu hình ranking hiện tại.

### 6.2 Phương pháp đánh giá (Evaluation Protocol)
Cấu hình từ log:
- `group_by: 'user'`
- `mode: {'valid': 'full', 'test': 'full'}`

Ý nghĩa:
- `group_by='user'`: metric được tính theo từng user rồi tổng hợp, phù hợp bản chất bài toán recommendation cá nhân hóa.
- `mode='full'`: full ranking, mỗi user được rank trên toàn bộ candidate items (sau khi loại trừ phần đã thấy theo protocol nội bộ).

Full ranking vs sampled ranking:
- Full ranking đáng tin cậy hơn vì phản ánh đúng độ khó thực tế khi chọn top-K từ không gian item lớn.
- Sampled ranking nhanh hơn nhưng có thể lạc quan giả tạo do số lượng negative items bị giới hạn.

### 6.3 Giải thích các metric
- `Recall@10`: tỷ lệ item liên quan được thu hồi trong top-10; nhấn mạnh độ bao phủ item đúng.
- `MRR@10`: trung bình nghịch đảo hạng của item đúng đầu tiên trong top-10; nhấn mạnh vị trí đúng sớm.
- `NDCG@10`: đo chất lượng thứ hạng có xét vị trí và độ giảm trọng số theo rank; cân bằng giữa đúng và thứ tự.
- `Hit@10`: user có ít nhất một item đúng trong top-10 hay không; phản ánh khả năng "gợi ý trúng".
- `Precision@10`: tỷ lệ item đúng trong top-10; nhấn mạnh độ tinh gọn/chính xác danh sách.

Khác biệt trọng tâm:
- Recall thiên về bao phủ.
- Precision thiên về độ tinh gọn.
- MRR/NDCG thiên về chất lượng thứ hạng (đúng sớm quan trọng hơn).
- Hit thiên về xác suất thành công ở mức user.

## 7. Đánh giá tổng quan: Ưu và Nhược điểm

### Ưu điểm
- Hiệu năng ranking tốt trên bộ MovieLens 100k trong cấu hình hiện tại.
- Khả năng xử lý dữ liệu thưa tốt nhờ khai thác cấu trúc đồ thị user-item.
- Dễ tích hợp tracking qua `summary.csv` + `history.csv` cho mục tiêu báo cáo thực nghiệm.

### Nhược điểm
- Chưa có thống kê đa seed, nên độ ổn định chưa được định lượng đầy đủ.
- Chưa benchmark sâu theo nhiều cấu hình LightGCN.
- Kết quả hiện tại tập trung vào best validation; cần chuẩn hóa báo cáo test để hoàn chỉnh hơn.

## 8. Kết luận

LightGCN trong RecBole cho thấy hiệu quả tốt trên MovieLens 100k với chất lượng gợi ý Top-10 đồng đều ở cả nhóm metric thu hồi (Recall/Hit), độ chính xác (Precision), và chất lượng thứ hạng (MRR/NDCG). Với dữ liệu sparse và mục tiêu recommendation theo ranking, LightGCN là lựa chọn rất phù hợp khi cần một baseline mạnh hơn các mô hình embedding thuần.

Trong bối cảnh coursework hoặc tài liệu kỹ thuật, cấu hình hiện tại đã đủ tốt để làm mốc so sánh. Để nâng mức thuyết phục cho báo cáo nghiên cứu, bước tiếp theo nên là chuẩn hóa protocol test result, chạy đa seed, và mở rộng hyperparameter search có kiểm soát.

## 9. Cập nhật thực nghiệm mở rộng (Multi-seed + Hyperparameter Search)

### 9.1 Experimental Stability (đa seed)

Nguồn dữ liệu: `csv_result/multi_seed_results.csv` và `csv_result/summary_multi_seed.csv`.

Kết quả từng seed (test result):

| Seed | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|
| 2020 | 0.1853 | 0.3760 | 0.2116 | 0.6819 | 0.1406 |
| 2021 | 0.2156 | 0.4403 | 0.2591 | 0.7349 | 0.1788 |
| 2022 | 0.2549 | 0.4760 | 0.2911 | 0.7964 | 0.1992 |
| 2023 | 0.1816 | 0.3689 | 0.2087 | 0.6723 | 0.1437 |
| 2024 | 0.2378 | 0.4584 | 0.2786 | 0.7847 | 0.1919 |

Tổng hợp mean ± std (n=5):

| Statistic | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|
| Mean | 0.2150 | 0.4239 | 0.2498 | 0.7340 | 0.1708 |
| Std | 0.0321 | 0.0487 | 0.0380 | 0.0570 | 0.0272 |

Nhận định:
- Hiệu năng trung bình cao hơn đáng kể so với single-run ban đầu, cho thấy cấu hình LightGCN có khả năng đạt chất lượng tốt khi đánh giá trên test set.
- Độ lệch chuẩn chưa nhỏ (đặc biệt MRR/NDCG/Hit), nghĩa là độ ổn định theo seed ở mức trung bình; nên báo cáo kết quả bằng mean ± std thay vì chỉ một seed đại diện.
- Seed ảnh hưởng rõ rệt đến hiệu năng xếp hạng, vì vậy đánh giá đa seed là bắt buộc cho kết luận mang tính nghiên cứu.

### 9.2 Hyperparameter Sensitivity

Nguồn dữ liệu:
- `csv_result/hyperparam_results_fast.csv` (4 tổ hợp theo cấu hình gọn để hoàn tất nhanh, cùng protocol đánh giá).

Kết quả fast profile:

| embedding_size | n_layers | reg_weight | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 1e-4 | 0.1867 | 0.3834 | 0.2175 | 0.6819 | 0.1456 |
| 32 | 2 | 1e-4 | 0.1878 | 0.3699 | 0.2119 | 0.6840 | 0.1415 |
| 64 | 1 | 1e-4 | 0.2001 | 0.4179 | 0.2382 | 0.7105 | 0.1601 |
| 64 | 2 | 1e-4 | 0.1360 | 0.2803 | 0.1511 | 0.5960 | 0.1041 |

Best config trong fast profile (theo MRR@10):
- embedding_size = 64
- n_layers = 1
- reg_weight = 1e-4

Phân tích độ nhạy:
- `embedding_size` tăng từ 32 lên 64 có thể cải thiện rõ rệt (khi giữ `n_layers=1`), cho thấy dung lượng biểu diễn là yếu tố quan trọng.
- `n_layers` có ảnh hưởng mạnh và không đơn điệu: `n_layers=2` cho kết quả kém rõ ở setting embedding_size=64 trong fast profile.
- Kết quả gợi ý cần tuning cẩn thận số lớp; tăng depth không đảm bảo tăng chất lượng trên ml-100k.

### 9.3 Final Conclusion Update

- Kết luận hiện tại đã chuyển từ single-run sang evidence-based theo đa seed: báo cáo chính nên ưu tiên mean ± std trên test set.
- LightGCN vẫn là lựa chọn mạnh cho bài toán ranking trên dữ liệu sparse, nhưng độ biến thiên theo seed cần được phản ánh minh bạch trong báo cáo.
- Hyperparameter tuning cho thấy ảnh hưởng đáng kể của `embedding_size` và đặc biệt `n_layers`; vì vậy kết luận cuối cùng cần gắn với cấu hình tốt nhất đã kiểm chứng, không chỉ cấu hình mặc định.
- Để hoàn thiện mức research-quality, cần hoàn tất full hyperparameter grid và giữ nguyên protocol đánh giá (split, full ranking, metric set) cho mọi run so sánh.
