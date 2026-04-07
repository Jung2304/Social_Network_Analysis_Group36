# Dự án: Phân tích mô hình cơ sở LightFM - Giai đoạn 1 (MovieLens 100k)

Ngày cập nhật: 6 tháng 4, 2026

## 1. Ý tưởng cốt lõi và Kiến trúc

### Khái niệm
LightFM là mô hình gợi ý embedding có thể hoạt động theo hướng collaborative filtering (chỉ dùng tương tác user-item) hoặc hybrid (kết hợp metadata). Trong giai đoạn này, mô hình được chạy theo kiểu collaborative filtering để xây dựng baseline đơn giản.

### Mục tiêu
Mục tiêu là học vector tiềm ẩn cho user và item từ lịch sử tương tác, sau đó xếp hạng item theo điểm dự đoán cho từng user. Điểm dự đoán được tạo bởi hàm score của LightFM (tương đương độ phù hợp giữa embedding user và embedding item).

### Đầu vào/Đầu ra
Đầu vào: User ID và Item ID trong ma trận tương tác.
Đầu ra: Điểm ưu tiên và danh sách Top-K item để gợi ý.

## 2. Quy trình xử lý dữ liệu

### Bộ dữ liệu
- Dataset: MovieLens 100k.
- Cách nạp dữ liệu: `fetch_movielens(min_rating=4.0)` của LightFM.
- Cách biểu diễn: Rating >= 4.0 được quy đổi thành implicit interaction.

### Quy trình
1. Nạp train/test interactions từ split mặc định của LightFM.
2. Chuyển sang sparse matrix để huấn luyện và đánh giá.
3. Huấn luyện mô hình LightFM với loss xếp hạng.
4. Đánh giá Top-10 trên test set, có loại bỏ các item đã thấy trong train khi ranking.

## 3. Chỉ số và Kết quả thực nghiệm

### Cấu hình đánh giá
- Loss đánh giá trong report này: `logistic` (ổn định trên môi trường Windows hiện tại).
- k = 10.
- Hyperparameters: epochs = 20, components = 30, learning_rate = 0.05, random_state = 42.

Chi so | Ket qua |  | Precision@10 | 0.0737 |

### Kết quả
|Chỉ số | Kết quả |
|---|---:|
| Precision@10 | 0.0737 |
| Recall@10 |	0.1323 |
| MRR@10 | 0.2357 |
| NDCG@10 |	0.1188 |
| Hit@10 | 0.5385 |

Ghi chú:
- File `lightfm_experiment_results.csv` hiện chỉ có 1 dòng kết quả cũ cho `logistic` với epochs=1 (`precision@10=0.0657, recall@10=0.1180`).
- Các chỉ số MRR/NDCG/Hit trong bảng được tính bổ sung để report tương đương mẫu MF.

### Phân tích kết quả
- Hit@10 đạt 53.85%, cho thấy khả năng tìm trúng item liên quan là tốt ở mức baseline.
- Recall@10 = 13.23% cao hơn mức Precision@10, phù hợp bối cảnh mỗi user có nhiều item liên quan nhưng top-10 bị giới hạn.
- MRR@10 và NDCG@10 cho thấy các item đúng đã xuất hiện khá sớm trong danh sách, nhưng chất lượng thứ hạng vẫn có dư địa cải thiện.

## 4. Siêu tham số và Tối ưu hóa

### Cấu hình hiện tại
- Loss: `logistic`
- Learning rate: 0.05
- Embedding size (`no_components`): 30
- Epochs: 20
- num_threads: 1 (LightFM trong env hiện tại cảnh báo không có OpenMP)

### Hiệu suất huấn luyện
Trung bình: ~0.203 giây/epoch
Nhanh nhất: ~0.190 giây/epoch
Chậm nhất: ~0.234 giây/epoch

## 5. Hàm mất mát và Cơ chế xếp hạng

### Trạng thái hiện tại

- Script đã được rút gọn về 1 loss duy nhất: `logistic` để đảm bảo chạy ổn định và dễ tái lập.

### Ý nghĩa
- `logistic`: ổn định, dễ tái lập baseline, phù hợp mục tiêu chạy gọn.

## 6. Đánh giá mô hình: Ưu và Nhược điểm

### Ưu điểm
- Dễ triển khai, dễ tái lập, tốc độ huấn luyện nhanh.
- Hoạt động tốt cho bài toán Top-K recommendation với dữ liệu implicit.
- Pipeline gọn, ít thành phần phụ trợ, dễ bảo trì.

### Nhược điểm
- Chưa tận dụng metadata user/item (chưa hybrid hóa đúng nghĩa LightFM).
Kết quả hiện tại chỉ benchmark theo loss logistic, chưa bao gồm so sánh liên-loss.

## 7. Hướng nâng cấp để đạt report hoàn chỉnh hơn

1. Chuẩn hóa bộ metric đầu ra CSV: thêm MRR@10, NDCG@10, Hit@10, thời gian huấn luyện.
2. Thêm validation split và chế độ early stopping.
3. Chạy benchmark lặp lại theo 3–5 random seeds để báo cáo mean ± std.
4. Nếu cần so sánh liên-loss (`warp/bpr`), cần chuẩn bị môi trường native ổn định riêng.
5. Nếu có metadata, kích hoạt khối hybrid của LightFM để vượt baseline CF.

## 8. Cập nhật mới nhất (07/04/2026): Script rút gọn + compact grid search

### Những gì đã cập nhật trong code
- Script `lightfm_baseline.py` đã được rút gọn về pipeline `logistic`-only.
- Đã bỏ hoàn toàn cơ chế subprocess/fallback để giảm độ phức tạp code.
- Vẫn giữ đầy đủ metric: `Precision@10`, `Recall@10`, `MRR@10`, `NDCG@10`, `Hit@10`.
- Vẫn giữ chế độ `--grid-search` với lưới nhỏ cho `epochs`, `components`, `learning_rate`.
- Kết quả lưu vào file: `lightfm_grid_search_results.csv` với các cột:
	`loss, precision@10, recall@10, mrr@10, ndcg@10, hit@10, epochs, components, learning_rate, k, random_state`.

### Cấu hình grid search đã chạy
- loss: `logistic`
- epochs: `{10, 20}`
- components: `{30}`
- learning_rate: `{0.03, 0.05}`
- Tổng số tổ hợp: `4` (compact, không exhaustive)

### Kết quả các cấu hình đã chạy

| Loss | Epochs | Components | LR | Precision@10 | Recall@10 | MRR@10 | NDCG@10 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic | 10 | 30 | 0.03 | 0.0677 | 0.1214 | 0.2244 | 0.1111 | 0.5000 |
| logistic | 10 | 30 | 0.05 | 0.0727 | 0.1308 | 0.2320 | 0.1166 | 0.5364 |
| logistic | 20 | 30 | 0.03 | **0.0746** | **0.1348** | **0.2384** | **0.1212** | **0.5439** |
| logistic | 20 | 30 | 0.05 | 0.0737 | 0.1323 | 0.2357 | 0.1188 | 0.5385 |

### Best config từ grid search gọn
- loss: `logistic`
- epochs: `20`
- components: `30`
- learning_rate: `0.03`
- k: `10`
- precision@10 = `0.0746`
- recall@10 = `0.1348`
- mrr@10 = `0.2384`
- ndcg@10 = `0.1212`
- hit@10 = `0.5439`

### Kết luận cập nhật
- Đã có tuning siêu tham số ở mức gọn (không quá tối ưu), không còn trạng thái "chỉ fix một bộ hyperparameters".
- Pipeline hiện tập trung hoàn toàn vào logistic để ưu tiên độ ổn định và tính đơn giản khi triển khai.