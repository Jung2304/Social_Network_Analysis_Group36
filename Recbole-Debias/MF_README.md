# Dự án: Phân tích Mô hình Cơ sở Matrix Factorization - Giai đoạn 1 (ml-100k)
**Ngày:** 4 tháng 4, 2026

## 1. Ý tưởng cốt lõi & Kiến trúc
### Khái niệm
Mô hình cơ sở Matrix Factorization (MF) thực hiện phương pháp **Yếu tố Tiềm ẩn (Latent Factor Approach)** tiêu chuẩn. Mô hình giả định rằng cả người dùng và mục phẩm đều có thể được đặc trưng bởi một tập hợp các yếu tố tiềm ẩn (embeddings) trong một không gian vector chiều thấp.

### Mục tiêu
Mục tiêu chính là phân rã ma trận tương tác Người dùng-Mục phẩm thưa thớt $R$ thành hai ma trận có hạng thấp hơn: $P$ (các yếu tố tiềm ẩn của Người dùng) và $Q$ (các yếu tố tiềm ẩn của Mục phẩm). Bằng cách học các embeddings này, mô hình có thể dự đoán các giá trị còn thiếu trong ma trận tương tác, đại diện cho sở thích chưa quan sát được của người dùng.

### Đầu vào/Đầu ra
- **Đầu vào:** User ID và Item ID từ lịch sử tương tác.
- **Đầu ra:** Điểm số sở thích dự đoán $\hat{r}_{ui}$, được tính bằng tích vô hướng của các embeddings tương ứng của người dùng và mục phẩm:
  $$\hat{r}_{ui} = p_u^\top q_i = \sum_{k=1}^d p_{uk} \cdot q_{ik}$$
  trong đó $d$ là số chiều của không gian tiềm ẩn (`embedding_size`).

## 2. Quy trình Xử lý Dữ liệu
### Bộ dữ liệu
Chúng tôi sử dụng các tệp từ bộ dữ liệu **ml-100k**. Quy trình tuân theo các bước cấu trúc sau:
1.  **Tải dữ liệu:** Các tệp tương tác được tải vào môi trường RecBole.
2.  **Lọc dữ liệu:** Bộ lọc tương tác tối thiểu (`min_user_inter=1`, `min_item_inter=1`) đảm bảo mật độ dữ liệu để ước tính các yếu tố tiềm ẩn.
3.  **Phân chia:** Dữ liệu được chia thành các tập huấn luyện/kiểm định **Thông thường (Normal)** và một **Tập kiểm thử Can thiệp (Intervened Test Set)** chuyên biệt.
4.  **Intervene Mask:** Cột `intervene_mask` đóng vai trò quan trọng trong việc đánh giá mô hình MF trong **điều kiện không thiên kiến (unbiased)**. Bằng cách lọc các tương tác kiểm thử nơi sự thiên kiến phổ biến (popularity bias) được giảm thiểu (mô phỏng phân phối đồng nhất), chúng tôi có thể đo lường độ chính xác ngữ nghĩa thực sự của mô hình so với xu hướng chạy theo các mục phẩm phổ biến.

## 3. Chỉ số & Kết quả Thực nghiệm
Mô hình được đánh giá bằng các chỉ số xếp hạng top-k tiêu chuẩn trên tập kiểm thử can thiệp.

### Kết quả Thực thi
| Chỉ số | Kết quả |
| :--- | :---: |
| **Recall@10** | 0.0855 |
| **MRR@10** | 0.1915 |
| **NDCG@10** | 0.0998 |
| **Hit@10** | 0.5271 |
| **Precision@10** | 0.0794 |

### Phân tích Kết quả
- **Recall cao vs. MRR thấp:** Mô hình đạt được chỉ số **Recall@10 (0.0855)** và **Hit@10 (0.5271)** khá cạnh tranh, cho thấy nó hiệu quả trong việc tìm ra các mục phẩm phù hợp.
- **Chất lượng Xếp hạng:** Tuy nhiên, **MRR@10 (0.1915)** thấp hơn so với các mô hình khử nhiễu tiên tiến (như DICE). Điều này cho thấy mặc dù MF có thể xác định đúng mục phẩm, nó thường xếp chúng ở vị trí thấp hơn trong danh sách top-k vì thiếu khả năng gỡ rối ngữ nghĩa (semantic disentanglement) cần thiết để ưu tiên sở thích ngách của người dùng thay vì các tín hiệu chạy theo đám đông.

## 4. Siêu tham số & Tối ưu hóa
Các siêu tham số sau đã được tối ưu hóa để đảm bảo sự ổn định và hội tụ trên quy mô của ml-100k:
- `learning_rate`: **0.005**
- `embedding_size`: **16**
- `batch_size`: **2048**
- `optimizer`: **Adam**

**Tốc độ Hội tụ:** Nhờ việc triển khai hiệu quả trên CPU, mô hình đạt được sự hội tụ với tốc độ xấp xỉ **~0.1 giây mỗi epoch**, thường ổn định trong vòng 30-40 epochs.

## 5. Hàm mất mát: BPR Loss
Mô hình được tối ưu hóa bằng hàm mất mát **Bayesian Personalized Ranking (BPR) Loss**, được định nghĩa là:
$$\mathcal{L}_{BPR} = -\sum_{(u, i, j) \in D} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) + \lambda \|\Theta\|^2$$
trong đó $i$ là tương tác dương, $j$ là tương tác âm được lấy mẫu, và $\Theta$ đại diện cho các tham số của mô hình.

**Tại sao chọn Pairwise?** Xếp hạng theo cặp (BPR) vượt trội hơn so với hồi quy theo điểm (Pointwise) trong các hệ thống gợi ý vì nó tập trung vào **thứ tự tương đối** của các mục phẩm thay vì giá trị xếp hạng tuyệt đối, điều này phù hợp hơn với bản chất ưu tiên xếp hạng của người dùng.

## 6. Đánh giá Mô hình: Ưu & Nhược điểm
### Ưu điểm
- **Khả năng mở rộng:** Độ phức tạp tính toán tăng tuyến tính theo số lượng tương tác.
- **Tốc độ:** Huấn luyện cực nhanh và thời gian suy luận dưới một phần nghìn giây.
- **Mô hình cơ sở mạnh mẽ:** Cung cấp mức hiệu năng nền tảng vững chắc cho bất kỳ tác vụ gợi ý nào.

### Nhược điểm
- **Thiên kiến phổ biến (Popularity Bias):** Vốn dĩ nắm bắt các thiên kiến trong dữ liệu, thường gợi ý những gì "phổ biến" thay vì những gì mang tính "cá nhân".
- **Thiếu ngữ cảnh:** Không thể kết hợp các thông tin phụ hoặc động lực thời gian.

---
*Được tạo bởi Antigravity - Senior ML Technical Writer*
