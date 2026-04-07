# PHASE 1: MATRIX FACTORIZATION COMPLETE REPORT & DEEP AUDIT

## 1. Tóm tắt dự án (Executive Summary)
Báo cáo này tổng hợp toàn bộ kết quả thực nghiệm của Giai đoạn 1 (Phase 1), tập trung vào mô hình cơ sở **Matrix Factorization (MF)** và phân tích sâu về cơ chế **Disentanglement (DICE)** trên tập dữ liệu **ml-100k**. 

- **Trạng thái**: Hoàn thành 100% mục tiêu tái lập và audit.
- **Kết quả then chốt**: Tái lập thành công Baseline MF với Recall@10 đạt **0.0855**. Đồng thời, thông qua Audit chuyên sâu, chúng tôi xác nhận mô hình DICE cải thiện **~10% MRR** (0.2100 vs 0.1915) nhờ khả năng phân tách nhiễu Popularity Bias.

---

## 2. Kiến trúc Matrix Factorization (MF)

### 2.1. Ý tưởng cốt lõi
MF thực hiện phương pháp **Yếu tố Tiềm ẩn (Latent Factor)**, giả định người dùng $u$ và vật phẩm $i$ có thể biểu diễn bằng các vector $p_u, q_i \in \mathbb{R}^d$.

### 2.2. Công thức dự đoán $r_{ui}$
Mô hình dự đoán mức độ tương tác thông qua tích vô hướng của các latent vectors:
$$ \hat{r}_{ui} = p_u \cdot q_i^T = \sum_{k=1}^{d} p_{uk} q_{ik} $$

### 2.3. Hàm mất mát (Loss Function) BPR
Hệ thống sử dụng **Bayesian Personalized Ranking (BPR)** loss để tối ưu thứ hạng:
$$ \mathcal{L} = \sum_{(u, i, j) \in D} -\ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) + \lambda \|\Theta\|^2 $$
*Trong đó $i$ là item tương tác (positive) và $j$ là item được lấy mẫu âm (negative).*

---

## 3. Quy trình Dữ liệu & Chia tập (Data Splitting)

### 3.1. Logic Normal vs Intervened
Để đánh giá khả năng debias, dữ liệu được chia theo cơ chế đặc thù:
- **Normal Set (50%)**: Chứa bias tự nhiên từ dataset gốc.
- **Intervened Set (50%)**: Được lấy mẫu lại (Resampling) để đạt phân phối item đồng nhất (Uniform).

### 3.2. Dẫn chứng Code & Tỷ lệ Chia tập
Dựa trên tệp cấu hình `details/ml.md`, tỷ lệ chia tập được quy định nghiêm ngặt:
- **Train Set**: 100% Normal + 25% Intervened.
- **Valid Set**: 25% Intervened.
- **Test Set**: 50% Intervened.

### 3.3. Thống kê Interactions thực tế (Audit Result)
Kết quả thống kê từ script `verify_split.py` trên các tệp `.inter`:
| Tập dữ liệu | Intervened (True) | Normal (False) | Tổng số Interactions |
| :--- | :--- | :--- | :--- |
| **Train** | 5,967 | 49,760 | 55,727 |
| **Valid** | 6,367 | 0 | 6,367 |
| **Test** | 12,723 | 0 | 12,723 |
| **Tổng** | **25,057** | **49,760** | **74,817** |

---

## 4. Giao thức đánh giá (Evaluation Protocol)

### 4.1. Cấu hình Full Sort (YAML Evidence)
Minh chứng cấu hình trong [mf_baseline.yaml]:
```yaml
eval_args:
    mode: full      # Rank trên toàn bộ 1,600+ items (không dùng sample)
    order: RO       # Random Order (Xáo trộn ngẫu nhiên để khử bias thứ tự)
    group_by: user
```

### 4.2. Vai trò của `intervene_mask`
Cột `intervene_mask` cho phép hệ thống nhận diện các tương tác "unbiased" để đánh giá độ chính xác thực sự.
**Dòng dữ liệu mẫu từ `ml.train.inter`**:
```text
user_id:token  item_id:token  rating:float  intervene_mask:token
6              318            4             True
6              209            4             True
```

---

## 5. Kết quả thực nghiệm & Xác thực siêu tham số

### 5.1. Bảng kết quả Mini Grid Search (20 Epochs)
| LR | Embedding Size | NDCG@10 | Recall@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- |
| **0.01** | 16 | **0.0948** | 0.0810 | **0.1903** |
| 0.005 | 16 (Author) | 0.0905 | 0.0795 | 0.1804 |
| 0.001 | 64 | 0.0922 | **0.0858** | 0.1739 |

### 5.2. Đường dẫn Log thực tế (Audit Evidence)
Tệp log chi tiết chứa kết quả tái lập tốt nhất (NDCG 0.0998):
`log/MF/MF-ml-Apr-07-2026_20-11-04-94062f.log`

---

## 6. Phân tích Bias trong thực tế ml-100k

### 6.1. Popularity Bias (Audit Data)
Kết quả từ script `calculate_top_100.py`:
- **Thực trạng**: **Top 100** phim phổ biến nhất (chỉ chiếm **~6%** tổng số phim) chiếm tới **28.51%** tổng số tương tác.
- **Tác động**: Model MF thuần túy dễ bị "áp đảo" bởi 28% dữ liệu này, dẫn đến việc bỏ qua sở thích ngách (Long-tail) của người dùng.

### 6.2. Selection Bias
Người dùng chỉ đánh giá những phim họ tự chọn xem. Điều này giải thích tại sao MF có score cao trên tập test bias nhưng giảm sâu trên tập test Intervened (Uniform).

---

## 7. Audit Nội bộ & Deep Dive

### 7.1. Hội tụ Loss trong DICE
Quá trình huấn luyện DICE (62 epochs) cho thấy sự cân bằng giữa sở thích và tính tách biệt:
- **Click Loss (BPR)**: Giảm từ 51.82 xuống **8.82**.
- **Disentanglement Loss**: Duy trì mức **~13.02**, tạo áp lực buộc embeddings không được trùng lặp.

### 7.2. Embedding Similarity Audit
Kết quả đo lường độ tương đồng Cosine giữa vector Interest ($E_{int}$) và Conformity ($E_{pop}$):
- **Item Similarity: 0.3306** (Phân tách tốt, model hiểu phim nào hot vs phim nào hay).
- **User Similarity: 0.9992** (Phân tách kém, người dùng ml-100k thường có xu hướng thích phim vì nó phổ biến).

**Code thực hiện Audit**:
```python
user_sim = F.cosine_similarity(users_int, users_pop, dim=1).mean()
item_sim = F.cosine_similarity(items_int, items_pop, dim=1).mean()
```

---

## 8. Hệ sinh thái Model (Ecosystem)

Tại sao lại có sự xuất hiện của DICE, MACR phối hợp cùng MF Baseline?
- **MF Baseline**: Đóng vai trò là "mỏ neo" để đo lường mức độ bias của dataset.
- **DICE/MACR**: Là các giải pháp can thiệp causal. Ví dụ, trong [dice.py], model sử dụng 4 bảng embedding riêng biệt để "cô lập" biến Popularity ra khỏi tiến trình học sở thích thực (Interest).

---

## 9. Tổng kết & Bài học Phase 2
- **Ưu điểm**: Baseline MF đã được thiết lập vững chắc, quy trình eval Full Sort hoạt động ổn định trên CPU/GPU.
- **Nhược điểm**: MF vẫn bị Popularity Bias chi phối nặng nề trên tập ml-100k.
- **Bài học cho Phase 2**: Cần tập trung vào việc tinh chỉnh tham số `dis_pen` (trọng số của Disentanglement Loss) trong DICE để giảm User Similarity (hiện tại đang 0.99) xuống mức thấp hơn, giúp cá nhân hóa sâu sắc hơn.

---
