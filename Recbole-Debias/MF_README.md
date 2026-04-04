# Dự án: Phân tích Mô hình Cơ sở Matrix Factorization - Giai đoạn 1 (ml-100k)
**Ngày:** 4 tháng 4, 2026

## 1. Ý tưởng cốt lõi & Kiến trúc
### Khái niệm
Mô hình cơ sở Matrix Factorization (MF) thực hiện phương pháp **Yếu tố Tiềm ẩn (Latent Factor Approach)** tiêu chuẩn. Mô hình giả định rằng cả người dùng và sản phẩm đều có thể được đại diện bằng các đặc trưng ẩn (embeddings) trong một không gian toán học nhỏ gọn.

### Mục tiêu
Mục tiêu chính là phân tích lịch sử tương tác giữa Người dùng và Sản phẩm để tìm ra các mối liên hệ ngầm. Bằng cách học các đặc trưng này, mô hình có thể dự đoán mức độ yêu thích của người dùng đối với các sản phẩm mà họ chưa từng xem qua.

### Đầu vào/Đầu ra
- **Đầu vào:** Mã định danh Người dùng (User ID) và Mã định danh Sản phẩm (Item ID).
- **Đầu ra:** Điểm số dự đoán mức độ ưu tiên. Điểm số này được tính bằng cách so khớp (tình tích vô hướng) giữa vector đặc trưng của người dùng và vector đặc trưng của sản phẩm. Nếu hai vector này "cùng hướng" hoặc tương đồng, điểm số sẽ cao, nghĩa là người dùng có khả năng thích sản phẩm đó.

## 2. Quy trình Xử lý Dữ liệu
### Bộ dữ liệu
Chúng tôi sử dụng bộ dữ liệu **ml-100k**. Quy trình tuân theo các bước sau:
1.  **Tải dữ liệu:** Đưa các tệp tương tác vào hệ thống huấn luyện.
2.  **Lọc dữ liệu:** Chỉ giữ lại những người dùng và sản phẩm có ít nhất một tương tác để đảm bảo dữ liệu đủ chất lượng để học.
3.  **Phân chia:** Chia dữ liệu thành các tập huấn luyện (để học), tập kiểm định (để tinh chỉnh) và một **Tập kiểm thử Can thiệp** đặc biệt.
4.  **Đánh giá không thiên kiến:** Chúng tôi sử dụng một cơ chế lọc đặc biệt (intervene mask) để kiểm tra xem mô hình thực sự hiểu sở hữu của người dùng hay chỉ đơn giản là gợi ý các sản phẩm đang "hot". Điều này giúp đo lường độ chính xác thực tế khi loại bỏ yếu tố chạy theo đám đông.

## 3. Chỉ số & Kết quả Thực nghiệm
Mô hình được đánh giá dựa trên khả năng đưa ra danh sách gợi ý tốt nhất cho người dùng.

### Kết quả Thực thi
| Chỉ số | Kết quả |
| :--- | :---: |
| **Recall@10** (Khả năng tìm đúng) | 0.0855 |
| **MRR@10** (Độ chính xác thứ hạng) | 0.1915 |
| **NDCG@10** (Độ tối ưu danh sách) | 0.0998 |
| **Hit@10** (Tỷ lệ gợi ý trúng) | 0.5271 |

### Phân tích Kết quả
- **Gợi ý trúng tốt nhưng thứ hạng chưa cao:** Mô hình MF cơ bản có khả năng tìm ra các sản phẩm người dùng thích khá tốt (Hit@10 đạt hơn 52%).
- **Hạn chế:** Tuy nhiên, thứ hạng của các sản phẩm đúng trong danh sách thường không đứng ở vị trí đầu tiên (MRR thấp hơn các mô hình tiên tiến). Điều này là do MF dễ bị ảnh hưởng bởi những sản phẩm phổ biến, dẫn đến việc ưu tiên các sản phẩm "xu hướng" thay vì các sản phẩm thực sự phù hợp với sở thích cá nhân cụ thể của người dùng.

## 4. Siêu tham số & Tối ưu hóa
Các thông số cài đặt giúp mô hình hoạt động ổn định:
- **Tốc độ học (Learning Rate):** 0.005 (giúp mô hình học từ từ và ổn định).
- **Độ lớn đặc trưng (Embedding Size):** 16 (số lượng đặc trưng dùng để mô tả người dùng/sản phẩm).
- **Kích thước lô (Batch Size):** 2048 (số lượng dữ liệu xử lý cùng lúc).

**Hiệu suất:** Trên máy tính thông thường, mô hình học rất nhanh, chỉ mất khoảng **0.1 giây** cho mỗi vòng lặp huấn luyện.

## 5. Hàm mất mát: BPR Loss
Chúng tôi sử dụng phương pháp **Xếp hạng cá nhân hóa (BPR)**. 

Thay vì cố gắng dự đoán chính xác số điểm người dùng sẽ chấm, mô hình tập trung vào việc học cách **so sánh**: Làm sao để sản phẩm mà người dùng đã mua/xem phải có điểm số cao hơn những sản phẩm mà họ bỏ qua. Phương pháp này hiệu quả hơn vì trong thực tế, thứ tự ưu tiên quan trọng hơn điểm số tuyệt đối.

## 6. Đánh giá Mô hình: Ưu & Nhược điểm
### Ưu điểm
- **Dễ triển khai:** Hoạt động tốt ngay cả với lượng dữ liệu lớn.
- **Tốc độ cực nhanh:** Phù hợp cho các ứng dụng cần phản hồi tức thì.
- **Hiệu quả nền tảng:** Luôn là tiêu chuẩn để so sánh với các mô hình phức tạp hơn.

### Nhược điểm
- **Chạy theo đám đông:** Dễ bị đánh lừa bởi các sản phẩm phổ biến.
- **Thiếu linh hoạt:** Không hiểu được các yếu tố ngữ cảnh (như thời gian, vị trí).

---
*Dịch và hiệu chỉnh bởi Antigravity*
