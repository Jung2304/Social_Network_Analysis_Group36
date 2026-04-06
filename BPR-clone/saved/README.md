Dưới đây là một **bản report hoàn chỉnh** dựa trên log kết quả bạn gửi. Mình viết theo kiểu có thể chỉnh sửa nhẹ rồi nộp luôn. Các thông số thực nghiệm, cấu hình mô hình, dữ liệu và kết quả đều được lấy trực tiếp từ file log bạn tải lên. 

---

# BÁO CÁO THỰC NGHIỆM MÔ HÌNH BPR TRONG HỆ GỢI Ý

## 1. Giới thiệu

Trong các hệ gợi ý hiện đại, một bài toán quan trọng là dự đoán và xếp hạng các mục (item) sao cho những mục phù hợp nhất được đề xuất cho người dùng. Với các hệ thống như gợi ý phim, nhạc, sản phẩm hay bài viết, dữ liệu thường có dạng **implicit feedback**, nghĩa là chỉ ghi nhận hành vi như xem, click, nghe hoặc mua, thay vì điểm đánh giá rõ ràng.

Trong báo cáo này, mô hình được sử dụng là **BPR (Bayesian Personalized Ranking)**, một mô hình gợi ý cơ bản nhưng rất quan trọng trong bài toán **personalized ranking**. Thí nghiệm được thực hiện bằng thư viện **RecBole**, với dataset mẫu **MovieLens 100K (ml-100k)**. Log chạy cho thấy đây là mô hình BPR thuộc nhóm **general recommender**, dùng đầu vào kiểu **pairwise**, đánh giá theo bài toán ranking, và được học bằng embedding cho user và item. 

## 2. Mục tiêu của mô hình

Khác với các mô hình dự đoán điểm số tuyệt đối, BPR tập trung vào **thứ hạng tương đối** giữa các item đối với từng user. Ý tưởng cốt lõi là:

* nếu user (u) đã tương tác với item (i),
* và chưa tương tác với item (j),

thì mô hình phải học sao cho:

[
\hat{y}*{ui} > \hat{y}*{uj}
]

Tức là item dương phải được xếp cao hơn item âm. Vì vậy, BPR rất phù hợp với hệ gợi ý top-N, nơi mục tiêu cuối cùng là đưa ra một danh sách item được sắp hạng thay vì một giá trị dự đoán đơn lẻ.

## 3. Tư duy và nguyên lý hoạt động của BPR

### 3.1. Biểu diễn người dùng và item bằng embedding

Trong BPR, mỗi người dùng và mỗi item được ánh xạ vào một vector đặc trưng trong không gian tiềm ẩn. Ở thí nghiệm này, kích thước embedding được đặt là **64**, và log mô hình cho thấy:

* `user_embedding: Embedding(944, 64)`
* `item_embedding: Embedding(1683, 64)` 

Điều đó có nghĩa là:

* có **944 user**
* có **1683 item**
* mỗi user/item được biểu diễn bởi một vector 64 chiều. 

Điểm phù hợp giữa user (u) và item (i) thường được tính bằng tích vô hướng:

[
\hat{y}_{ui} = p_u^T q_i
]

trong đó (p_u) là vector của user và (q_i) là vector của item.

### 3.2. Học theo cặp (pairwise learning)

BPR không học trên từng điểm dữ liệu đơn lẻ, mà học trên các bộ ba dạng:

[
(u, i, j)
]

với:

* (u): user
* (i): item dương, tức item user đã tương tác
* (j): item âm, tức item user chưa tương tác

Mục tiêu là tối đa hóa xác suất để (i) được xếp cao hơn (j). Trong log, cấu hình của RecBole xác nhận điều này qua:

* `MODEL_INPUT_TYPE = InputType.PAIRWISE`
* `loss = BPRLoss()` 

### 3.3. Hàm mất mát

Hàm mất mát đặc trưng của BPR là:

[
\mathcal{L}*{BPR} = - \sum \ln \sigma(\hat{y}*{ui} - \hat{y}_{uj}) + \lambda ||\Theta||^2
]

Trong đó:

* (\sigma) là hàm sigmoid
* (\hat{y}*{ui} - \hat{y}*{uj}) là độ chênh điểm giữa item dương và item âm
* (\lambda ||\Theta||^2) là regularization để giảm overfitting

Ý nghĩa trực giác là nếu mô hình xếp item dương cao hơn item âm với khoảng cách lớn, loss sẽ nhỏ; nếu xếp sai, loss sẽ lớn và mô hình bị phạt.

## 4. Cấu hình thực nghiệm

Theo file log, thí nghiệm được chạy bằng script `run_recbole.py`, với seed cố định là **2020**, trạng thái `reproducibility = True`, và checkpoint được lưu trong thư mục `saved`. 

### 4.1. Dữ liệu

Dataset sử dụng là **ml-100k**, nằm trong đường dẫn dataset mẫu của RecBole. Thống kê dữ liệu như sau:

* Số user: **944**
* Số item: **1683**
* Số tương tác: **100000**
* Mức sparsity: **93.7058%**
* Các trường dữ liệu được giữ lại: `user_id`, `item_id`, `rating`, `timestamp` 

### 4.2. Cấu hình chia tập dữ liệu

Dữ liệu được chia theo tỉ lệ:

* Train: **80%**
* Validation: **10%**
* Test: **10%**

thông qua cấu hình:

`eval_args = {'split': {'RS': [0.8, 0.1, 0.1]}, ... }` 

### 4.3. Siêu tham số huấn luyện

Các siêu tham số chính:

* `epochs = 300`
* `train_batch_size = 2048`
* `learner = adam`
* `learning_rate = 0.001`
* `weight_decay = 0.0`
* `eval_step = 1`
* `stopping_step = 10`
* `embedding_size = 64` 

Ngoài ra, negative sampling trên tập train được cấu hình là uniform với `sample_num = 1`, nghĩa là với mỗi mẫu dương sẽ sinh một mẫu âm ngẫu nhiên. 

### 4.4. Thiết lập đánh giá

Mô hình được đánh giá theo top-10 với các metric:

* Recall@10
* MRR@10
* NDCG@10
* Hit@10
* Precision@10

Metric được dùng để chọn mô hình tốt nhất là:

* `valid_metric = MRR@10` 

Điều đó nghĩa là checkpoint tốt nhất sẽ được chọn dựa trên giá trị MRR@10 trên tập validation.

## 5. Kiến trúc mô hình

Log cho biết mô hình BPR được khởi tạo như sau:

* User embedding: `(944, 64)`
* Item embedding: `(1683, 64)`
* Loss: `BPRLoss()`
* Số tham số có thể huấn luyện: **168128** 

Đây là một mô hình gọn, đơn giản, ít tham số hơn nhiều so với các mô hình sâu phức tạp. Nhờ đó, thời gian train mỗi epoch tương đối ngắn.

## 6. Quá trình huấn luyện

Quá trình train cho thấy loss giảm đều và điểm validation tăng dần qua nhiều epoch đầu. Ở các epoch đầu tiên:

* epoch 0: `train loss = 27.7240`, `MRR@10 = 0.0257`
* epoch 10: `train loss = 12.4353`, `MRR@10 = 0.3095`
* epoch 20: `train loss = 9.1907`, `MRR@10 = 0.3477`
* epoch 50: `train loss = 5.8605`, `MRR@10 = 0.3893`

Nhìn chung, loss giảm mạnh từ khoảng **27.72** xuống còn khoảng **5.86** ở epoch tốt nhất, cho thấy mô hình học được quan hệ ưu tiên giữa item dương và item âm. Đồng thời, các metric validation tăng đáng kể, chứng tỏ năng lực ranking của mô hình được cải thiện rõ rệt.

Mặc dù cấu hình đặt `epochs = 300`, quá trình huấn luyện đã dừng sớm ở **epoch 61**, và RecBole ghi nhận:

`Finished training, best eval result in epoch 50` 

Điều này phù hợp với `stopping_step = 10`, nghĩa là sau một số epoch không còn cải thiện theo metric validation, hệ thống sẽ dừng để tránh train dư thừa. 

## 7. Kết quả thực nghiệm

### 7.1. Kết quả tốt nhất trên validation

Checkpoint tốt nhất đạt được tại **epoch 50**, với kết quả validation:

* Recall@10 = **0.2085**
* MRR@10 = **0.3893**
* NDCG@10 = **0.2302**
* Hit@10 = **0.7402**
* Precision@10 = **0.1583** 

### 7.2. Kết quả cuối cùng trên test

Sau khi nạp lại checkpoint tốt nhất, mô hình được đánh giá trên tập test và cho kết quả:

* Recall@10 = **0.2466**
* MRR@10 = **0.4895**
* NDCG@10 = **0.2928**
* Hit@10 = **0.7815**
* Precision@10 = **0.1962** 

Đây là kết quả quan trọng nhất của thí nghiệm, vì nó phản ánh hiệu năng cuối cùng của mô hình trên dữ liệu chưa thấy trong quá trình huấn luyện.

## 8. Phân tích kết quả

### 8.1. Ý nghĩa của các metric

**Recall@10 = 0.2466** cho thấy trong top 10 item được đề xuất, mô hình thu hồi được khoảng 24.66% các item liên quan. Đây là một mức khá tốt đối với một baseline pairwise đơn giản. 

**Precision@10 = 0.1962** nghĩa là trong 10 item được gợi ý, khoảng 19.62% là item đúng hoặc phù hợp. Chỉ số này phản ánh mức chính xác của danh sách đề xuất. 

**Hit@10 = 0.7815** là kết quả nổi bật, cho thấy trong khoảng 78.15% trường hợp, top 10 đề xuất có chứa ít nhất một item đúng. Điều này chứng minh mô hình có khả năng đưa item phù hợp vào danh sách đề xuất với tần suất cao. 

**MRR@10 = 0.4895** thể hiện rằng item đúng đầu tiên thường xuất hiện ở vị trí khá cao trong danh sách. Đây là một dấu hiệu rất tốt về chất lượng xếp hạng đầu danh sách. Vì MRR@10 cũng là metric chọn mô hình trên validation, kết quả test cao cho thấy mô hình tổng quát hóa tốt.

**NDCG@10 = 0.2928** cho thấy mô hình không chỉ tìm được item đúng mà còn có xu hướng đặt chúng ở vị trí cao hơn trong top-10, tức là chất lượng sắp hạng là tương đối tốt. 

### 8.2. Xu hướng hội tụ

Từ epoch 0 đến khoảng epoch 50, validation score tăng liên tục từ **0.0257** lên **0.3893**, trong khi train loss giảm mạnh. Sau epoch 50, metric validation không còn cải thiện rõ rệt nữa, dù một số metric riêng lẻ như Recall hoặc Hit có dao động tăng nhẹ ở các epoch sau. Vì metric chọn mô hình là MRR@10, RecBole giữ checkpoint tại epoch 50 là hợp lý.

Điều này cho thấy mô hình đã hội tụ tốt và không cần dùng hết 300 epoch. Early stopping giúp tiết kiệm thời gian train và giảm nguy cơ overfitting.

### 8.3. Đánh giá tổng quan

Kết quả test cho thấy BPR hoạt động ổn định trên bộ dữ liệu MovieLens 100K. Đây là một mô hình đơn giản nhưng hiệu quả, đặc biệt phù hợp với dữ liệu implicit và bài toán ranking. Với chỉ **168128 tham số**, mô hình đã đạt Hit@10 gần **0.78** và MRR@10 gần **0.49**, là mức hiệu năng đáng ghi nhận cho một baseline nền tảng.

## 9. Ưu điểm và hạn chế của mô hình

### 9.1. Ưu điểm

BPR có một số ưu điểm nổi bật:

Thứ nhất, mô hình tối ưu trực tiếp cho bài toán xếp hạng, nên phù hợp với mục tiêu thực tế của hệ gợi ý.

Thứ hai, BPR hoạt động tốt với dữ liệu implicit, không cần rating rõ ràng.

Thứ ba, cấu trúc mô hình đơn giản, số tham số ít, train nhanh. Log thực nghiệm cho thấy mỗi epoch chỉ mất từ khoảng 0.3 đến 0.8 giây trong phần lớn quá trình chạy.

Thứ tư, mô hình dễ cài đặt, dễ phân tích và là baseline tốt để so sánh với các mô hình phức tạp hơn.

### 9.2. Hạn chế

Tuy nhiên, BPR cũng có các hạn chế:

Thứ nhất, mô hình giả định rằng mọi item chưa tương tác đều là âm tương đối, trong khi trên thực tế nhiều item có thể chỉ là “chưa được thấy”.

Thứ hai, chất lượng phụ thuộc vào negative sampling. Nếu lấy mẫu âm quá dễ hoặc không đại diện, mô hình có thể học chưa tối ưu.

Thứ ba, BPR dùng biểu diễn tuyến tính qua embedding và dot product, nên khó nắm bắt các quan hệ ngữ cảnh phức tạp như thời gian, nội dung văn bản, đặc trưng ảnh hoặc tín hiệu đồ thị.

Thứ tư, mô hình gặp khó khăn với cold-start cho user mới hoặc item mới vì dựa nhiều vào lịch sử tương tác.

## 10. Kết luận

Thí nghiệm với mô hình BPR trên dataset ml-100k trong RecBole đã được thực hiện thành công. Cấu hình chạy sử dụng embedding 64 chiều, học bằng Adam với learning rate 0.001, batch size 2048, đánh giá top-10 và chọn mô hình theo MRR@10. Dữ liệu gồm 944 user, 1683 item và 100000 tương tác, với độ thưa rất cao, khoảng 93.71%. 

Quá trình huấn luyện cho thấy loss giảm đều và metric validation tăng ổn định. Mô hình tốt nhất xuất hiện ở epoch 50. Trên tập test, kết quả đạt được là Recall@10 = 0.2466, MRR@10 = 0.4895, NDCG@10 = 0.2928, Hit@10 = 0.7815 và Precision@10 = 0.1962. 

Nhìn chung, BPR là một mô hình nền tảng hiệu quả cho bài toán recommendation dựa trên ranking. Kết quả thực nghiệm cho thấy mô hình có khả năng đưa item phù hợp vào top-10 với tần suất cao, đồng thời xếp item đúng ở vị trí khá sớm trong danh sách đề xuất. Đây là một baseline rất phù hợp cho các nghiên cứu tiếp theo, đặc biệt khi muốn so sánh với các mô hình nâng cao hơn như NeuMF, LightGCN hoặc các mô hình tuần tự.