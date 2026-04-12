# Recommendation System Audit & Implementation (Phase 1)

Dự án này tập trung vào việc triển khai và kiểm định (Audit) các mô hình gợi ý nâng cao trên tập dữ liệu MovieLens 100k, xây dựng nền tảng vững chắc cho việc mở rộng sang các loại dữ liệu ngách (Thơ) ở Giai đoạn 2.

## 📁 Cấu trúc thư mục (Modular Architecture)

Dự án được tổ chức theo nguyên lý OOP và Modularity để đảm bảo tính mở rộng cao:

```text
/ (Root)
├── data/
│   ├── raw/           # Dữ liệu gốc (ml-100k, poetry)
│   └── processed/     # Dữ liệu đã xử lý (.inter và feature matrices)
├── configs/           # Quản lý cấu hình model qua file .yaml
├── models/            # Mã nguồn các mô hình (Inherited from BaseModel)
│   ├── base_model.py  # Abstract Base Class (Interface chung)
│   └── lightfm_model.py # Mô hình LightFM Hybrid (Custom PyTorch)
├── scripts/           # Script thực thi chính
│   └── train_eval.py  # Entry point cho huấn luyện và đánh giá
├── utils/             # Module bổ trợ (DataLoader, Metrics, PathManager)
├── logs/              # Nhật ký thực nghiệm và các tệp kết quả
└── docs/              # Hệ thống báo cáo phân tích chi tiết
```

## 🚀 Hướng dẫn thực thi

Để chạy huấn luyện và đánh giá mô hình, sử dụng script tập trung với file cấu hình tương ứng:

```powershell
# Thiết lập PYTHONPATH và chạy script
$env:PYTHONPATH = "."
python -m scripts.train_eval
```

Mô hình sẽ tự động load tham số từ `configs/lightfm_hybrid.yaml`, thực hiện huấn luyện 50 epochs và xuất kết quả Metrics (Recall@10, MRR, NDCG).

## 📊 Báo cáo & Tài liệu liên quan

Hệ thống báo cáo chi tiết được lưu trữ trong thư mục `/docs/`:

1.  [Báo cáo Tổng hợp Phase 1 (MF & LightFM)](docs/PHASE1_MF_COMPLETE_REPORT_FINAL.md): Phân tích chuyên sâu về Matrix Factorization, Popularity Bias và kết quả đối chứng 4 kịch bản LightFM.
2.  [Phân tích LightFM Hybrid](docs/LIGHTFM_FULL_ANALYSIS.md): Chi tiết về cơ chế cộng dồn Embedding và hiệu quả của WARP Loss.

## 🛠 Nguyên tắc phát triển (Clean Code)

- **Scalability**: Để thêm mô hình mới (ví dụ: DeepFM), chỉ cần kế thừa từ `BaseRecommendationModel` trong `models/base_model.py`.
- **Decoupling**: Logic xử lý dữ liệu, huấn luyện và đánh giá được vận hành bởi các class riêng biệt trong `utils/`.
- **Centralized Paths**: Toàn bộ đường dẫn được quản lý bởi `PathManager` để tránh tình trạng "hard-coded" đường dẫn.

---
**Senior ML Auditor & Architect**
*Ngày cập nhật: 12/04/2026*
