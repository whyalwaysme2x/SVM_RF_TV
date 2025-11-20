# TV Segment & Price Advisor

Ứng dụng web dự đoán phân khúc thị trường và giá TV sử dụng SVM và RandomForest.

## Cấu trúc thư mục

```
SVM_RF_TV/
├── frontend/          # Giao diện người dùng (HTML, CSS, JavaScript)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── backend/           # Backend FastAPI
│   ├── api.py        # API endpoints
│   ├── tv_pipeline.py # Model pipeline và xử lý dữ liệu
│   ├── main.py       # Entry point
│   └── requirements.txt
└── SVM_RF_TV/
    ├── Model/        # Model files (.pkl)
    │   ├── svm_best_model.pkl
    │   └── rf_best_model.pkl
    └── Data/         # Dataset
        └── cleaned_tv_dataset_fi.csv
```

## Cài đặt

1. Cài đặt dependencies:
```bash
cd backend
pip install -r requirements.txt
```

## Chạy ứng dụng

### Cách 1: Sử dụng script
```bash
cd backend
run.bat
```

### Cách 2: Chạy thủ công
```bash
cd backend
python main.py
```

Hoặc:
```bash
cd backend
uvicorn api:app --reload
```

Backend sẽ chạy tại: `http://localhost:8000`

Frontend sẽ được phục vụ tại: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

## Sử dụng

1. Mở trình duyệt và truy cập `http://localhost:8000`
2. Điền thông tin cấu hình TV vào form
3. Nhấn "Predict" để xem dự đoán phân khúc và giá

## Test Model

Để kiểm tra model và confidence scores:

```bash
cd backend
python check_model.py
```

Script này sẽ:
- Kiểm tra model nào đang được sử dụng (.joblib hay .pkl)
- Load model và kiểm tra xem có hỗ trợ `predict_proba` không
- Test với sample data
- Hiển thị confidence score và cảnh báo nếu confidence thấp

**Lưu ý quan trọng:**
- Model `.joblib` được ưu tiên load trước (model mới với `probability=True`, confidence cao ~88%)
- Model `.pkl` từ notebook có thể có confidence thấp do được train với `probability=False`
- **Nếu confidence thấp (< 50%), hãy restart server** để đảm bảo dùng model mới

## API Endpoints

- `GET /` - Frontend UI
- `GET /api/health` - Health check
- `GET /api/options` - Lấy danh sách options cho các trường
- `POST /api/predict` - Dự đoán phân khúc và giá

## Model

- **SVM (RBF kernel)**: Dự đoán phân khúc thị trường (Entry/Mid/Premium)
- **RandomForest**: Dự đoán giá đề xuất

Models được load từ `SVM_RF_TV/Model/`:
- `svm_best_model.pkl` - Model phân loại phân khúc
- `rf_best_model.pkl` - Model dự đoán giá

Nếu models không tồn tại, hệ thống sẽ tự động train từ dữ liệu trong `SVM_RF_TV/Data/cleaned_tv_dataset_fi.csv`.
