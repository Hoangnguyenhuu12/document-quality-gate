# AI Document Quality Gate

Hệ thống thẩm định chất lượng tài liệu và đánh giá mức độ sẵn sàng cho nhận diện (OCR Readiness) độc lập, chuyên dụng cho **hợp đồng kinh tế, tài liệu pháp lý và hồ sơ nội bộ doanh nghiệp**.

---

## 1. Định Vị & Triết Lý

- **Quality Gate là "Người gác cổng an ninh"**: Đứng trước hệ thống OCR và AI RAG để sàng lọc, đảm bảo tài liệu đầu vào đủ tiêu chuẩn nhận diện.
- **Mục tiêu cốt lõi**: Đánh giá chỉ số **OCR Readiness** và ra quyết định định tuyến 3 luồng:
  - **Good**: Tài liệu rõ nét, thẳng hàng $\rightarrow$ Đi thẳng vào OCR (`direct_to_ocr`).
  - **Medium**: Có khuyết tật nhẹ (nghiêng góc, bóng mờ, tương phản yếu) $\rightarrow$ Tự động tăng cường/khử khuyết tật rồi mới OCR (`enhance_before_ocr`).
  - **Bad**: Mờ nhòe nặng, chụp quá xa/thiếu pixel hoặc bị lóa flash $\rightarrow$ Từ chối và yêu cầu gửi lại bản scan rõ nét hơn (`reject_and_request_reupload`).

---

## 2. Các Chỉ Số Đo Lường Vật Lý

1. **Blur Detection (Độ Mờ Nhòe)**: Phương sai năng lượng tần số cao qua toán tử vi phân Laplacian 2D.
2. **Contrast Detection (Độ Tương Phản)**: Độ lệch chuẩn độ sáng (RMS contrast) giữa mực in/con dấu và nền giấy.
3. **Resolution Estimation (Độ Phân Giải)**: Ước lượng DPI hiệu dụng so với hình học văn bản khổ A4 tiêu chuẩn.
4. **Shadow & Illumination (Bóng Mờ & Ánh Sáng Cục Bộ)**: Đo độ lệch chuẩn ánh sáng giữa lưới $3 \times 3$ các vùng để phát hiện bóng tay, nếp gấp trang hợp đồng.
5. **Skew Detection (Độ Nghiêng Lệch Góc)**: Xác định góc nghiêng (độ) của văn bản scan để xoay thẳng lại trước khi OCR.
6. **Glare Detection (Chói Lóa Đèn Flash / Bìa Nhựa)**: Phát hiện vùng cháy sáng do chụp qua bao nylon, bìa ép nhựa hoặc đèn flash camera.

---

## 3. Cài Đặt

```bash
cd document-quality-gate
pip install -r requirements.txt
```

---

## 4. Sử Dụng Qua Dòng Lệnh (CLI)

```bash
# Thẩm định tệp PDF hợp đồng
python cli.py hop_dong.pdf

# Thẩm định tệp ảnh và xuất file JSON báo cáo
python cli.py ban_scan.png --output quality_result.json

# Tự động khắc phục ảnh scan (xoay thẳng, khử bóng, tăng tương phản)
python cli.py ban_scan.png --enhance ban_scan_sach_se.png

# Tự động khắc phục toàn bộ tệp PDF hợp đồng & xuất từng trang ảnh có thứ tự
python cli.py hop_dong.pdf --enhance hop_dong_sach_se.pdf

# Bỏ qua bộ nhớ Cache để ép kiểm tra lại từ đầu
python cli.py hop_dong.pdf --no-cache
```

---

## 5. Khởi Chạy REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- `POST /quality/assess`: Thẩm định chất lượng qua file tải lên (form-data) hoặc đường dẫn file.
- `POST /quality/assess-path`: Thẩm định qua JSON body `{ "file_path": "...", "use_cache": true }`.
- `POST /quality/enhance`: Tự động căn chỉnh và làm sạch ảnh tài liệu, trả về file đã xử lý.
- `GET /storage/report/{identifier}`: Tra cứu báo cáo JSON đã lưu trong hệ thống theo mã băm SHA-256.

---

## 6. Cấu Trúc Lưu Trữ (Storage Architecture)

Hệ thống tự động phân loại và lưu trữ theo cấu trúc chuẩn trong thư mục `storage/`:
- `storage/uploads/YYYY-MM-DD/`: Lưu trữ bản gốc các file nhận vào theo ngày.
- `storage/processed/`: Lưu trữ file PDF hoặc ảnh đã được tự động làm đẹp (`enhanced_<tên_file>.pdf`).
- `storage/processed/<tên_tài_liệu>/`: Lưu các trang ảnh rời có đánh số thứ tự chuẩn (`page_001.png`, `page_002.png`...).
- `storage/reports/`: Lưu báo cáo thẩm định dạng JSON (`<sha256>.json`).
- `storage/cache/`: Cơ sở dữ liệu SQLite (`assessments.db`) phục vụ caching tức thì (<1ms).

---

## 7. Sử Dụng Trong Python Code

```python
from src.pipeline.quality_pipeline import DocumentQualityPipeline

pipeline = DocumentQualityPipeline.from_config_file("configs/default.yaml")

# 1. Thẩm định chất lượng (tự động cache SHA-256 và lưu report)
assessment = pipeline.assess_document("path/to/hop_dong_kinh_te.pdf")

print("OCR Readiness:", assessment.ocr_readiness)
print("Thành phần tài liệu:", assessment.composition)
print("Các trang lỗi nặng:", assessment.defective_pages)
print("Các trang cần tăng cường:", assessment.pages_needing_enhancement)

# 2. Tự động phục chế và xuất file PDF hoàn chỉnh cùng các trang ảnh theo thứ tự
enhanced_pdf_path = pipeline.enhance_pdf("path/to/hop_dong_kinh_te.pdf", export_page_images=True)
print("File PDF đã làm sạch lưu tại:", enhanced_pdf_path)
```

---

## 8. Các Tính Năng Cao Cấp (Senior QA & Enterprise Ready)

1. **Nhận Diện Trang Trắng (Blank Page Detection - Duplex Scans)**:
   - Các hợp đồng scan 2 mặt thường có mặt sau trắng tinh. Nếu dùng thuật toán đo độ mờ thông thường (Laplacian variance = 0), hệ thống sẽ báo nhầm thành `severe_blur` và từ chối hợp đồng.
   - Module `BlankPageDetector` tự động nhận diện trang trắng bằng mật độ mực và độ đồng nhất, chấm điểm 5.0 tuyệt đối và phân loại `page_type="blank_page"`.

2. **Bảo Toàn Con Dấu Đỏ & Chữ Ký Xanh (Color Preservation)**:
   - Khi làm nét và khử bóng giấy trắng thông thường qua grayscale, con dấu đỏ và chữ ký mực xanh sẽ bị biến thành đen xám, làm mất giá trị pháp lý khi kiểm toán.
   - Thuật toán `ImageEnhancer.enhance_document_color` tính toán độ chênh lệch ánh sáng nền (illumination gain) trên không gian màu đa kênh, đẩy nền giấy ố vàng về trắng tinh (255) nhưng giữ nguyên vẹn sắc độ đỏ của con dấu và xanh của chữ ký.

3. **Thẩm Định Hợp Đồng Toàn Diện (Full-Document Evaluation)**:
   - Mặc định quét thẩm định toàn bộ tài liệu (không còn giới hạn 5 trang như các bản mẫu thử nghiệm), đảm bảo các trang chữ ký, phụ lục ở trang cuối cùng đều được rà soát nghiêm ngặt.

4. **Cơ Chế "Skim-Scan & Targeted Patch" Siêu Tốc**:
   - Quét lượt 1 (Skim Pass): Trích xuất cấu trúc văn bản số (Digital Text) và chỉ số chất lượng từng trang scan.
   - Lượt 2 (Targeted Patch): Tái sử dụng siêu dữ liệu góc nghiêng (`skew_angle`) và độ xoay (`coarse_rotation`) từ lượt 1, chỉ phục chế các trang bị lỗi mà không cần đo đạc lại, ghép nguyên vẹn vào file PDF hoàn chỉnh.

5. **Tự Động Phát Hiện Và Xoay Đứng Văn Bản (Coarse Orientation)**:
   - Tự động phát hiện các trang bị nhân viên scan ngang (xoay 90°/270°) và xoay thẳng về đứng (portrait) trước khi đưa vào OCR.

6. **Tự Động Huỷ Bộ Nhớ Đệm Khi Đổi Cấu Hình (Config-Hash Invalidation)**:
   - Mã băm Cache Key kết hợp cả SHA-256 nội dung tệp và SHA-256 chuỗi cấu hình YAML. Khi chuyên viên QA thay đổi ngưỡng chấp thuận trong `configs/default.yaml`, hệ thống tự động nhận biết và đánh giá lại thay vì trả về kết quả cũ.


