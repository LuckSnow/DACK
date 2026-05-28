#  Vietnamese Traffic Sign Detection

Chương trình nhận diện biển báo giao thông Việt Nam sử dụng YOLOv8.

---

##  Cài đặt

### Linux / macOS
```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Cài đặt dependencies (nếu chưa)
pip install -r requirements.txt
```

### Windows
```cmd
# Kích hoạt virtual environment
venv\Scripts\activate

# Cài đặt dependencies (nếu chưa)
pip install -r requirements.txt
```

---

##  Nhận diện Ảnh

### Nhận diện một ảnh
```bash
python Detection/image.py --image_path path/to/image.jpg
```

### Nhận diện toàn bộ folder
```bash
python Detection/image.py --folder path/to/folder
```

### Tùy chọn
- `--conf 0.25` - Ngưỡng tự tin (0-1, mặc định 0.25)
- `--imgsz 1280` - Kích thước xử lý (640, 832, 1024, 1280)
  - `1280` tốt cho ảnh lớn/biển báo nhỏ
  - `640` nhanh hơn, phù hợp ảnh bình thường

### Ví dụ
```bash
# Ảnh đơn lẻ với ngưỡng cao
python Detection/image.py --image_path test.jpg --conf 0.3

# Folder ảnh với kích thước cao
python Detection/image.py --folder bienbao --imgsz 1280 --conf 0.25
```

**Output:** Ảnh kết quả có bounding box lưu trong `output/`

---

##  Nhận diện Video

```bash
python Detection/inference.py --model model02/best28121.pt --source video.mp4 --imgsz 1280 --conf 0.2
```

### Tùy chọn
- `--source` - Đường dẫn video
- `--imgsz` - Kích thước xử lý (mặc định 1280)
- `--conf` - Ngưỡng tự tin (mặc định 0.2)

### Ví dụ
```bash
# Nhận diện video với cấu hình tối ưu
python Detection/inference.py --model model02/best28121.pt --source video.mp4 --imgsz 1280 --conf 0.25

# Nhận diện nhanh
python Detection/inference.py --model model02/best28121.pt --source video.mp4 --imgsz 640 --conf 0.3
```

---

##  Kết quả

- **Ảnh:** Lưu trong `output/` với bounding box, tên biển báo, độ tự tin
- **Video:** In trực tiếp lên video hoặc lưu video đầu ra

---

## Danh mục biển báo (29 loại)

Các nhóm biển báo được phát hiện:
- **Cấm**: Cấm đỗ, cấm quay đầu, cấm rẽ, cấm xe máy/ô tô...
- **Cảnh báo**: Nguy hiểm, đi chậm...
- **Giới hạn**: Tốc độ, trọng tải, chiều cao...
- **Chỉ dẫn**: Phương hướng, làn xe được phép...
- **Khác**: Các biển báo khác

---

**Mô hình:** YOLOv8 (model02/best28121.pt)
