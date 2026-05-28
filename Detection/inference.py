"""
🚦 Vietnamese Traffic Sign Detection - Video Inference Only
===========================================================
Tối ưu hóa: Chỉ nhận diện biển báo giao thông từ file Video.
Hỗ trợ: Tự động cấu hình kích thước nhận diện (imgsz) và ngưỡng tự tin (conf) để bắt biển báo từ xa.

Usage:
    python inference_video.py --model Models/1.pt --source video.mp4 --imgsz 1280 --conf 0.2
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# Danh mục 29 nhóm biển báo giao thông - Model02
# ============================================================
VIE_NAMES = {
    "one way prohibition":             "Cấm đi ngược chiều",
    "no parking":                      "Cấm đỗ xe",
    "no stopping and parking":         "Cấm dừng và đỗ xe",
    "no turn left":                    "Cấm rẽ trái",
    "no turn right":                   "Cấm rẽ phải",
    "no u turn":                       "Cấm quay đầu",
    "no u and left turn":              "Cấm quay đầu và rẽ trái",
    "no u and right turn":             "Cấm quay đầu và rẽ phải",
    "no motorbike entry/turning":      "Cấm xe máy",
    "no car entry/turning":            "Cấm ô tô",
    "no truck entry/turning":          "Cấm xe tải",
    "other prohibition":               "Biển cấm khác",
    "indication":                      "Biển chỉ dẫn",
    "direction":                       "Biển phương hướng",
    "speed limit":                     "Giới hạn tốc độ",
    "weight limit":                    "Giới hạn trọng tải",
    "height limit":                    "Giới hạn chiều cao",
    "pedestrian crossing":             "Đường người đi bộ cắt ngang",
    "intersection danger":             "Nguy hiểm giao lộ",
    "road danger":                     "Nguy hiểm đường bộ",
    "pedestrian danger":               "Nguy hiểm người đi bộ",
    "construction danger":             "Công trình xây dựng",
    "slow warning":                    "Đi chậm",
    "other warning":                   "Cảnh báo khác",
    "vehicle permission lane":         "Làn xe được phép",
    "vehicle and speed permission lane": "Làn xe và tốc độ cho phép",
    "overpass route":                  "Tuyến đường vượt",
    "no more prohibition":             "Hết hiệu lực cấm",
    "other":                           "Biển báo khác",
}

# ============================================================
# Hàm xác định màu BGR/RGB theo nhóm biển báo (Model02)
# ============================================================
def get_class_color_bgr(class_name: str) -> tuple:
    """Trả về màu BGR dựa theo nhóm biển báo."""
    name = class_name.lower()
    if "no more prohibition" in name:
        return (0, 180, 0)       # Xanh lá - Hết cấm
    if any(k in name for k in ["prohibition", "no parking", "no stopping",
                                "no turn", "no u", "no motorbike",
                                "no car", "no truck", "one way"]):
        return (0, 0, 220)       # Đỏ - Biển cấm
    if any(k in name for k in ["limit"]):
        return (0, 100, 255)     # Cam đậm - Giới hạn
    if any(k in name for k in ["danger", "warning", "slow", "crossing"]):
        return (0, 165, 255)     # Cam - Cảnh báo
    if any(k in name for k in ["indication", "direction", "permission", "overpass"]):
        return (200, 100, 0)     # Xanh dương - Chỉ dẫn/hiệu lệnh
    return (128, 128, 128)       # Xám - Khác

def get_class_color_rgb(class_name: str) -> tuple:
    """Trả về màu RGB (dùng cho PIL)."""
    b, g, r = get_class_color_bgr(class_name)
    return (r, g, b)

FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:\\Windows\\Fonts\\tahoma.ttf",
    "C:\\Windows\\Fonts\\arial.ttf"
]

_font_cache = {}

def get_pil_font(size=16):
    """Lấy font hỗ trợ tiếng Việt Unicode (có cache)."""
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except:
                continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font

def enhance_image(frame, brightness=1.0, contrast=1.2):
    """Nâng cao chất lượng ảnh: tăng contrast và brightness để nhận diện biển báo nhỏ/mờ."""
    # Chuyển sang LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Tăng contrast cho kênh L (Luminance)
    l = cv2.convertScaleAbs(l, alpha=contrast, beta=brightness * 10)
    l = np.clip(l, 0, 255).astype(np.uint8)
    
    # Gộp lại
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def apply_clahe(frame, clip_limit=2.0, tile_size=(8, 8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) - nâng cao contrast địa phương."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray)
    
    # Chuyển lại BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr

def sharpen_image(frame, kernel_strength=1.5):
    """Làm sắc nét ảnh để các chi tiết biển báo rõ hơn."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) / kernel_strength
    sharpened = cv2.filter2D(frame, -1, kernel)
    return np.uint8(np.clip(sharpened, 0, 255))

def draw_detections(frame, results, model_classes, scale_x=1.0, scale_y=1.0):
    """Vẽ bounding box và nhãn tiếng Việt lên khung hình."""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return frame

    annotated = frame.copy()
    boxes_info = []
    summary_list = []

    # Phase 1: Vẽ khung bằng OpenCV
    for box in results[0].boxes:
        ox1, oy1, ox2, oy2 = box.xyxy[0].tolist()
        x1, y1 = int(ox1 * scale_x), int(oy1 * scale_y)
        x2, y2 = int(ox2 * scale_x), int(oy2 * scale_y)

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        code = str(model_classes[cls_id]) if cls_id < len(model_classes) else f"class_{cls_id}"
        
        color_bgr = get_class_color_bgr(code)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        
        boxes_info.append((x1, y1, x2, y2, conf, cls_id, code))
        summary_list.append(f"{code}({conf:.0%})")

    # Phase 2: Vẽ chữ bằng PIL (Sửa triệt để lỗi "str | None" cho Pylance)
    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_label = get_pil_font(15)
    font_vie = get_pil_font(13)

    for x1, y1, x2, y2, conf, cls_id, code in boxes_info:
        color_rgb = get_class_color_rgb(code)
        label = f"{code} {conf:.0%}"
        
        # Đảm bảo trả về chuỗi str thuần túy, không có khả năng bị None
        vie_name = str(VIE_NAMES.get(code, code))

        # Gán mã biển báo phía trên box
        bbox_label = draw.textbbox((0, 0), label, font=font_label)
        tw, th = bbox_label[2] - bbox_label[0], bbox_label[3] - bbox_label[1]
        label_y = max(y1 - th - 10, 2)
        
        draw.rectangle([x1, label_y, x1 + tw + 10, label_y + th + 6], fill=color_rgb)
        draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255), font=font_label)

        # Gán nghĩa tiếng Việt phía dưới box
        bbox_vie = draw.textbbox((0, 0), vie_name, font=font_vie)
        vtw, vth = bbox_vie[2] - bbox_vie[0], bbox_vie[3] - bbox_vie[1]
        vie_y = y2 + 4
        
        draw.rectangle([x1, vie_y, x1 + vtw + 8, vie_y + vth + 6], fill=(0, 0, 0))
        draw.text((x1 + 4, vie_y + 2), vie_name, fill=(255, 255, 255), font=font_vie)

    # Thêm banner tóm tắt tổng số biển báo tìm được ở góc trên cùng
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if summary_list:
        summary_text = "DETECTED: " + ", ".join(summary_list)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), (0, 0, 0), -1)
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 8), summary_text, fill=(0, 255, 0), font=get_pil_font(13))
        annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return annotated

def main():
    parser = argparse.ArgumentParser(description="🚦 Vietnamese Traffic Sign Detection - Video Only")
    parser.add_argument('--model', '-m', default='model02/best28121.pt', help='Đường dẫn file định trọng số YOLO (.pt)')
    parser.add_argument('--source', '-s', required=True, help='Đường dẫn file video đầu vào (.mp4, .avi...)')
    parser.add_argument('--conf', '-c', type=float, default=0.22, help='Ngưỡng tự tin')
    parser.add_argument('--imgsz', type=int, default=1280, help='Kích thước ảnh đưa vào mô hình')
    parser.add_argument('--skip', type=int, default=1, help='Số khung hình bỏ qua giữa mỗi lần predict')
    parser.add_argument('--output', '-o', default='output', help='Thư mục lưu video kết quả')
    parser.add_argument('--display-width', type=int, default=1024, help='Chiều rộng cửa sổ hiển thị')
    
    # 🆕 Tùy chọn cho nhận diện biển báo ở xa/mờ/nhỏ
    parser.add_argument('--enhance', action='store_true', help='Bật tăng cường chất lượng ảnh (LAB enhancement)')
    parser.add_argument('--clahe', action='store_true', help='Bật CLAHE - nâng cao contrast địa phương')
    parser.add_argument('--sharpen', action='store_true', help='Bật làm sắc nét ảnh')
    parser.add_argument('--brightness', type=float, default=1.0, help='Độ sáng (1.0=bình thường, >1.0=sáng hơn)')
    parser.add_argument('--contrast', type=float, default=1.2, help='Độ tương phản (1.0=bình thường, >1.0=cao hơn)')
    
    args = parser.parse_args()

    # 1. Khởi tạo mô hình
    if not os.path.exists(args.model):
        print(f"❌ Không tìm thấy file mô hình tại: {args.model}")
        sys.exit(1)
        
    print(f"🔄 Đang tải mô hình YOLO: {args.model}...")
    model = YOLO(args.model)
    model_classes = model.names
    print(f"✅ Tải thành công! Mô hình có: {len(model_classes)} classes.")

    # 2. Mở File Video
    if not os.path.exists(args.source):
        print(f"❌ Không tìm thấy file video tại: {args.source}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {args.source}")
        sys.exit(1)

    # 3. Thu thập thông số cấu hình video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = min(args.display_width / w, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)

    # 4. Khởi tạo bộ ghi video đầu ra (Sửa lỗi "VideoWriter_fourcc" của Pylance)
    writer = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.source))[0]
        out_path = os.path.join(args.output, f"detected_{base_name}.mp4")
        # Sử dụng thuộc tính chuẩn hóa lớp cha của OpenCV để lách qua linter Pylance
        fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_video, (disp_w, disp_h))
        print(f"💾 Video kết quả sẽ được lưu tại: {out_path}")

    # 🆕 In các tùy chọn enhancement
    enhancements = []
    if args.enhance:
        enhancements.append(f"LAB Enhancement(brightness={args.brightness}, contrast={args.contrast})")
    if args.clahe:
        enhancements.append("CLAHE")
    if args.sharpen:
        enhancements.append("Sharpen")
    
    enh_str = " | ".join(enhancements) if enhancements else "Không có"
    print(f"🚀 Bắt đầu xử lý. Cấu hình Inference: imgsz={args.imgsz} | conf={args.conf}")
    print(f"🎨 Enhancement: {enh_str}")
    
    frame_count = 0
    fps_history = deque(maxlen=30)
    prev_time = time.time()
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🆕 Áp dụng enhancement nếu được chọn
        frame_for_predict = frame.copy()
        if args.enhance:
            frame_for_predict = enhance_image(frame_for_predict, brightness=args.brightness, contrast=args.contrast)
        if args.clahe:
            frame_for_predict = apply_clahe(frame_for_predict)
        if args.sharpen:
            frame_for_predict = sharpen_image(frame_for_predict)

        # Thực hiện nhận diện (Predict) dựa theo tham số thiết lập skip_frames
        if frame_count % (args.skip + 1) == 0 or last_results is None:
            last_results = model.predict(
                source=frame_for_predict, conf=args.conf, iou=0.4, imgsz=args.imgsz, verbose=False
            )

        # SỬA LỖI CÚ PHÁP DÒNG 276: Bọc lại biểu thức kiểm tra chuẩn xác
        if scale < 1.0:
            display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            display_frame = frame.copy()

        # Vẽ các hộp nhận diện đè lên khung hình hiển thị
        if last_results is not None:
            annotated = draw_detections(display_frame, last_results, model_classes, scale_x=scale, scale_y=scale)
            num_det = len(last_results[0].boxes) if last_results[0].boxes is not None else 0
        else:
            annotated = display_frame
            num_det = 0

        # Tính toán FPS hiển thị thời gian thực
        curr_time = time.time()
        dt = curr_time - prev_time
        if dt > 0:
            fps_history.append(1.0 / dt)
        prev_time = curr_time
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        # Vẽ HUD thông tin cơ bản lên video
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, disp_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(annotated, (0, disp_h - 35), (disp_w, disp_h), (0, 0, 0), -1)
        info_str = f"Khung hinh: {frame_count}/{total_frames} | Tim thay: {num_det} bien bao | Nhan 'q' de thoat"
        cv2.putText(annotated, info_str, (10, disp_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # In tiến độ ra Terminal
        if total_frames > 0:
            progress = (frame_count / total_frames) * 100
            sys.stdout.write(f"\r⏳ Dang xu ly: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {avg_fps:.1f} | Detections: {num_det} ")
            sys.stdout.flush()

        # Hiển thị và lưu khung hình
        cv2.imshow("Traffic Sign Video Inference", annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Đã bấm dừng chương trình.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n\n✅ Hoàn thành! Đã xử lý toàn bộ {frame_count} khung hình.")

if __name__ == '__main__':
    main()