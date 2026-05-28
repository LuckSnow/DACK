"""
🚦 Vietnamese Traffic Sign Detection - Image Recognition
========================================================
Chương trình nhận diện biển báo giao thông từ ảnh tĩnh.
Sử dụng mô hình YOLOv8 đã huấn luyện trên tập dữ liệu biển báo Việt Nam.

Usage:
    python image.py --image_path path/to/image.jpg
    python image.py --image_path path/to/image.jpg --conf 0.25
    python image.py --image_path path/to/image.jpg --conf 0.25 --save
"""

import argparse
import os
from pathlib import Path
import glob

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
# Hàm xác định màu BGR theo nhóm biển báo
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


# ============================================================
# Lớp nhận diện biển báo giao thông
# ============================================================
class TrafficSignDetector:
    """Nhận diện biển báo giao thông từ ảnh."""
    
    def __init__(self, model_path: str = "model02/best28121.pt", conf: float = 0.25, imgsz: int = 640):
        """
        Khởi tạo detector.
        
        Args:
            model_path: Đường dẫn tới file mô hình (.pt)
            conf: Ngưỡng tự tin tối thiểu (0-1)
            imgsz: Kích thước ảnh xử lý (640, 832, 1024, 1280). Ảnh lớn hơn = chính xác hơn nhưng chậm hơn
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        print(f"✓ Tải mô hình thành công: {model_path}")
        print(f"✓ Ngưỡng tự tin: {conf}")
        print(f"✓ Kích thước xử lý (imgsz): {imgsz}px")
    
    def detect(self, image_path: str) -> dict:
        """
        Nhận diện biển báo trong ảnh.
        
        Args:
            image_path: Đường dẫn tới ảnh
            
        Returns:
            dict: Thông tin về các biển báo được phát hiện
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        
        # Lấy thông tin ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        h, w = image.shape[:2]
        
        # Chạy inference với imgsz
        results = self.model(image_path, conf=self.conf, imgsz=self.imgsz, verbose=False)
        result = results[0]
        
        # Xử lý kết quả
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            bbox = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "class_name_vi": VIE_NAMES.get(class_name, class_name),
                "confidence": confidence,
                "bbox": bbox
            }
            detections.append(detection)
        
        return {
            "image_path": image_path,
            "image_size": {"width": w, "height": h},
            "total_signs": len(detections),
            "detections": detections
        }
    
    def draw_results(self, image_path: str, output_path: str | None = None, verbose: bool = True) -> str:
        """
        Vẽ bounding box và label lên ảnh.
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            output_path: Đường dẫn lưu ảnh kết quả (nếu None, sẽ lưu trong thư mục output/)
            verbose: In thông báo hay không (mặc định: True)
            
        Returns:
            str: Đường dẫn ảnh đã lưu
        """
        # Nhận diện
        detection_result = self.detect(image_path)
        detections = detection_result["detections"]
        
        # Tải ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Chuyển đổi BGR sang RGB cho PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(image_pil)
        
        # Lấy font hỗ trợ tiếng Việt
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 32)
            except:
                font = ImageFont.load_default()
        
        h, w = image.shape[:2]
        
        # Vẽ từng phát hiện
        for det in detections:
            x1 = int(det["bbox"]["x1"])
            y1 = int(det["bbox"]["y1"])
            x2 = int(det["bbox"]["x2"])
            y2 = int(det["bbox"]["y2"])
            
            class_name = det["class_name"]
            class_name_vi = det["class_name_vi"]
            confidence = det["confidence"]
            
            # Lấy màu
            color = get_class_color_bgr(class_name)
            color_rgb = (color[2], color[1], color[0])  # BGR to RGB for PIL
            
            # Vẽ bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            
            # Tạo label
            label = f"{class_name_vi} ({confidence:.2f})"
            
            # Tính kích thước text
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Vẽ nền cho text
            text_x = x1
            text_y = y1 - text_h - 8
            draw.rectangle(
                [text_x, text_y, text_x + text_w + 8, y1],
                fill=color_rgb
            )
            
            # Vẽ text
            draw.text(
                (text_x + 4, text_y + 2),
                label,
                font=font,
                fill=(255, 255, 255)
            )
        
        # Chuyển đổi trở lại BGR cho OpenCV
        image_result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Lưu ảnh
        if output_path is None:
            os.makedirs("output", exist_ok=True)
            filename = Path(image_path).stem + "_detected.jpg"
            output_path = os.path.join("output", filename)
        
        cv2.imwrite(output_path, image_result)
        
        if verbose:
            print(f"\n✓ Ảnh kết quả đã lưu: {output_path}")
        
        return output_path
    
    def detect_folder(self, folder_path: str, output_folder: str | None = None) -> dict:
        """
        Nhận diện biển báo trong toàn bộ ảnh trong folder.
        
        Args:
            folder_path: Đường dẫn tới folder chứa ảnh
            output_folder: Folder lưu ảnh kết quả (nếu None, sẽ lưu trong folder output/)
            
        Returns:
            dict: Thông tin chi tiết về kết quả nhận diện toàn bộ folder
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Không tìm thấy folder: {folder_path}")
        
        # Tìm tất cả các file ảnh
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not image_files:
            raise FileNotFoundError(f"Không tìm thấy ảnh nào trong folder: {folder_path}")
        
        image_files.sort()
        
        # Tạo folder output
        if output_folder is None:
            output_folder = os.path.join("output", Path(folder_path).name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Xử lý từng ảnh
        folder_results = {
            "folder_path": folder_path,
            "total_images": len(image_files),
            "total_signs_found": 0,
            "images": [],
            "sign_statistics": {}
        }
        
        print(f"\n{'='*60}")
        print(f"🔍 NHẬN DIỆN FOLDER: {folder_path}")
        print(f"📊 Tổng ảnh: {len(image_files)}")
        print(f"{'='*60}\n")
        
        for idx, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            print(f"[{idx}/{len(image_files)}] Xử lý: {filename}...", end=" ")
            
            try:
                # Nhận diện ảnh
                detection_result = self.detect(image_path)
                num_signs = detection_result["total_signs"]
                
                # Vẽ kết quả (không in message trung gian)
                output_path = os.path.join(output_folder, 
                                         Path(filename).stem + "_detected.jpg")
                self.draw_results(image_path, output_path=output_path, verbose=False)
                
                # Lưu thống kê
                image_info = {
                    "filename": filename,
                    "num_signs": num_signs,
                    "detections": detection_result["detections"],
                    "output_path": output_path
                }
                folder_results["images"].append(image_info)
                folder_results["total_signs_found"] += num_signs
                
                # Cập nhật thống kê biển báo
                for det in detection_result["detections"]:
                    class_name_vi = det["class_name_vi"]
                    if class_name_vi not in folder_results["sign_statistics"]:
                        folder_results["sign_statistics"][class_name_vi] = 0
                    folder_results["sign_statistics"][class_name_vi] += 1
                
                print(f"✓ ({num_signs} biển báo)")
                
            except Exception as e:
                print(f"❌ Lỗi: {str(e)}")
                folder_results["images"].append({
                    "filename": filename,
                    "error": str(e)
                })
        
        return folder_results
    
    def print_folder_results(self, folder_results: dict):
        """In kết quả nhận diện folder ra console."""
        print("\n" + "="*70)
        print("📊 TÓM TẮT KẾT QUẢ NHẬN DIỆN FOLDER")
        print("="*70)
        print(f"Folder: {folder_results['folder_path']}")
        print(f"Tổng ảnh: {folder_results['total_images']}")
        print(f"Tổng biển báo phát hiện: {folder_results['total_signs_found']}")
        print("-"*70)
        
        # Thống kê biển báo
        if folder_results['sign_statistics']:
            print("\n📈 THỐNG KÊ LOẠI BIỂN BÁO:")
            for class_name, count in sorted(folder_results['sign_statistics'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"  • {class_name}: {count}")
        
        # Chi tiết từng ảnh với tên biển báo
        print("\n📋 CHI TIẾT NHẬN DIỆN TỪNG ẢNH:")
        print("-"*70)
        
        for idx, img_info in enumerate(folder_results['images'], 1):
            if 'error' in img_info:
                print(f"\n[{idx}] ❌ {img_info['filename']}: {img_info['error']}")
            else:
                print(f"\n[{idx}] ✓ {img_info['filename']}")
                print(f"     Tổng biển báo: {img_info['num_signs']}")
                
                if img_info['num_signs'] > 0:
                    print(f"     Chi tiết:")
                    for det_idx, det in enumerate(img_info['detections'], 1):
                        print(f"       ({det_idx}) {det['class_name_vi']}")
                        print(f"           - Tên gốc: {det['class_name']}")
                        print(f"           - Độ tự tin: {det['confidence']:.2%}")
                
                print(f"     📁 Ảnh kết quả (có bounding box): {img_info['output_path']}")
        
        print("\n" + "="*70)
        print(f"\n✅ Tất cả ảnh đã được xử lý!")
        print(f"📁 Các ảnh kết quả được lưu trong folder: {os.path.dirname(folder_results['images'][0]['output_path']) if folder_results['images'] and 'output_path' in folder_results['images'][0] else 'output'}")
        print("="*70)
    
    def print_results(self, detection_result: dict):
        """In kết quả phát hiện ra console."""
        print("\n" + "="*60)
        print("📊 KẾT QUẢ NHẬN DIỆN BIỂN BÁO")
        print("="*60)
        print(f"Ảnh: {detection_result['image_path']}")
        
        # Hiển thị kích thước ảnh
        if 'image_size' in detection_result:
            size = detection_result['image_size']
            print(f"Kích thước ảnh: {size['width']}x{size['height']}px")
        
        print(f"Số lượng biển báo phát hiện: {detection_result['total_signs']}")
        print("-"*60)
        
        if detection_result['total_signs'] > 0:
            for i, det in enumerate(detection_result["detections"], 1):
                print(f"\n[{i}] {det['class_name_vi']}")
                print(f"    - Tên gốc: {det['class_name']}")
                print(f"    - Độ tự tin: {det['confidence']:.2%}")
                bbox = det["bbox"]
                print(f"    - Vị trí: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) -> ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
        else:
            print("\n⚠ Không phát hiện biển báo nào trong ảnh")
        
        print("\n" + "="*60)


# ============================================================
# Hàm chính
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Nhận diện biển báo giao thông từ ảnh hoặc folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Nhận diện một ảnh
  python image.py --image_path test.jpg
  python image.py --image_path test.jpg --imgsz 1024 --conf 0.3
  
  # Nhận diện toàn bộ folder
  python image.py --folder bienbao
  python image.py --folder bienbao --imgsz 1280 --conf 0.25
  
  # Tùy chọn imgsz (kích thước xử lý):
  # 640  = mặc định, nhanh, phù hợp ảnh bình thường
  # 832  = cân bằng, phù hợp ảnh vừa
  # 1024 = chi tiết cao, phù hợp ảnh lớn/biển báo nhỏ
  # 1280 = chi tiết rất cao, phù hợp ảnh rất lớn
        """
    )
    
    parser.add_argument("--image_path", "-i", 
                       help="Đường dẫn tới ảnh cần nhận diện")
    parser.add_argument("--folder", "-f",
                       help="Đường dẫn tới folder chứa các ảnh")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Ngưỡng tự tin (0-1, mặc định: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Kích thước xử lý (640, 832, 1024, 1280; mặc định: 640)")
    parser.add_argument("--model", "-m", default="model02/best28121.pt",
                       help="Đường dẫn tới mô hình (mặc định: model02/best28121.pt)")
    parser.add_argument("--save", "-s", 
                       help="Lưu ảnh kết quả vào đường dẫn này (chỉ dùng với --image_path)")
    parser.add_argument("--output", "-o",
                       help="Folder lưu kết quả (chỉ dùng với --folder)")
    
    args = parser.parse_args()
    
    # Kiểm tra tham số đầu vào
    if not args.image_path and not args.folder:
        parser.print_help()
        print("\n❌ Lỗi: Phải chỉ định --image_path hoặc --folder")
        return 1
    
    if args.image_path and args.folder:
        print("❌ Lỗi: Chỉ được chỉ định --image_path hoặc --folder, không cả hai")
        return 1
    
    try:
        # Tạo detector
        detector = TrafficSignDetector(model_path=args.model, conf=args.conf, imgsz=args.imgsz)
        
        # Xử lý ảnh đơn lẻ
        if args.image_path:
            print(f"\n🔍 Đang nhận diện ảnh: {args.image_path}")
            result = detector.detect(args.image_path)
            detector.print_results(result)
            
            output_path: str | None = args.save if args.save else None
            detector.draw_results(args.image_path, output_path=output_path)
        
        # Xử lý folder
        elif args.folder:
            print(f"\n🔍 Đang nhận diện folder: {args.folder}")
            folder_results = detector.detect_folder(args.folder, output_folder=args.output)
            detector.print_folder_results(folder_results)
        
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}")
        return 1
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
