import cv2
import face_recognition
import numpy as np
import os

KNOWN_FACES_FOLDER = "F:\\Face_Reco\\glasses_face"
UNKNOWN_IMAGE_PATH = "F:\\Face_Reco\\small_know_face"
# KNOWN_FACES_FOLDER = "F:\\Face_Reco\\know_face"
# UNKNOWN_IMAGE_PATH = "F:\\Face_Reco\\unknow_face"
REPORT_FILE = "F:\\Face_Reco\\quality_report.txt"


def check_face_quality(image_path):
    issues = []
    image = cv2.imread(image_path)
    if image is None:
        return ["❌ Không thể đọc ảnh"]
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    if len(face_locations) == 0:
        return ["❌ Không tìm thấy khuôn mặt"]
    if len(face_locations) > 1:
        return ["❌ Phát hiện nhiều khuôn mặt trong ảnh"]
    
    landmarks = face_recognition.face_landmarks(rgb_image, face_locations)
    if not landmarks:
        return ["❌ Không thể phát hiện các điểm đặc trưng"]
    
    landmark = landmarks[0]
    left_eye = np.mean(landmark['left_eye'], axis=0)
    right_eye = np.mean(landmark['right_eye'], axis=0)
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    if abs(eye_angle) > 10:
        issues.append(f"❌ Góc mặt lệch ({int(eye_angle)} độ)")
    
    (top, right, bottom, left) = face_locations[0]
    face_height = bottom - top
    face_width = right - left
    img_height, img_width = image.shape[:2]
    
    if (face_height / img_height < 0.15) or (face_width / img_width < 0.15):
        issues.append("❌ Khuôn mặt chiếm dưới 15% diện tích ảnh")
    
    face_roi = image[top:bottom, left:right]
    if face_roi.size > 0:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_value < 15:
            issues.append(f"❌ Độ mờ cao (Laplacian: {blur_value:.1f})")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[..., 2].mean()
    contrast = hsv[..., 2].std()
    
    if brightness < 60:
        issues.append(f"❌ Ánh sáng yếu ({brightness:.1f}/255)")
    elif brightness > 200:
        issues.append(f"❌ Ánh sáng chói ({brightness:.1f}/255)")
    
    if contrast < 40:
        issues.append(f"❌ Độ tương phản thấp ({contrast:.1f})")
    
    # Phát hiện kính
    eye_regions = [landmark['left_eye'], landmark['right_eye']]
    for eye_region in eye_regions:
        x_coords = [p[0] for p in eye_region]
        y_coords = [p[1] for p in eye_region]
        x_min, x_max = max(0, min(x_coords) - 5), min(img_width, max(x_coords) + 5)
        y_min, y_max = max(0, min(y_coords) - 5), min(img_height, max(y_coords) + 5)
        
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size > 500:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contrast_value = roi_gray.std()
            brightness_value = roi_gray.mean()
            if contrast_value < 20 or brightness_value > 200:
                issues.append("❌ Có thể đang đeo kính")
                break
    
    return issues if issues else ["✅ Ảnh đạt chất lượng tốt"]


def evaluate_images(folder_path):
    results = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(valid_extensions):
            path = os.path.join(folder_path, file)
            issues = check_face_quality(path)
            results.append(f"\n📷 {file}:")
            results.extend([f"  - {issue}" for issue in issues])
    
    return results


def generate_report():
    report = ["📄 BÁO CÁO CHẤT LƯỢNG ẢNH KHUÔN MẶT", ""]
    report.append("=== ẢNH ĐÃ BIẾT ===")
    report.extend(evaluate_images(KNOWN_FACES_FOLDER))
    report.append("\n=== ẢNH CHƯA BIẾT ===")
    report.extend(evaluate_images(UNKNOWN_IMAGE_PATH))
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"✅ Đã tạo báo cáo tại: {REPORT_FILE}")


if __name__ == "__main__":
    generate_report()
