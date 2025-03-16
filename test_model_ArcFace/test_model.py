import insightface
import cv2
import numpy as np
from sklearn.neighbors import KDTree

# Khởi tạo mô hình ArcFace từ insightface
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)  # ctx_id=0: CPU, ctx_id=1: GPU nếu có

# Cơ sở dữ liệu lưu trữ
FACE_DATABASE = {
    "encodings": [],       # Embeddings từ ArcFace
    "names": [],           # Tên người dùng
    "registered_ages": []  # Tuổi khi đăng ký
}

# Tham số điều chỉnh
AGE_DIFF_WEIGHT = 0.15  # Trọng số ảnh hưởng của chênh lệch tuổi
BASE_THRESHOLD = 0.6    # Ngưỡng similarity cơ bản

# Hàm đăng ký người dùng
def register_user(name, image_path, registered_age):
    """Đăng ký người dùng với ảnh và tuổi"""
    img = cv2.imread(image_path)
    faces = model.get(img)
    
    if len(faces) == 0:
        print("❌ Không phát hiện khuôn mặt")
        return False
    
    embedding = faces[0].embedding
    FACE_DATABASE["encodings"].append(embedding)
    FACE_DATABASE["names"].append(name)
    FACE_DATABASE["registered_ages"].append(registered_age)
    print(f"✅ Đã đăng ký {name} (tuổi {registered_age})")
    return True

# Hàm điều chỉnh ngưỡng theo chênh lệch tuổi
def dynamic_threshold(age_diff):
    """Điều chỉnh ngưỡng similarity dựa trên chênh lệch tuổi"""
    return BASE_THRESHOLD - (age_diff * AGE_DIFF_WEIGHT / 10)

# Hàm so sánh có xét đến tuổi
def age_aware_compare(query_embedding, query_age):
    """So sánh embedding với database, có xét đến tuổi"""
    similarities = []
    
    for db_embedding, db_name, db_age in zip(FACE_DATABASE["encodings"],
                                             FACE_DATABASE["names"],
                                             FACE_DATABASE["registered_ages"]):
        # Tính similarity cosine
        cos_sim = np.dot(query_embedding, db_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
        )
        
        # Điều chỉnh similarity theo chênh lệch tuổi
        age_diff = abs(query_age - db_age)
        adjusted_sim = cos_sim * (1 - AGE_DIFF_WEIGHT) + (1 - age_diff / 100) * AGE_DIFF_WEIGHT
        similarities.append((db_name, adjusted_sim, age_diff))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Hàm xác minh ảnh
def verify_image(image_path, query_age):
    """Xác minh ảnh mới với database"""
    img = cv2.imread(image_path)
    faces = model.get(img)
    
    if len(faces) == 0:
        print("❌ Không phát hiện khuôn mặt trong ảnh")
        return None
    
    query_embedding = faces[0].embedding
    results = age_aware_compare(query_embedding, query_age)
    
    # Lấy kết quả tốt nhất
    best_match = results[0]
    dynamic_thresh = dynamic_threshold(best_match[2])
    
    if best_match[1] > dynamic_thresh:
        return {
            "name": best_match[0],
            "confidence": best_match[1],
            "age_difference": best_match[2]
        }
    else:
        return {"status": "NO_MATCH"}

# Hàm cải tiến: Thêm ảnh tham chiếu
def add_reference_image(name, new_image_path, age):
    """Thêm ảnh tham chiếu cho người đã đăng ký"""
    return register_user(name, new_image_path, age)

# Hàm tối ưu: Tạo embedding tổng hợp
def temporal_adjustment(name):
    """Tạo embedding tổng hợp từ nhiều ảnh của cùng một người"""
    indices = [i for i, n in enumerate(FACE_DATABASE["names"]) if n == name]
    if len(indices) < 2:
        return  # Không đủ dữ liệu để tổng hợp
    
    embeddings = [FACE_DATABASE["encodings"][i] for i in indices]
    ages = [FACE_DATABASE["registered_ages"][i] for i in indices]
    age_weights = np.array([1 / (abs(age - 25) + 1) for age in ages])  # Ưu tiên tuổi trung bình
    combined_embedding = np.average(embeddings, axis=0, weights=age_weights)
    
    # Cập nhật lại database
    for i in indices[::-1]:
        FACE_DATABASE["encodings"].pop(i)
        FACE_DATABASE["names"].pop(i)
        FACE_DATABASE["registered_ages"].pop(i)
    FACE_DATABASE["encodings"].append(combined_embedding)
    FACE_DATABASE["names"].append(name)
    FACE_DATABASE["registered_ages"].append(25)  # Tuổi trung bình giả định

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đăng ký người dùng
    register_user("AnhVu", "known/Vu_20.jpg", 20)
    add_reference_image("AnhVu", "known/Vu_11.jpg", 11)
    
    # Tối ưu embedding
    temporal_adjustment("AnhVu")
    
    # Xác minh ảnh trẻ em
    result = verify_image("unknown/uf4.jpg", 4)
    
    if result and "name" in result:
        print(f"🔍 Kết quả: {result['name']}")
        print(f"   Độ tin cậy: {result['confidence']*100:.1f}%")
        print(f"   Chênh lệch tuổi: {result['age_difference']} năm")
    else:
        print("🔴 Không tìm thấy kết quả phù hợp")