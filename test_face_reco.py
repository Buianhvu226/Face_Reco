# import face_recognition
# import cv2
# import os
# import pickle
# import numpy as np
# from sklearn.neighbors import KDTree
# from concurrent.futures import ThreadPoolExecutor
# import queue

# # 📌 Thư mục chứa ảnh đã biết
# KNOWN_FACES_FOLDER = "F:\\Face_Reco\\know_face"
# # KNOWN_FACES_FOLDER = "F:\\Face_Reco\\small_know_face"
# # UNKNOWN_IMAGE_PATH = "F:\\Face_Reco\\unknow_face\\uf10.png"
# UNKNOWN_IMAGE_PATH = "F:\\Face_Reco\\unknow_face\\uf5.jpg"
# ENCODINGS_FILE = "face_encodings.pkl"  # Lưu mã hóa khuôn mặt

# # 📌 Hàng đợi xử lý đa luồng
# task_queue = queue.Queue()


# # 📌 Hàm cắt khuôn mặt từ ảnh
# def extract_face(image, scale=0.5):
#     small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
#     face_locations = face_recognition.face_locations(small_image, model="cnn")
#     face_encodings = face_recognition.face_encodings(small_image, face_locations)

#     cropped_faces = []
#     for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
#         top, right, bottom, left = [int(coord / scale) for coord in (top, right, bottom, left)]
#         face_image = image[top:bottom, left:right]
#         cropped_faces.append((face_image, encoding))
    
#     return cropped_faces


# # 📌 Hàm xử lý ảnh đa luồng
# def process_image(image_path):
#     image = face_recognition.load_image_file(image_path)
#     image = cv2.resize(image, (500, 500))  # Giảm kích thước ảnh để tăng tốc
#     face_encodings = face_recognition.face_encodings(image)
#     if face_encodings:
#         return os.path.splitext(os.path.basename(image_path))[0], face_encodings[0]
#     return None

# # 📌 Hàm tải và mã hóa khuôn mặt
# def load_and_encode_faces(image_folder):
#     if os.path.exists(ENCODINGS_FILE):
#         print("📂 Đang tải dữ liệu đã mã hóa...")
#         with open(ENCODINGS_FILE, "rb") as f:
#             return pickle.load(f)
    
#     print("🔄 Mã hóa lại khuôn mặt...")
#     known_face_encodings = []
#     known_face_names = []
#     image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
#                    if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff", ".webp"))]
    
#     with ThreadPoolExecutor(max_workers=14) as executor:
#         results = list(executor.map(process_image, image_paths))
    
#     for result in results:
#         if result:
#             name, encoding = result
#             known_face_names.append(name)
#             known_face_encodings.append(encoding)
    
#     with open(ENCODINGS_FILE, "wb") as f:
#         pickle.dump((known_face_encodings, known_face_names), f)
    
#     return known_face_encodings, known_face_names


# # 📌 Hàm nhận diện khuôn mặt với KDTree + Đa luồng
# def recognize_faces(image_path, known_face_encodings, known_face_names):
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     face_locations = face_recognition.face_locations(rgb_image, model="cnn")
#     face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

#     if not face_locations:
#         print("❌ Không tìm thấy khuôn mặt!")
#         return

#     # Xây dựng KDTree để tìm nhanh hơn
#     tree = KDTree(known_face_encodings)

#     # Sử dụng ThreadPoolExecutor để chạy đa luồng
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             task_queue.put(executor.submit(match_face, tree, known_face_names, image, face_encoding, (top, right, bottom, left)))


# # 📌 Hàm so khớp khuôn mặt
# def match_face(tree, known_face_names, image, face_encoding, bbox):
#     top, right, bottom, left = bbox
    
#     # Kiểm tra số lượng khuôn mặt đã biết
#     num_known_faces = len(known_face_names)
#     k_value = min(5, num_known_faces)  # Giới hạn k tối đa bằng số lượng khuôn mặt đã biết

#     dist, index = tree.query([face_encoding], k=k_value)  # Lấy k kết quả gần nhất

#     best_match_index = index[0][0]
#     best_distance = dist[0][0]
#     best_similarity = (1 - best_distance) * 100  # Chuyển đổi khoảng cách thành %

#     name = "Unknown"
#     if best_distance < 0.5:  # Nếu độ giống trên 50%
#         name = known_face_names[best_match_index]

#     # Hiển thị kết quả chi tiết
#     print(f"📌 Nhận diện: {name} - Độ giống cao nhất: {best_similarity:.2f}%")
#     print("📋 Danh sách kết quả gần nhất:")
#     for i in range(k_value):  # Duyệt qua tất cả k kết quả
#         match_name = known_face_names[index[0][i]]
#         match_similarity = (1 - dist[0][i]) * 100
#         print(f"  🔹 {match_name}: {match_similarity:.2f}%")

#     # Cắt và hiển thị khuôn mặt
#     face_img = image[top:bottom, left:right]
#     face_img = cv2.resize(face_img, (250, 250))
#     text = f"{name} ({best_similarity:.2f}%)"
#     cv2.putText(face_img, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
#     cv2.imshow(f"Face - {name}", face_img)

#     cv2.waitKey()  # Giữ ảnh hiển thị


# # 🟢 Chạy chương trình
# if __name__ == "__main__":
#     known_face_encodings, known_face_names = load_and_encode_faces(KNOWN_FACES_FOLDER)
#     recognize_faces(UNKNOWN_IMAGE_PATH, known_face_encodings, known_face_names)

#     # Chờ tất cả task trong queue hoàn thành
#     while not task_queue.empty():
#         task_queue.get().result()

#     cv2.destroyAllWindows()




import face_recognition
import cv2
import os
import pickle
import numpy as np
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor
import queue

# 📌 Thư mục chứa ảnh đã biết
KNOWN_FACES_FOLDER = "F:\\Face_Reco\\know_face"
# UNKNOWN_IMAGE_PATH = "F:\\Face_Reco\\unknow_face\\uf10.png"
UNKNOWN_IMAGE_PATH = "F:\\Face_Reco\\unknow_face\\uf41.jpg"
ENCODINGS_FILE = "face_encodings.pkl"  # Lưu mã hóa khuôn mặt

# 📌 Hàng đợi xử lý đa luồng
task_queue = queue.Queue()


# 📌 Hàm cắt khuôn mặt từ ảnh
def extract_face(image, scale=0.5):
    small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    face_locations = face_recognition.face_locations(small_image, model="hog")
    face_encodings = face_recognition.face_encodings(small_image, face_locations)

    cropped_faces = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [int(coord / scale) for coord in (top, right, bottom, left)]
        face_image = image[top:bottom, left:right]
        cropped_faces.append((face_image, encoding))
    
    return cropped_faces


# 📌 Hàm tải và mã hóa khuôn mặt
def load_and_encode_faces(image_folder):
    if os.path.exists(ENCODINGS_FILE):
        print("📂 Đang tải dữ liệu đã mã hóa...")
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)

    print("🔄 Mã hóa lại khuôn mặt...")
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jfif") or filename.endswith(".bmp") or filename.endswith(".gif") or filename.endswith(".tiff") or filename.endswith(".webp"):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            cropped_faces = extract_face(image)

            for _, encoding in cropped_faces:
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])

    # Lưu dữ liệu đã mã hóa
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names


# 📌 Hàm nhận diện khuôn mặt với KDTree + Đa luồng
def recognize_faces(image_path, known_face_encodings, known_face_names):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_locations:
        print("❌ Không tìm thấy khuôn mặt!")
        return

    # Xây dựng KDTree để tìm nhanh hơn
    tree = KDTree(known_face_encodings)

    # Sử dụng ThreadPoolExecutor để chạy đa luồng
    with ThreadPoolExecutor(max_workers=4) as executor:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            task_queue.put(executor.submit(match_face, tree, known_face_names, image, face_encoding, (top, right, bottom, left)))


# 📌 Hàm so khớp khuôn mặt
def match_face(tree, known_face_names, image, face_encoding, bbox):
    top, right, bottom, left = bbox
    
    # Kiểm tra số lượng khuôn mặt đã biết
    num_known_faces = len(known_face_names)
    k_value = min(5, num_known_faces)  # Giới hạn k tối đa bằng số lượng khuôn mặt đã biết

    dist, index = tree.query([face_encoding], k=k_value)  # Lấy k kết quả gần nhất

    best_match_index = index[0][0]
    best_distance = dist[0][0]
    best_similarity = (1 - best_distance) * 100  # Chuyển đổi khoảng cách thành %

    name = "Unknown"
    if best_distance < 0.5:  # Nếu độ giống trên 50%
        name = known_face_names[best_match_index]

    # Hiển thị kết quả chi tiết
    print(f"📌 Nhận diện: {name} - Độ giống cao nhất: {best_similarity:.2f}%")
    print("📋 Danh sách kết quả gần nhất:")
    for i in range(k_value):  # Duyệt qua tất cả k kết quả
        match_name = known_face_names[index[0][i]]
        match_similarity = (1 - dist[0][i]) * 100
        print(f"  🔹 {match_name}: {match_similarity:.2f}%")

    # Cắt và hiển thị khuôn mặt
    face_img = image[top:bottom, left:right]
    face_img = cv2.resize(face_img, (250, 250))
    text = f"{name} ({best_similarity:.2f}%)"
    cv2.putText(face_img, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
    cv2.imshow(f"Face - {name}", face_img)

    cv2.waitKey()  # Giữ ảnh hiển thị


# 🟢 Chạy chương trình
if __name__ == "__main__":
    known_face_encodings, known_face_names = load_and_encode_faces(KNOWN_FACES_FOLDER)
    recognize_faces(UNKNOWN_IMAGE_PATH, known_face_encodings, known_face_names)

    # Chờ tất cả task trong queue hoàn thành
    while not task_queue.empty():
        task_queue.get().result()

    cv2.destroyAllWindows()