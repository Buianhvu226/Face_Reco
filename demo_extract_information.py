from google import genai
import os

client = genai.Client(api_key="AIzaSyD-Nw9rDFIa5mLgtjrPvlYHeW0Uh4i3jd8")

def analyze_child_response_gemini(response_text, missing_persons):
    try:
        # Đơn giản hóa prompt
        prompt = f"""
        Phân tích câu trả lời: "{response_text}"
        Nhiệm vụ:
        1. Tìm thông tin có thể có: tên, tên phụ huynh, địa chỉ hoặc trường học,...
        2. So sánh với danh sách: {missing_persons}
        3. Kết quả: Chỉ cần đưa ra tên người phù hợp hoặc "Không tìm thấy". Không cần giải thích chi tiết.

        Ví dụ câu trả lời chỉ cần là: "Lê Như Bảo" hoặc "Không tìm thấy"
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text if response.text else "Không tìm thấy kết quả"
            
    except Exception as e:
        print(f"Lỗi xử lý: {str(e)}")
        return "Không tìm thấy kết quả"

# Test
child_response = "Cháu mất tích hôm chiều 14/02/2025"
missing_persons_data = [
    {
        "name": "Nguyễn Văn Nam",
        "age": "10 tuổi",
        "school": "Trường Tiểu học ABC, Quận 1",
        "last_seen": "15/02/2025",
        "description": "Mặc đồng phục học sinh, cặp sách màu xanh",
        "parents": "Bố: Nguyễn Văn An, Mẹ: Trần Thị Mai",
        "address": "123 Lê Lợi, Quận 1, TP.HCM",
        "status": "Mất tích"
    },
    {
        "name": "Trần Thị Minh Anh",
        "age": "8 tuổi",
        "school": "Trường Tiểu học XYZ, Quận 3",
        "last_seen": "14/02/2025",
        "description": "Tóc dài ngang vai, mặc áo hồng quần jean",
        "parents": "Bố: Trần Văn Hùng, Mẹ: Lê Thị Hoa",
        "address": "45 Nguyễn Thị Minh Khai, Quận 3, TP.HCM",
        "status": "Mất tích"
    },
    {
        "name": "Lê Hoàng Bảo",
        "age": "12 tuổi",
        "school": "Trường THCS DEF, Quận Tân Bình",
        "last_seen": "16/02/2025",
        "description": "Cao 1m45, mặc áo thể thao màu đỏ",
        "parents": "Bố: Lê Văn Dũng, Mẹ: Phạm Thị Lan",
        "address": "78 Cộng Hòa, Quận Tân Bình, TP.HCM",
        "status": "Mất tích"
    },
    {
        "name": "Phạm Thu Trang",
        "age": "9 tuổi",
        "school": "Trường Tiểu học Sunshine, Quận 7",
        "last_seen": "13/02/2025",
        "description": "Đeo kính cận, mặc váy hoa",
        "parents": "Bố: Phạm Văn Thành, Mẹ: Nguyễn Thị Hương",
        "address": "256 Nguyễn Thị Thập, Quận 7, TP.HCM",
        "status": "Mất tích"
    },
    {
        "name": "Hoàng Minh Đức",
        "age": "11 tuổi",
        "school": "Trường Tiểu học Victory, Quận 2",
        "last_seen": "12/02/2025",
        "description": "Cao 1m40, da ngăm, thích chơi bóng đá",
        "parents": "Bố: Hoàng Văn Tuấn, Mẹ: Võ Thị Nga",
        "address": "89 Mai Chí Thọ, Quận 2, TP.HCM",
        "status": "Mất tích"
    }
]

result = analyze_child_response_gemini(child_response, missing_persons_data)
print(f"Kết quả: {result}")
