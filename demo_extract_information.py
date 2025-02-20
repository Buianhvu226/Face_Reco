import openai

# Nhập API Key của bạn
openai.api_key = "YOUR_OPENAI_API_KEY"

def analyze_child_response(response_text, missing_persons):
    prompt = f"""
    Một đứa trẻ bị lạc trả lời: "{response_text}".
    Hãy trích xuất thông tin gồm tên, tên bố mẹ, địa chỉ nhà, và tên trường học.
    Sau đó, so khớp thông tin này với danh sách người bị thất lạc sau:
    {missing_persons}
    Trả về người phù hợp nhất hoặc thông báo không tìm thấy kết quả.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # hoặc "gpt-3.5-turbo" để tiết kiệm chi phí
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Độ chính xác cao hơn với nhiệt độ thấp
        max_tokens=150
    )
    
    return response.choices[0].message["content"]

# Ví dụ danh sách người thất lạc (truy vấn từ Supabase)
missing_persons_data = [
    {"name": "Nguyễn Văn Nam", "parent_names": "Nguyễn Anh, Trần Mai", "address": "Đà Nẵng", "school_name": "Tiểu học ABC"},
    {"name": "Trần Thị Minh", "parent_names": "Trần Hoàng, Nguyễn Hoa", "address": "Hà Nội", "school_name": "Tiểu học XYZ"}
]

# Nhập câu trả lời của trẻ
child_response = "Cháu tên là Nam, bố cháu tên là Anh, nhà cháu ở Đà Nẵng, học trường ABC."
result = analyze_child_response(child_response, missing_persons_data)
print(result)
