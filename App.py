import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
from langdetect import detect

# Cấu hình giao diện
st.set_page_config(page_title="Giải bài tập từ ảnh", page_icon=":memo:", layout="wide")

st.markdown("""
    <style>
        body { background-color: #0E1117; color: white; }
        .stTextArea textarea { background-color: #262730; color: white; }
        .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
        .stFileUploader label { color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Giải bài tập tự động từ ảnh ✨")

def preprocess_image(image, sharpen=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        resized = cv2.filter2D(resized, -1, kernel)
    return resized

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def crop_text_area(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = image[y:y+h, x:x+w]
    return cropped

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

uploaded_file = st.file_uploader("**Tải ảnh đề bài lên:**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh gốc đã tải lên", use_container_width=True)

    sharpen = st.checkbox("**Làm nét ảnh để tăng độ rõ nét**", value=True)
    run_ocr = st.button("**Bắt đầu đọc đề bài**")

    if run_ocr:
        img = np.array(image)
        processed = preprocess_image(img, sharpen=sharpen)
        rotated = deskew(processed)
        cropped = crop_text_area(rotated)

        temp_text = pytesseract.image_to_string(cropped, lang="eng")
        lang_code = detect_language(temp_text)

        if lang_code == "vi":
            ocr_lang = "vie"
        else:
            ocr_lang = "eng"

        extracted_text = pytesseract.image_to_string(cropped, lang=ocr_lang)

        st.subheader("📄 Kết quả nhận diện:")
        st.text_area("**Nội dung trích xuất được:**", extracted_text, height=400)

        st.success(f"✅ Hệ thống tự động phát hiện ngôn ngữ: {'Tiếng Việt' if ocr_lang == 'vie' else 'Tiếng Anh'}")

        st.subheader("📷 Ảnh sau xử lý:")
        st.image(cropped, caption="Ảnh sau khi xoay và cắt gọn", use_container_width=True)
