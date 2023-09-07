import streamlit as st
from PIL import Image
import os
#Trick to not init function multitime
if "ocr_detector" not in st.session_state:
    print("INIT MODEL")
    os.system("wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb")
    os.system("sudo dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb")
    from src.setup import Setup
    Setup().ocr_model_downloader()
    
    from src.OCR import OCRDetector
    st.session_state.ocr_detector = OCRDetector()
    print("DONE INIT MODEL")

st.set_page_config(page_title="Vietnamese OCR", layout="wide", page_icon = "./storage/linhai.jpeg")
hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style='text-align: center; color: grey;'>Input: Image </h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey;'>Output: The Vietnamese or English text in the image (if any).</h2>", unsafe_allow_html=True)
left_col, right_col = st.columns(2)

#LEFT COLUMN
upload_image = left_col.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp", ])

if left_col.button("OCR Detect"):
    image, texts, boxes = st.session_state.ocr_detector.text_detector(upload_image, is_local=True)
    left_col.write("**RESULTS:** ")
    left_col.write(texts)
    
    #RIGHT COLUMN
    visualize_image = st.session_state.ocr_detector.visualize_ocr(image, texts, boxes)
    right_col.write("**ORIGIN IMAGE:** ")
    right_col.image(image)
    right_col.write("**OCR IMAGE:** ")
    right_col.image(visualize_image)