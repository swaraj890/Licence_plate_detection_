import cv2
import numpy as np
import streamlit as st
import sqlite3
import smtplib
from email.message import EmailMessage
from PIL import Image
from ultralytics import YOLO
import easyocr
import pandas as pd
import re

# ---------------------- DATABASE SETUP ----------------------
def get_db_connection():
    return sqlite3.connect("license_plate.db")

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS license_plates (
            plate TEXT PRIMARY KEY,
            owner TEXT,
            email TEXT,
            detections INTEGER DEFAULT 0
        )
    ''')

    preloaded_data = [
        ("Y6133", "John Doe", "johndoe@example.com", 2),
        ("786133", "Jane Smith", "janesmith@example.com", 5),
        ("Y86433", "Alex Brown", "alexbrown@example.com", 1),
        ("YB6433", "Emily White", "swaraj890806@gmail.com", 3),
        ("NN773", "Chris Green", "swaraj890806@gmail.com", 4),
    ]

    cursor.executemany("INSERT OR IGNORE INTO license_plates VALUES (?, ?, ?, ?)", preloaded_data)
    conn.commit()
    conn.close()

initialize_database()

# ---------------------- EMAIL NOTIFICATION ----------------------
def send_email(receiver_email, plate, count):
    sender_email = "ayushtiwari.creatorslab@gmail.com"
    sender_password = "fixk brsn svwe tjyt"

    subject = f"🚗 License Plate Detected: {plate}"
    body = f"Hello,\n\nYour vehicle with plate number {plate} has been detected. \nTotal detections: {count}\n\nBest Regards,\nLicense Plate Detector"

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Email failed: {e}")

# ---------------------- LICENSE PLATE DETECTION ----------------------
model = YOLO("license_plate_detector.pt")  
reader = easyocr.Reader(['en'])

detected_plates = []

def preprocess_plate(plate):
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate = cv2.GaussianBlur(plate, (5, 5), 0)
    _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate = cv2.resize(plate, (200, 60))
    return plate

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.replace(" ", ""))

def detect_plate(image):
    global detected_plates
    image = np.array(image)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model(image)
    extracted_text = "No plate detected"

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            plate_roi = image[y1:y2, x1:x2]

            if plate_roi is not None:
                plate_roi = preprocess_plate(plate_roi)
                extracted_text = reader.readtext(plate_roi, detail=0)
                extracted_text = clean_text(" ".join(extracted_text))

                if extracted_text and len(extracted_text) > 4:
                    detected_plates.append({"License Plate": extracted_text})
                    process_detected_plate(extracted_text)

    return image, extracted_text

def process_detected_plate(plate):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT owner, email, detections FROM license_plates WHERE plate = ?", (plate,))
    result = cursor.fetchone()

    if result:
        owner, email, detections = result
        new_detections = detections + 1
        cursor.execute("UPDATE license_plates SET detections = ? WHERE plate = ?", (new_detections, plate))
        conn.commit()
        send_email(email, plate, new_detections)
    else:
        st.warning(f"No record found for `{plate}`. Consider adding it to the database.")

    conn.close()

def process_video(video_path):
    global detected_plates
    detected_plates = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            detect_plate(frame)
        frame_count += 1

    cap.release()

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="wide")

st.title("🚗 License Plate Recognition System")
st.markdown("### Detect license plates from images & videos, store in a database, and send email alerts to owners.")

option = st.sidebar.radio("Choose Input Type", ["Image", "Video", "Database Search"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image, plate_text = detect_plate(image)
        st.image(processed_image, caption="Detected License Plate", use_column_width=True)
        st.subheader(f"Extracted License Plate: `{plate_text}`")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        st.video("temp_video.mp4")
        process_video("temp_video.mp4")

elif option == "Database Search":
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM license_plates", conn)
    conn.close()
    
    st.subheader("📋 Stored License Plates")
    st.dataframe(df)

    search_query = st.text_input("🔍 Search by License Plate")
    if search_query:
        filtered_df = df[df["plate"].str.contains(search_query, case=False)]
        st.dataframe(filtered_df)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50; 
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)
