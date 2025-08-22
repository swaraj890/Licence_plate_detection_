# Licence_plate_detection_
automatic license plate detection and recognition using a YOLO-based model trained on  Russian number plate datasets


# 🚗 License Plate Detector

This project focuses on **automatic license plate detection and recognition** using a **YOLO-based `.pt` model** trained on **Russian number plate datasets**.  
It allows users to upload vehicle images, detects the license plate, stores the extracted information in a database, and can optionally **send email notifications** with detection results.

---

## 📌 Motivation
With the increase in vehicle usage, **automatic license plate recognition (ALPR)** is crucial for traffic monitoring, parking management, toll systems, and security purposes.  
This project demonstrates an end-to-end ALPR pipeline: from **image input → detection → database storage → email notification**.

---

## ⚙️ Tech Stack
- **Model**: YOLO `.pt` model trained on Russian license plate dataset (Harcases)  
- **Database**: `.db` (SQLite) for storing license plate records  
- **Backend**: Python  
- **Libraries**:  
  - OpenCV (image processing)  
  - Torch / Ultralytics YOLO (model inference)  
  - SQLite3 (database)  
  - smtplib (email sending)  

---

## 🎯 Features
- Upload vehicle images for license plate detection.  
- Detects number plates using YOLO `.pt` model.  
- Extracted license plate numbers are stored in a **SQLite database**.  
- Automatic **email notification system** with detected results.  
- Configurable **user credentials** for email service.  
- Supports **real-time detection** via webcam (optional extension).  

---


## 📂 Project Structure
- licence_plate_database/
- │── app.py # main Streamlit app
- │── app1.py # Alternative app (testing/demo)
- │── car_plate_detector.h5 # Pretrained Keras/TensorFlow model
- │── license_plate_detector.pt # YOLO PyTorch model for detection
- │── haarcascade_russian_plate_number.xml # Haar Cascade for plate detection
- │── license_plate.db # SQLite DB to store detection records
- │── plates.db # Secondary DB for testing
- │── Cars116.png # Sample test image
- │── Cars120.png # Sample test image
- │── demo.mp4 # Demo video for detection

