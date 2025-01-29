
###   Using DLIB Library 

import sys
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox, QStatusBar, QComboBox
import dlib
from skimage.feature import local_binary_pattern
import pandas as pd
from scipy.spatial.distance import cosine

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)

# Define directories for saving known faces and attendance records
KNOWN_FACES_FOLDER = "known_faces"
UPLOAD_FOLDER = "uploads"
attendance_file = "attendance.csv"

os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize known faces and names
known_faces, known_names = [], []

# Load Dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize OpenCV face cascade for uploaded images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_known_faces():
    """Load all known faces and their names from saved images."""
    known_faces.clear()
    known_names.clear()
    for filename in os.listdir(KNOWN_FACES_FOLDER):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(KNOWN_FACES_FOLDER, filename)
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_face = cv2.resize(rgb_img, (100, 100))  # Ensure consistency in size
            known_faces.append(resized_face)
            known_names.append(filename.split('.')[0])
    return known_faces, known_names

def extract_face_features(face_image):
    """Extract features from the face image using LBP (Local Binary Patterns) and HOG."""
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(gray_face)  # Apply Histogram Equalization to improve image contrast
    
    # LBP Feature Extraction
    lbp = local_binary_pattern(gray_face, 8, 1, method='uniform')  # Local Binary Pattern with uniform pattern
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram
    
    return lbp_hist

def compare_faces(face_features):
    """Compare the extracted features with known faces using Euclidean distance or cosine similarity."""
    best_match_index = None
    min_distance = float('inf')

    for i, known_face in enumerate(known_faces):
        known_face_features = extract_face_features(known_face)
        distance = cosine(face_features, known_face_features)  # Cosine similarity instead of Euclidean distance
        
        logging.info(f"Comparing face with known face {i}, distance: {distance}")  # Log distances

        if distance < min_distance and distance < 0.2:  # Threshold for matching (cosine similarity)
            min_distance = distance
            best_match_index = i
    
    if best_match_index is not None:
        return known_names[best_match_index]
    return "Unknown"

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition and Attendance System")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: #f4f4f4; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;")
        self.initUI()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.video_capture = cv2.VideoCapture(0)
        self.frame = None

        # Load known faces
        global known_faces, known_names
        known_faces, known_names = load_known_faces()

    def initUI(self):
        """Setup UI layout with custom styles."""
        self.layout = QVBoxLayout()

        self.header = QLabel("Face Recognition and Attendance System", self)
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("font-size: 26px; font-weight: bold; color: #fff; padding: 20px 0;")
        self.layout.addWidget(self.header)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(1300, 500)
        self.image_label.setStyleSheet("border: 2px solid #5c5c5c; background-color: #2e2e2e;")
        self.layout.addWidget(self.image_label)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter name for new face")
        self.name_input.setStyleSheet("padding: 10px; font-size: 16px; border-radius: 8px; border: 1px solid #444; color: #ddd; background-color: #333;")
        self.layout.addWidget(self.name_input)

        self.action_combo = QComboBox(self)
        self.action_combo.addItem("Entry")
        self.action_combo.addItem("Exit")
        self.action_combo.setStyleSheet("padding: 10px; font-size: 16px; border-radius: 8px; border: 1px solid #444; background-color: #333; color: #ddd;")
        self.layout.addWidget(self.action_combo)

        self.button_layout = QVBoxLayout()
        self.recognize_button = self.create_button("Start Recognition", "#4CAF50", self.start_recognition)
        self.button_layout.addWidget(self.recognize_button)

        self.register_button = self.create_button("Register Face", "#2196F3", self.register_face, False)
        self.button_layout.addWidget(self.register_button)

        self.upload_button = self.create_button("Upload Image", "#FF5722", self.upload_image)
        self.button_layout.addWidget(self.upload_button)

        self.attendance_button = self.create_button("Mark Attendance", "#FF9800", self.mark_attendance_with_action, False)
        self.button_layout.addWidget(self.attendance_button)

        self.summary_button = self.create_button("View Attendance Summary", "#9C27B0", self.view_attendance_summary)
        self.button_layout.addWidget(self.summary_button)

        self.layout.addLayout(self.button_layout)

        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("background-color: #333; color: white; font-size: 14px;")
        self.layout.addWidget(self.status_bar)

        self.setLayout(self.layout)

    def create_button(self, text, color, func, enabled=True):
        """Creates a button with hover effect."""
        button = QPushButton(text, self)
        button.setStyleSheet(f"""
            background-color: {color}; color: white; padding: 14px; font-size: 16px; border-radius: 8px; border: none;
            transition: background-color 0.3s;
        """)
        button.setEnabled(enabled)
        button.clicked.connect(func)
        return button

    def start_recognition(self):
        """Start the face recognition process."""
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Camera Error", "Failed to open camera.")
            return

        self.status_bar.showMessage("Camera started. Ready for face recognition.")
        self.recognize_button.setEnabled(False)
        self.register_button.setEnabled(True)
        self.attendance_button.setEnabled(True)
        self.timer.start(50)

    def update_frame(self):
        """Update video frame and run face recognition."""
        ret, frame = self.video_capture.read()
        if not ret:
            logging.error("Failed to capture frame.")
            return
        
        self.frame = frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)  # Apply Histogram Equalization for better contrast

        faces = detector(gray_frame)
        logging.info(f"Number of faces detected: {len(faces)}")  # Log the number of faces detected

        recognized_name = "Unknown"
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = frame[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))  # Resize for comparison

            # Extract features and compare with known faces
            face_features = extract_face_features(face_roi_resized)
            recognized_name = compare_faces(face_features)

            if recognized_name != "Unknown":
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box for recognized
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red box for unknown

            cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.show_image(frame)

    def show_image(self, frame):
        """Display the current frame on the label."""
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        resized_frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

    def update_label(self, text):
        """Update the status label."""
        self.status_bar.showMessage(text)

    def register_face(self):
        """Register a new face.""" 
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name.")
            return

        if self.frame is None or not self.frame.any():
            QMessageBox.warning(self, "Error", "No face detected.")
            return

        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)  # Apply Histogram Equalization for better contrast
        faces = detector(gray_frame)

        if len(faces) > 0:
            face = faces[0]  # Take the first detected face
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = self.frame[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))  # Resize for storage
            known_faces.append(face_roi_resized)
            known_names.append(name)
            cv2.imwrite(os.path.join(KNOWN_FACES_FOLDER, f"{name}.jpg"), face_roi)
            QMessageBox.information(self, "Registration Success", f"Face of {name} registered successfully!")
        else:
            QMessageBox.warning(self, "Error", "No face detected in the current frame.")

    def upload_image(self):
        """Allow the user to upload an image for face registration.""" 
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.jpeg *.png)", options=options)

        if file_name:
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Input Error", "Please enter a name.")
                return

            img = cv2.imread(file_name)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Take the first detected face
                face_roi = img[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi, (100, 100))  # Resize for storage
                known_faces.append(face_roi_resized)
                known_names.append(name)
                cv2.imwrite(os.path.join(KNOWN_FACES_FOLDER, f"{name}.jpg"), face_roi)
                QMessageBox.information(self, "Registration Success", f"Face of {name} registered successfully!")
            else:
                QMessageBox.warning(self, "Error", "No face detected in the uploaded image.")

    def mark_attendance_with_action(self):
        """Mark attendance with entry or exit action.""" 
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name.")
            return

        if name not in known_names:
            QMessageBox.warning(self, "Error", f"No face recognized for {name}.")
            return

        action = self.action_combo.currentText()
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_record = pd.DataFrame({"Name": [name], "Action": [action], "Time": [time]})
        if os.path.exists(attendance_file):
            existing_data = pd.read_csv(attendance_file)
            updated_data = pd.concat([existing_data, new_record], ignore_index=True)
            updated_data.to_csv(attendance_file, index=False)
        else:
            new_record.to_csv(attendance_file, index=False)

        self.update_label(f"Attendance marked for {name} - {action} at {time}")

    def view_attendance_summary(self):
        """View attendance summary.""" 
        if os.path.exists(attendance_file):
            data = pd.read_csv(attendance_file)
            QMessageBox.information(self, "Attendance Summary", data.to_string(index=False))
        else:
            QMessageBox.warning(self, "No Data", "No attendance records found.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())





