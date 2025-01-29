


import sys
import os
import cv2
import numpy as np
import logging
import requests
from datetime import datetime
from geopy.geocoders import Nominatim
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox, QStatusBar, QComboBox
import face_recognition
import pandas as pd

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)

# Define directories for saving known faces and attendance records
KNOWN_FACES_FOLDER = "known_faces"
UPLOAD_FOLDER = "uploads"
attendance_file = "attendance.csv"

os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize known faces
known_faces, known_names = [], []

def get_model_path():
    """Get the correct path to the shape predictor model."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'face_recognition_models', 'models', 'shape_predictor_68_face_landmarks.dat')
    return model_path

def load_face_recognition_model():
    """Load the shape predictor model."""
    model_path = get_model_path()
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        raise RuntimeError(f"Unable to find model at {model_path}")
    return model_path

def load_known_faces():
    """Load all known faces and their names from saved images."""
    known_faces.clear()
    known_names.clear()
    for filename in os.listdir(KNOWN_FACES_FOLDER):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(KNOWN_FACES_FOLDER, filename)
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                known_faces.append(face_encodings[0])
                known_names.append(filename.split('.')[0])
            else:
                logging.warning(f"No face found in {filename}")
    return known_faces, known_names

def get_geolocation():
    """Get the current geolocation (latitude, longitude, Place, and country)."""
    geolocator = Nominatim(user_agent="face_recognition_system")
    try:
        response = requests.get("http://ipinfo.io/json")
        data = response.json()
        location_str = data.get("loc", "").split(",")
        if len(location_str) == 2:
            latitude, longitude = map(float, location_str)
            location = geolocator.reverse((latitude, longitude), language='en')
            if location:
                return latitude, longitude, location.address
            else:
                return None, None, "Unknown Location"
        else:
            return None, None, "Unknown Location"
    except Exception as e:
        logging.error(f"Error fetching geolocation: {e}")
        return None, None, "Unknown Location"

def mark_attendance_in_csv(name, action="entry", status_callback=None):
    """Mark attendance in the CSV file."""
    logging.info(f"Marking attendance for {name} with action: {action}")

    if not os.path.exists(attendance_file):
        logging.info(f"{attendance_file} does not exist. Creating a new file.")
        with open(attendance_file, "w", newline="") as f:
            f.write("Serial Number,Name,Time,Action,Latitude,Longitude,Place,Country\n")

    today = datetime.now().strftime('%Y-%m-%d')
    try:
        df = pd.read_csv(attendance_file)
    except Exception as e:
        logging.error(f"Error reading attendance file: {e}")
        df = pd.DataFrame(columns=["Serial Number", "Name", "Time", "Action", "Latitude", "Longitude", "Place", "Country"])

    latitude, longitude, location = get_geolocation()
    Place, country = "", ""
    if location:
        location_parts = location.split(", ")
        if len(location_parts) >= 2:
            Place = location_parts[0]
            country = location_parts[-1]

    existing_entry = df[(df['Name'] == name) & (df['Time'].str.contains(today)) & (df['Action'] == action)]

    if not existing_entry.empty:
        message = f"Attendance already marked for {name} today with action {action}."
        logging.info(message)
        if status_callback:
            status_callback(message)
    else:
        serial_number = len(df) + 1
        new_entry = {
            "Serial Number": serial_number,
            "Name": name,
            "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Action": action,
            "Latitude": latitude if latitude else "",
            "Longitude": longitude if longitude else "",
            "Place": Place,
            "Country": country
        }

        new_entry_df = pd.DataFrame([new_entry])
        df = pd.concat([df, new_entry_df], ignore_index=True)

        logging.info(f"Attendance DataFrame before writing to CSV: \n{df.tail()}")

        try:
            df.to_csv(attendance_file, index=False)
            logging.info(f"Attendance marked for {name} with action {action}")
        except Exception as e:
            logging.error(f"Error writing to attendance file: {e}")

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

        # Load the face recognition model
        load_face_recognition_model()

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

        self.export_button = self.create_button("Export Attendance", "#FFEB3B", self.export_attendance)
        self.button_layout.addWidget(self.export_button)

        self.toggle_theme_button = self.create_button("Toggle Dark/Light Mode", "#FFC107", self.toggle_theme)
        self.button_layout.addWidget(self.toggle_theme_button)

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
            return
        self.frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(distances)
            name = "Unknown"

            if distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
                self.update_label(f"Recognized: {name}")
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box for recognized
            else:
                self.update_label("No face recognized")
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Red box for unknown

            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

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

        if not self.frame.any():
            QMessageBox.warning(self, "Error", "No face detected.")
            return

        rgb_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)
        if face_encodings:
            known_faces.append(face_encodings[0])
            known_names.append(name)
            cv2.imwrite(os.path.join(KNOWN_FACES_FOLDER, f"{name}.jpg"), self.frame)
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
            face_encodings = face_recognition.face_encodings(rgb_img)

            if face_encodings:
                known_faces.append(face_encodings[0])
                known_names.append(name)
                cv2.imwrite(os.path.join(KNOWN_FACES_FOLDER, f"{name}.jpg"), img)
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
        logging.info(f"Attempting to mark {name} with {action} action.")
        
        mark_attendance_in_csv(name, action, status_callback=self.update_label)
        QMessageBox.information(self, "Attendance", f"Attendance marked for {name} with action: {action}")

    def view_attendance_summary(self):
        """View attendance summary for all users."""
        try:
            df = pd.read_csv(attendance_file)
            summary = df.groupby('Name').agg({'Action': 'count', 'Time': 'max'}).reset_index()
            summary_str = summary.to_string(index=False)
            QMessageBox.information(self, "Attendance Summary", summary_str)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read attendance file: {e}")

    def export_attendance(self):
        """Export attendance records to a CSV file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Attendance", "", "CSV Files (*.csv);;Excel Files (*.xlsx)", options=options)
        if file_name:
            try:
                df = pd.read_csv(attendance_file)
                if file_name.endswith('.xlsx'):
                    df.to_excel(file_name, index=False)
                else:
                    df.to_csv(file_name, index=False)
                QMessageBox.information(self, "Export Success", "Attendance records exported successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not export attendance file: {e}")

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        current_style = self.styleSheet()
        if "background-color: #1e1e1e;" in current_style:
            self.setStyleSheet("background-color: #ffffff; color: #000000; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;")
            self.status_bar.setStyleSheet("background-color: #f0f0f0; color: black; font-size: 14px;")
        else:
            self.setStyleSheet("background-color: #1e1e1e; color: #f4f4f4; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;")
            self.status_bar.setStyleSheet("background-color: #333; color: white; font-size: 14px;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())






