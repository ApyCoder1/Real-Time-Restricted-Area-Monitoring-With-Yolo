import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import random

video_path = 'v.mp4'

class PPEApp:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.cap = None
        self.alert_played = False
        self.class_colors = {}
        self.restricted_area = None  # For defining the restricted region

    def load_models(self, model_paths):
        """
        Load multiple YOLO models.
        """
        for model_name, path in model_paths.items():
            self.models[model_name] = YOLO(path)
        self.current_model = self.models["Intrusion"]  # Set 'Intrusion' as the default model

    def generate_class_colors(self, model):
        """
        Generate a unique random color for each class in the given model.
        """
        colors = {}
        for class_id in model.names:
            colors[model.names[class_id]] = tuple(
                random.randint(0, 255) for _ in range(3)
            )
        return colors

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)  # 0 for default
        if not self.cap.isOpened():
            st.error("Error: Unable to access the webcam.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None

    def play_alert_sound(self, sound_path):
        try:
            from playsound import playsound
            playsound(sound_path)
        except Exception as e:
            print(f"Error playing sound: {e}")

    def draw_roi(self, frame):
        """
        Draw a custom region of interest (ROI) in the shape of an ellipse on the frame.
        """
        if self.current_model == self.models["Intrusion"]:  # Check if current model is 'best.pt'
            height, width, _ = frame.shape
            center = (width // 2, height // 2)  # Center of the frame
            axes = (width // 4, height // 8)  # Width and height of the ellipse
            angle = 0  # No rotation
            startAngle = 0
            endAngle = 360
            color = (0, 0, 255)  # Red color for the ROI
            thickness = 2
            self.restricted_area = (center, axes)  # Save the restricted area
            frame = cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, color, thickness)
        return frame

    def is_near_restricted_area(self, box):
        """
        Check if the bounding box of any detected object is near the restricted area.
        This function checks if any part of the box is near the restricted area (within a distance threshold).
        """
        if self.restricted_area:
            center, axes = self.restricted_area
            x1, y1, x2, y2 = box
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = np.linalg.norm(np.array(center) - np.array(object_center))  # Distance from center
            return distance < (min(axes) + 50)  # Threshold distance for being "near" the restricted area
        return False

    def update_frame(self, model, confidence_threshold, selected_classes, alert_classes):
        if not self.cap:
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        results = model(frame, conf=confidence_threshold, iou=0.3)
        detected_classes = []
        annotated_frame = frame.copy()
        near_restricted_area = False  # Track if any object is near the restricted area
        alert_triggered_for_current_frame = False  # Flag to avoid repeated alerts in the same frame

        for result in results[0].boxes:
            class_id = int(result.cls)
            class_name = model.names[class_id]

            if class_name in selected_classes:
                detected_classes.append(class_name)
                color = self.class_colors.get(class_name, (0, 255, 0))
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                conf = result.conf[0]
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # Check if the object is near the restricted area
                if self.is_near_restricted_area([x1, y1, x2, y2]):
                    near_restricted_area = True
                    cv2.putText(annotated_frame, "Object near restricted area!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    # Trigger alert if the detected class is in the alert classes
                    if class_name in alert_classes and not self.alert_played:
                        self.alert_played = True
                        threading.Thread(target=self.play_alert_sound, args=("alert.mp3",), daemon=True).start()
                        alert_triggered_for_current_frame = True

        # Reset alert if no object is detected in the alert classes
        if not any(class_name in alert_classes for class_name in detected_classes):
            self.alert_played = False

        # Draw ROI if the current model is 'best.pt'
        annotated_frame = self.draw_roi(annotated_frame)

        return annotated_frame, detected_classes, near_restricted_area, alert_triggered_for_current_frame

    def run(self):
        st.set_page_config(page_title="Real-Time Object Monitoring System", layout="wide")

        st.markdown("<h2>🛡️ Real-Time Restricted Area Monitoring</h2>", unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align: center;'>Switch between models and monitor real-time detections.</p>",
            unsafe_allow_html=True,
        )

        st.sidebar.title("🔧 Setting")
        model_paths = {
            "Intrusion": "best.pt",
           
        }
        selected_model = st.sidebar.selectbox("Select Model", options=model_paths.keys())
        if self.current_model != self.models.get(selected_model):
            self.current_model = self.models[selected_model]
            self.class_colors = self.generate_class_colors(self.current_model)

        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
        available_classes = list(self.current_model.names.values())
        selected_classes = st.sidebar.multiselect("Objects to Detect", available_classes, default=[])

        # Update alert_classes options dynamically
        if selected_classes:
            alert_classes = st.sidebar.multiselect("Objects for Alert Sound", available_classes, default=[])
        else:
            alert_classes = st.sidebar.multiselect("Objects for Alert Sound", ["None"] + available_classes, default=[])

        if "None" in alert_classes:
            alert_classes.remove("None")

        start_button = st.sidebar.button("▶️ Start Webcam")
        stop_button = st.sidebar.button("⏹️ Stop Webcam")

        if start_button:
            if not self.cap:
                with st.spinner("Starting the webcam..."):
                    if self.start_webcam():
                        st.success("Webcam started successfully!")
            else:
                st.warning("Webcam is already running.")

        if stop_button:
            if self.cap:
                with st.spinner("Stopping the webcam..."):
                    self.stop_webcam()
                    st.success("Webcam stopped.")

        if self.cap:
            st.subheader("📺 Live Video Feed")
            frame_placeholder = st.empty()
            detection_info = st.empty()

            while self.cap.isOpened():
                result = self.update_frame(self.current_model, confidence_threshold, selected_classes, alert_classes)
                if result:
                    frame, detected_classes, near_restricted_area, alert_triggered_for_current_frame = result
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", caption="🔍 Real-time Detection")
                        detection_info.write(f"Detected Classes: {', '.join(detected_classes) if detected_classes else 'None'}")
                        if near_restricted_area:
                            detection_info.write("⚠️ Object near restricted area!")

                        # If the alert was triggered in the current frame, reset alert_played for the next frame
                        if alert_triggered_for_current_frame:
                            self.alert_played = False

if __name__ == "__main__":
    app = PPEApp()
    app.load_models({
        "Intrusion": "best.pt",  # Replace with actual paths
       
       
    })
    app.run()
