# Real-Time Restricted Area Monitoring System

## Prerequisites
Ensure you have Python 3.12 or later installed. You can check your version using:
```sh
python --version
```
If Python is not installed, download and install it from [Python's official website](https://www.python.org/downloads/).

## Step 1: Create a Virtual Environment (Recommended)
To avoid conflicts, create a virtual environment:
```sh
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

## Step 2: Install Dependencies
Manually install the required libraries:
```sh
pip install streamlit opencv-python numpy ultralytics playsound
```

## Step 3: Download the YOLO Model
Ensure you have the required YOLO model (`best.pt`). Place it in the same directory as the script or provide its correct path.

## Step 4: Running the Application
Start the Streamlit application:
```sh
streamlit run main.py
```
Replace `main.py` with the actual script name.

## Troubleshooting
- If `cv2` installation fails, try installing it using:
  ```sh
  pip install opencv-python-headless
  ```
- If `playsound` does not work, use `pip install simpleaudio` as an alternative sound-playing library.
- Ensure you have the correct permissions for webcam access.

## Notes
- The application requires a webcam for real-time monitoring.
- You can customize detection classes and alert settings in the sidebar.

## Additional Resources
- **YouTube Channel:** [ApyCoder](https://www.youtube.com/@ApyCoder)
- **Website:** [ApyCoder.com](https://www.apycoder.com)

Enjoy real-time monitoring with AI-powered detection! 🚀

