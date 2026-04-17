# Facial-Recognition

# YOLO Face Detection System 🎯

A real-time object/face detection system powered by **YOLO (Ultralytics)** with a **Flask-based web interface** and **Text-to-Speech (TTS) voice alerts**. This project enables live detection from webcam/video streams with interactive controls and audio feedback.

---

## 🚀 Features

* 🔍 Real-time object/face detection using YOLO
* 🎥 Live video streaming via Flask
* 🔊 Voice alerts using Text-to-Speech (pyttsx3)
* 🌐 Web-based control interface
* 🎯 Confidence threshold filtering
* 🔇 Mute/Unmute voice feedback
* 🖥️ Standalone CLI detection mode
* 📷 Multi-source support (webcam, video, images, folders)

---

## 📁 Project Structure

```
.
├── run.py              # Flask web server (routes + streaming)
├── detector.py         # Core detection logic (YOLO + TTS)
├── yo.py               # CLI-based detection script
├── v.py                # Camera index scanner
├── templates/
│   └── index.html      # Web UI (user provided)
├── my_model.pt         # YOLO trained model (user provided)
└── README.md
```

---

## ⚙️ Requirements

* Python 3.8 or higher
* pip package manager
* Webcam / video source
* YOLO trained model (.pt file)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

pip install ultralytics opencv-python flask pyttsx3 numpy
```

---

## 🧠 Model Setup

Place your YOLO model file in the root directory:

```
my_model.pt
```

Or update the model path inside `run.py`:

```python
det = Detector(model_path="my_model.pt", source="0", thresh=0.5, cooling=3.0)
```

---

## 📷 Step 1 — Find Camera Index

```bash
python v.py
```

Example:

```
Camera found at index 0
Camera found at index 1
```

---

## 🌐 Step 2 — Run Web Application

```bash
python run.py
```

Open your browser:

```
http://localhost:5000
```

---

## 🎮 Web Controls

| Route         | Function                     |
| ------------- | ---------------------------- |
| `/`           | Home page (video + controls) |
| `/start`      | Start detection              |
| `/stop`       | Stop detection               |
| `/mute`       | Toggle voice alerts          |
| `/video_feed` | Live MJPEG video stream      |

---

## 🖥️ CLI Mode (Optional)

Run detection without Flask:

```bash
python yo.py --model my_model.pt --source 0
```

### Arguments

| Argument       | Description                  |
| -------------- | ---------------------------- |
| `--model`      | Path to YOLO model           |
| `--source`     | Camera index / video / image |
| `--thresh`     | Confidence threshold         |
| `--resolution` | Resize (e.g., 640x480)       |
| `--record`     | Save output video            |

---

## 🔊 Voice Alert System

* Powered by **pyttsx3**
* Automatically selects female voice (if available)
* Adjustable speech rate
* Cooling delay to prevent repeated announcements
* Runs asynchronously using threading

---

## 🛠️ Troubleshooting

### Camera not detected

```bash
python v.py
```

* Ensure camera is not used by another application

### Model file not found

* Place `my_model.pt` in project root
* Or update path in `run.py`

### No voice output

```bash
pip install pyttsx3
```

Linux:

```bash
sudo apt-get install espeak ffmpeg libespeak1
```

### Low FPS / Lag

* Reduce resolution (e.g., 320x240)
* Use smaller YOLO model
* Close background applications

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.
Do not use for surveillance or identification without proper consent.

---

## 👨‍💻 Author

Your Name
(Add GitHub / LinkedIn if needed)

---

## ⭐ Contributions

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---
