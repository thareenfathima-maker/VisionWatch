# 👁️ VisionWatch
### AI-Powered Real-Time Industrial Safety Monitoring System

VisionWatch is an AI-powered computer vision application designed to improve workplace safety through real-time monitoring and automated detection.

The system uses computer vision and deep learning to identify safety violations and monitor restricted areas.

## 🚀 Features

- **Person Detection:** Detects people through a camera feed.
- **Helmet Detection:** Identifies whether workers are wearing safety helmets.
- **Restricted Zone Monitoring:** Detects unauthorized entry into predefined restricted areas.
- **Real-Time Monitoring:** Processes live camera input.
- **Automated Alerts:** Generates alerts for detected safety violations.
- **Event Logging:** Records detected safety events.
- **Dashboard:** Provides a visual interface for monitoring detection events.
- **Evidence Capture:** Supports capturing evidence of detected events.

## 🛠️ Technologies Used

- Python
- OpenCV
- YOLO11
- Computer Vision
- Deep Learning
- Streamlit (if used in the dashboard)
- Pandas

## 📂 Project Structure

```text
VisionWatch/
│
├── app.py
├── app_backup.py
├── dashboard_test.py
├── requirements.txt
├── yolo11n.pt
│
├── models/
│   └── best.pt
│
├── src/
│   ├── camera.py
│   ├── detector.py
│   ├── event_logger.py
│   ├── helmet_detection.py
│   ├── person_detection.py
│   ├── restricted_zone.py
│   └── zone_detection.py
│
├── tests/
│
├── evidence/
│
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/thareenfathima-maker/VisionWatch.git
```

2. Navigate to the project directory:

```bash
cd VisionWatch
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application using its configured entry point.

## 🎯 Project Objective

The primary objective of VisionWatch is to reduce workplace safety violations through intelligent surveillance, automated detection and real-time monitoring.

## 🏆 Achievement

**Second Prize – Hackathon**

Developed as part of a team-based hackathon project.

## 👩‍💻 Team Members

- Thareen Fathima
- Varsha H D
- Dishashree P

## 🔮 Future Enhancements

- Improved detection accuracy.
- Advanced safety analytics.
- More intelligent alert mechanisms.
- Enhanced monitoring and reporting.

---

⭐ If you find this project interesting, consider starring the repository!