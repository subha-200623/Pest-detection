""" 

# AI-Based Pest Detection using TensorFlow and IoT (Web Camera)

## 🌱 Project Overview

This project leverages **Artificial Intelligence (TensorFlow)** and **IoT (Web Camera)** to detect pests on plants in real-time. When a plant is shown in front of the web camera, the system analyzes the visual feed, detects the presence of pests (if any), and identifies the type of pest using a trained deep learning model.

It aims to support farmers and agriculture researchers in early detection and diagnosis of pest-related problems, reducing crop loss and improving yield quality.

---

## 🚀 Key Features

- Real-time pest detection using live webcam feed
- Identification of pest species by name
- Trained on custom pest dataset using TensorFlow
- Lightweight for deployment on IoT devices (e.g., Raspberry Pi)
- User-friendly interface and easy-to-understand output

---

## 🧰 Tools & Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV (for camera integration)
- NumPy, Matplotlib
- Jupyter Notebook (for model training)
- Webcam (for real-time video capture)

---

## 📁 Folder Structure

```
AI-Pest-Detection/
│
├── model/                  # Trained model files (.h5)
├── images/                 # Sample pest images
├── utils/                  # Helper functions
├── main.py                 # Main execution script
├── train_model.ipynb       # Model training notebook
├── read.py                 # Step-by-step execution code
└── README.md               # Project documentation (this file)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Pest-Detection.git
   cd AI-Pest-Detection
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the following are connected:**
   - Webcam or USB camera
   - Trained model (`.h5` file) inside the `model/` folder

---

## ▶️ Step-by-Step Execution Guide

1. **Train the model (if needed)**  
   Run the notebook to train and save the pest detection model:
   ```
   train_model.ipynb
   ```

2. **Start the Pest Detection System**
   Run the main script:
   ```bash
   python main.py
   ```

3. **Webcam will open. Show a plant leaf in front of the camera.**

4. **The system will:**
   - Capture the image frame
   - Preprocess the input
   - Run prediction using TensorFlow
   - Display the name of the pest (if detected)

---

## 🖼️ Example Output

```
Pest Detected: Aphid
Confidence: 98.5%
```

_Image will be displayed in a pop-up window with bounding boxes (if applicable)._

---

## 🔄 Future Enhancements

- Mobile app integration for farmers
- Integration with cloud database for pest reports
- Adding more pest classes for larger-scale identification
- Multilingual voice-based alerts

---

## 🤝 Contributing

Feel free to fork the repo and submit a Pull Request (PR). Contributions are always welcome.

---

## 📜 License

This project is open-source and available under the MIT License.

"""
