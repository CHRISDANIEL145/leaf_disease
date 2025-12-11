# 🌿 LeafAI - AI-Driven Crop Disease Prediction

A **futuristic, modern web interface** for plant disease detection using deep learning.

![LeafAI](https://img.shields.io/badge/AI-Powered-00d4aa?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Backend-00ff88?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-ee4c2c?style=for-the-badge&logo=pytorch)

---

## ✨ Features

- 🧠 **Deep Learning Models** - 4 specialized models (LGNM, ADSD, S-ViT Lite, ECAM)
- 🎯 **96%+ Accuracy** - Trained on 87,000+ plant leaf images
- ⚡ **Instant Analysis** - Sub-second disease detection
- 🎨 **Futuristic UI** - Glassmorphism, neon effects, smooth animations
- 📊 **Dashboard** - Real-time statistics and charts
- 📱 **Responsive** - Works on desktop, tablet, and mobile

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Navigate to the webapp directory:**
   ```bash
   cd "d:\leaf disease\webapp"
   ```

2. **Install dependencies:**
   ```bash
   pip install flask torch torchvision pillow
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   ```
   http://localhost:5000
   ```

---

## 📁 Project Structure

```
webapp/
├── app.py                  # Flask backend
├── README.md               # Documentation
├── templates/
│   ├── base.html          # Base template with nav/footer
│   ├── index.html         # Home page
│   ├── predict.html       # Upload & prediction page
│   ├── about.html         # About technology page
│   └── dashboard.html     # Results dashboard
└── static/
    ├── css/
    │   └── styles.css     # Custom CSS (glassmorphism, animations)
    ├── js/
    │   └── main.js        # JavaScript (interactions, utilities)
    ├── uploads/           # Uploaded images
    └── images/            # Static images
```

---

## 🎨 UI Pages

### 1. Home Page (`/`)
- Animated hero section with gradient
- Feature cards with hover effects
- Statistics showcase
- Call-to-action buttons

### 2. Predict Page (`/predict`)
- Drag-and-drop image upload
- Real-time image preview
- AI analysis with confidence bar
- Disease details and treatment recommendations

### 3. Dashboard (`/dashboard`)
- Statistics cards
- Confidence distribution chart (Chart.js)
- Recent predictions list
- Full predictions table

### 4. About Page (`/about`)
- 4 model architecture cards
- Dataset information
- Technology stack

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | GET | Upload page |
| `/predict` | POST | Submit image for prediction |
| `/dashboard` | GET | Dashboard page |
| `/about` | GET | About page |
| `/api/predictions` | GET | JSON array of recent predictions |
| `/api/stats` | GET | JSON statistics object |

---

## 🎯 Using Your Trained Model

To use your custom trained model:

1. Open `app.py`
2. Find the `load_model()` function
3. Update `model_paths` list with your model path:
   ```python
   model_paths = [
       '../model_files/adsd/best_model.pth',  # Your model path
       # ... other paths
   ]
   ```

---

## 🎨 Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Neon Teal | `#00d4aa` | Primary accent |
| Neon Green | `#00ff88` | Success states |
| Neon Cyan | `#00e5ff` | Highlights |
| Neon Purple | `#a855f7` | Secondary accent |
| Space Black | `#0a0a0f` | Background |

---

## 📦 Dependencies

- **Flask** - Web framework
- **PyTorch** - Deep learning
- **TorchVision** - Image transforms
- **Pillow** - Image processing
- **Tailwind CSS** - Styling (CDN)
- **Lucide Icons** - Icons (CDN)
- **Chart.js** - Charts (CDN)
- **AOS** - Scroll animations (CDN)

---

## 🖥️ System Requirements

- **CPU**: Any modern processor
- **GPU**: Optional (CUDA for faster inference)
- **RAM**: 4GB minimum
- **Browser**: Chrome, Firefox, Safari, Edge

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Made with 💚 for Agriculture**
