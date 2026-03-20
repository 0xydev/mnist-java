# MNIST Java

![Java](https://img.shields.io/badge/Java-17-ED8B00?style=flat&logo=openjdk&logoColor=white)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4-6DB33F?style=flat&logo=springboot&logoColor=white)
![DJL](https://img.shields.io/badge/DJL-0.36.0-FF6F00?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)
![ONNX](https://img.shields.io/badge/ONNX-Runtime%20Web-005CED?style=flat&logo=onnx&logoColor=white)

MNIST handwritten digit recognition in Java. Training, API, frontend and a browser demo all in one repo.

> [Turkce versiyon](README.md)

## Architecture

```
┌───────────────────────────────────────────────────────┐
│  Training (Java + DJL)                                │
│  MNIST Dataset --> CNN (Conv2d->ReLU->MaxPool) x 2    │
│                --> Flatten -> Dense(128) -> Dense(10)  │
│  Augmentation: +-3px shift, +-15 degree rotation      │
│  Normalization: mean=0.1307, std=0.3081               │
└─────────────────────┬─────────────────────────────────┘
                      | model weights
         ┌────────────┼────────────┐
         v            v            v
  ┌────────────┐ ┌─────────┐ ┌──────────────┐
  │  REST API  │ │  ONNX   │ │   Frontend   │
  │  Spring    │ │  Export  │ │   React +    │
  │  Boot      │ │  Script  │ │   Vite       │
  │  :8080     │ │          │ │   :3000      │
  └────────────┘ └────┬─────┘ └──────────────┘
                      v
              ┌──────────────┐
              │ Static Site  │
              │ (Astro etc.) │
              │ ONNX Runtime │
              │ Web          │
              └──────────────┘
```

## Quick Start

```bash
git clone https://github.com/0xydev/mnist-java.git
cd mnist-java
./mvnw spring-boot:run
```

Once the app is up, train the model:

```bash
curl -X POST http://localhost:8080/api/train
```

On first run it downloads the MNIST dataset automatically (~60MB). Training takes about 3-4 minutes and reaches ~99% test accuracy.

Predict:

```bash
curl -X POST http://localhost:8080/api/predict -F "image=@digit.png"
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Draw digits on a canvas at `http://localhost:3000` and get instant predictions. Vite proxies API requests to port 8080.

### ONNX Export

To run the model in the browser, export it to ONNX format. While the backend is running:

```bash
pip install torch onnx requests numpy
python scripts/export_onnx.py
```

This creates `export/mnist.onnx`. Drop it into any static site's `public/` folder and use it with `onnxruntime-web`. A ready-made React component is included at `astro-component/MnistDemo.jsx`.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/train` | Trains the model |
| `POST` | `/api/predict` | Predicts digit from uploaded image |
| `GET`  | `/api/export` | Returns model weights as JSON |

### Example response: `POST /api/predict`

```json
{
  "prediction": "7",
  "confidence": "99.83%",
  "processedImage": "data:image/png;base64,...",
  "top5": [
    {"digit": "7", "probability": "0.9983"},
    {"digit": "9", "probability": "0.0012"}
  ]
}
```

`processedImage` returns the 28x28 preprocessed image as base64 PNG. Useful for debugging in the frontend.

## Model

| | |
|---|---|
| Architecture | CNN: 2x (Conv2d + ReLU + MaxPool), 2x Linear |
| Parameters | ~421K |
| Dataset | MNIST (60K train, 10K test) |
| Epochs | 10 |
| Test accuracy | ~99% |
| Augmentation | Random shift (+-3px), rotation (+-15 degrees) |

### Preprocessing

User drawings are converted to MNIST format before inference:

1. Convert to grayscale
2. Find digit via bounding box, crop
3. Resize to 20x20 preserving aspect ratio
4. Center in 28x28 frame (4px padding)
5. Normalize (mean=0.1307, std=0.3081)

The same preprocessing runs in both the API and the browser ONNX demo.

## Project Structure

```
mnist-java/
├── pom.xml
├── src/main/java/com/adikti/mnist/
│   ├── MnistApplication.java
│   ├── controller/
│   │   └── MnistController.java
│   ├── service/
│   │   ├── TrainingService.java
│   │   └── PredictionService.java
│   └── model/
│       └── RandomAugmentation.java
├── frontend/
│   ├── src/App.jsx
│   └── vite.config.js
├── astro-component/
│   └── MnistDemo.jsx
├── scripts/
│   └── export_onnx.py
└── export/
    └── mnist.onnx
```

## Requirements

- Java 17+
- Maven 3.9+ (wrapper included)
- Node.js 18+ (for frontend)
- Python 3.10+ (ONNX export only)

## License

MIT
