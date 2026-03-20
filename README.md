# MNIST Java

![Java](https://img.shields.io/badge/Java-17-ED8B00?style=flat&logo=openjdk&logoColor=white)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4-6DB33F?style=flat&logo=springboot&logoColor=white)
![DJL](https://img.shields.io/badge/DJL-0.36.0-FF6F00?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)
![ONNX](https://img.shields.io/badge/ONNX-Runtime%20Web-005CED?style=flat&logo=onnx&logoColor=white)

Java ile MNIST el yazısı rakam tanıma. Eğitim, API, frontend ve tarayıcıda çalışan demo hepsi bu repoda.

> [English version](README.en.md)

## Mimari

```
┌───────────────────────────────────────────────────────┐
│  Eğitim (Java + DJL)                                  │
│  MNIST Dataset --> CNN (Conv2d->ReLU->MaxPool) x 2    │
│                --> Flatten -> Dense(128) -> Dense(10)  │
│  Augmentation: +-3px kayma, +-15 derece döndürme      │
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
              │ Statik Site  │
              │ (Astro vb.)  │
              │ ONNX Runtime │
              │ Web          │
              └──────────────┘
```

## Hızlı Başlangıç

```bash
git clone https://github.com/0xydev/mnist-java.git
cd mnist-java
./mvnw spring-boot:run
```

Uygulama ayağa kalktıktan sonra modeli eğitmek için:

```bash
curl -X POST http://localhost:8080/api/train
```

İlk çalıştırmada MNIST veri setini otomatik indirir (~60MB). Eğitim 3-4 dakika sürer, %99 civarında test doğruluğu elde edilir.

Tahmin:

```bash
curl -X POST http://localhost:8080/api/predict -F "image=@digit.png"
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

`http://localhost:3000` adresinde canvas üzerinde rakam çizebilir, anlık tahmin alabilirsiniz. Vite, API isteklerini 8080 portuna yönlendirir.

### ONNX Export

Modeli tarayıcıda çalıştırmak için ONNX formatına çevirebilirsiniz. Backend çalışırken:

```bash
pip install torch onnx requests numpy
python scripts/export_onnx.py
```

`export/mnist.onnx` dosyası oluşur. Herhangi bir statik sitenin `public/` klasörüne koyup `onnxruntime-web` ile kullanabilirsiniz. `astro-component/MnistDemo.jsx` buna hazır bir React componenti içerir.

## API

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/api/train` | Modeli eğitir |
| `POST` | `/api/predict` | Resimden rakam tahmini yapar |
| `GET`  | `/api/export` | Model ağırlıklarını JSON olarak döner |

### Örnek yanıt: `POST /api/predict`

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

`processedImage` modelin girdi olarak aldığı 28x28 resmi base64 PNG olarak döndürür. Frontend'de debug için görüntülenebilir.

## Model

| | |
|---|---|
| Mimari | CNN: 2x (Conv2d + ReLU + MaxPool), 2x Linear |
| Parametre | ~421K |
| Veri seti | MNIST (60K eğitim, 10K test) |
| Epoch | 10 |
| Test doğruluğu | ~%99 |
| Augmentation | Rastgele kayma (+-3px), döndürme (+-15 derece) |

### Ön İşleme

Kullanıcının çizdiği resim modele verilmeden önce MNIST formatına dönüştürülür:

1. Gri tonlamaya çevrilir
2. Bounding box ile rakam bulunup kırpılır
3. En-boy oranı korunarak 20x20'ye küçültülür
4. 28x28 karenin ortasına yerleştirilir (4px boşluk)
5. Normalize edilir (mean=0.1307, std=0.3081)

Bu işlem hem API'de hem tarayıcı tarafındaki ONNX demoda aynı şekilde uygulanır.

## Proje Yapısı

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

## Gereksinimler

- Java 17+
- Maven 3.9+ (wrapper dahil)
- Node.js 18+ (frontend için)
- Python 3.10+ (sadece ONNX export için)

## Lisans

MIT
