import { useRef, useState, useEffect, useCallback } from "react";
import * as ort from "onnxruntime-web";

const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

function centerDigit(imageData, width, height) {
  const pixels = imageData.data;

  let minX = width, minY = height, maxX = 0, maxY = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (pixels[(y * width + x) * 4] > 20) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (minX > maxX) return new Float32Array(28 * 28);

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(
    imageData.canvas || document.createElement("canvas"),
    minX, minY, cropW, cropH,
    0, 0, cropW, cropH
  );

  const targetSize = 20;
  let newW, newH;
  if (cropW > cropH) {
    newW = targetSize;
    newH = Math.max(1, Math.round((cropH / cropW) * targetSize));
  } else {
    newH = targetSize;
    newW = Math.max(1, Math.round((cropW / cropH) * targetSize));
  }

  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = 28;
  resizeCanvas.height = 28;
  const resizeCtx = resizeCanvas.getContext("2d");
  resizeCtx.fillStyle = "#000";
  resizeCtx.fillRect(0, 0, 28, 28);

  const offsetX = Math.floor((28 - newW) / 2);
  const offsetY = Math.floor((28 - newH) / 2);
  resizeCtx.drawImage(cropCanvas, 0, 0, cropW, cropH, offsetX, offsetY, newW, newH);

  const finalData = resizeCtx.getImageData(0, 0, 28, 28).data;
  const result = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    result[i] = (finalData[i * 4] / 255 - MNIST_MEAN) / MNIST_STD;
  }
  return result;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map((v) => v / sum);
}

export default function MnistDemo() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [session, setSession] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 280, 280);
    ctx.strokeStyle = "#FFF";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ort.InferenceSession.create("/mnist.onnx").then((s) => {
      setSession(s);
      setLoading(false);
    });
  }, []);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scale = 280 / rect.width;
    const src = e.touches ? e.touches[0] : e;
    return { x: (src.clientX - rect.left) * scale, y: (src.clientY - rect.top) * scale };
  };

  const startDraw = (e) => {
    e.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const ctx = canvasRef.current.getContext("2d");
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const stopDraw = (e) => {
    if (e) e.preventDefault();
    setIsDrawing(false);
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 280, 280);
    ctx.strokeStyle = "#FFF";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    setPrediction(null);
  };

  const predict = useCallback(async () => {
    if (!session) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, 280, 280);

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 280;
    tempCanvas.height = 280;
    tempCanvas.getContext("2d").putImageData(imageData, 0, 0);
    imageData.canvas = tempCanvas;

    const input = centerDigit(imageData, 280, 280);
    const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);

    const results = await session.run({ input: tensor });
    const scores = Array.from(results.output.data);
    const probs = softmax(scores);
    const best = probs.indexOf(Math.max(...probs));

    setPrediction({
      digit: best,
      confidence: probs[best],
      all: probs.map((p, i) => ({ digit: i, probability: p })),
    });
  }, [session]);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 700, margin: "0 auto", padding: "2rem 1rem" }}>
      <h2 style={{ textAlign: "center", marginBottom: 4 }}>MNIST Digit Recognition</h2>
      <p style={{ textAlign: "center", color: "#888", fontSize: "0.85rem", marginBottom: "2rem" }}>
        Java + DJL + PyTorch &mdash; running in your browser via ONNX Runtime
      </p>

      {loading && <p style={{ textAlign: "center", color: "#aaa" }}>Loading model...</p>}

      <div style={{ display: "flex", gap: 32, justifyContent: "center", flexWrap: "wrap", alignItems: "flex-start" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            style={{ border: "2px solid #333", borderRadius: 10, cursor: "crosshair", touchAction: "none", width: 280, height: 280 }}
            onMouseDown={startDraw}
            onMouseMove={draw}
            onMouseUp={stopDraw}
            onMouseLeave={stopDraw}
            onTouchStart={startDraw}
            onTouchMove={draw}
            onTouchEnd={stopDraw}
          />
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={predict} disabled={!session} style={btnStyle("#3b82f6", "#fff")}>
              Predict
            </button>
            <button onClick={clear} style={btnStyle("#272727", "#ccc")}>
              Clear
            </button>
          </div>
        </div>

        {prediction && (
          <div style={{ minWidth: 220, animation: "fadeIn 0.3s" }}>
            <div style={{ fontSize: "4.5rem", fontWeight: 800, color: "#3b82f6", lineHeight: 1 }}>
              {prediction.digit}
            </div>
            <div style={{ color: "#888", fontSize: "0.85rem", marginBottom: 16 }}>
              {(prediction.confidence * 100).toFixed(2)}% confidence
            </div>
            {prediction.all.map(({ digit, probability }) => (
              <div key={digit} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <span style={{ width: 14, textAlign: "right", fontSize: "0.85rem", color: digit === prediction.digit ? "#3b82f6" : "#888", fontWeight: digit === prediction.digit ? 700 : 400 }}>
                  {digit}
                </span>
                <div style={{ flex: 1, height: 6, background: "#1a1a1a", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ width: `${probability * 100}%`, height: "100%", background: "#3b82f6", borderRadius: 3, transition: "width 0.3s" }} />
                </div>
                <span style={{ width: 48, textAlign: "right", fontSize: "0.75rem", color: "#888" }}>
                  {(probability * 100).toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const btnStyle = (bg, color) => ({
  padding: "8px 20px",
  border: "none",
  borderRadius: 6,
  fontSize: "0.9rem",
  fontWeight: 600,
  cursor: "pointer",
  background: bg,
  color,
});
