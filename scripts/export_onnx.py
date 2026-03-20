import json, base64, os, struct
import numpy as np
import torch
import torch.nn as nn
import requests

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

resp = requests.get("http://localhost:8080/api/export")
resp.raise_for_status()
params = resp.json()["parameters"]

model = MnistCNN()
state = model.state_dict()
keys = list(state.keys())

for i, key in enumerate(keys):
    raw = base64.b64decode(params[i]["data"])
    arr = np.frombuffer(raw, dtype=np.float32).reshape(params[i]["shape"])
    state[key] = torch.tensor(arr)

model.load_state_dict(state)
model.eval()

os.makedirs("export", exist_ok=True)
torch.onnx.export(model, torch.randn(1, 1, 28, 28), "export/mnist.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                  opset_version=13)
print(f"Exported to export/mnist.onnx ({os.path.getsize('export/mnist.onnx') / 1024:.0f} KB)")
