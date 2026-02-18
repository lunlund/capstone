from ultralytics import YOLO

# Load the exported ONNX model
onnx_model = YOLO("yolo26n.engine")

# Run inference
results = onnx_model("images/bus.jpg")
