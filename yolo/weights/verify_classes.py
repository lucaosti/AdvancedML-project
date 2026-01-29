from ultralytics import YOLO

model = YOLO('yolo/weights/fine-tuned-yolo-weights.pt')

print("--- Model Class Mapping ---")
print(model.names)