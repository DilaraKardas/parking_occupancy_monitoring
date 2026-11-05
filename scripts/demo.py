from ultralytics import YOLO

model = YOLO("yolo26n.pt")

#car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

video_path = "parking_lot.mp4"


results = model.predict(
    source = video_path,
    classes = vehicle_classes,
    conf = 0.5,
    save = True
)

for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"vehicle: {class_name},"
            f"Confidence: {confidence:.2f}"
        )

print("Detection finished.")

