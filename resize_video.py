import cv2

input_path = "videos/parking_lot.mp4"
output_path = "videos/parking_lot_640p.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_width = 960
target_height = int(height * (target_width / width))

print(f"Orijinal Çözünürlük: {width}x{height}")
print(f"Yeni Çözünürlük: {target_width}x{target_height}")

# Video yazıcıyı ayarla
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi yeni boyutlara yeniden boyutlandır (INTER_AREA interpolasyonu kaliteyi korur)
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    out.write(resized_frame)

cap.release()
out.release()
print("Video başarıyla küçültüldü ve kaydedildi:", output_path)