from ultralytics import YOLO
import cv2
import math

# Load trained model
model = YOLO('runs/detect/train4/weights/best.pt')

# Start Webcam 
cap = cv2.VideoCapture(0)

# Set resolution (Optional: delete if screen is black)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success:
        break

    # Run the AI
    results = model(img, stream=True, device=0) # Added device=0 for GPU speed

    # Initialize counters for this frame
    ripe_count = 0
    unripe_count = 0

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Get Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 2. Get Class Name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # 3. SET COLOR BASED ON CLASS 
            if class_name == 'lime_ripe':
                color = (0, 255, 255) # Yellow in BGR
                ripe_count += 1       # Count it
            elif class_name == 'lime_unripe':
                color = (0, 255, 0)   # Green in BGR
                unripe_count += 1     # Count it
            else:
                color = (255, 0, 0)   # Blue (for any other class)

            # 4. Draw the Box with the dynamic 'color'
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # 5. Put Text
            conf = math.ceil((box.conf[0] * 100)) / 100
            label = f'{class_name} {conf}'
            
            # Draw a small background rectangle for text so it's readable
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Filled box
            
            # Write white text on top of the colored box
            cv2.putText(img, label, (x1, y1 - 2), 0, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    # --- DASHBOARD UI (Top Left Corner) ---
    # Draw a semi-transparent background for the stats
    cv2.rectangle(img, (10, 10), (250, 90), (0, 0, 0), -1) # Black box
    
    # Write the Counts
    cv2.putText(img, f'Ripe (Yellow):   {ripe_count}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f'Unripe (Green):  {unripe_count}', (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the dashboard
    cv2.imshow('Lime Detection System', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Testing Git tracking feature