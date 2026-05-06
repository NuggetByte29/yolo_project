import cv2

print("Scanning for cameras... (This might take a few seconds)")

# Check the first 5 indexes
for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera found at Index {index}")
            # Optional: Show a popup to see WHICH camera it is
            cv2.imshow(f"Camera Index {index}", frame)
            cv2.waitKey(2000) # Shows for 2 seconds then closes
            cv2.destroyAllWindows()
        else:
            print(f"⚠️ Index {index} opened, but returned no image.")
    else:
        print(f"❌ No camera at Index {index}")
    cap.release()