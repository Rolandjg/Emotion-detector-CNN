import cv2
import mediapipe as mp
from PIL import Image
from infer import infer

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # model_selection: 0 for short-range (2m), 1 for full-range (5m)
    # min_detection_confidence: minimum confidence threshold (0.0-1.0)
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = face_detection.process(rgb_frame)
            
            prediction_text = None
            
            if results.detections:
                # Get the detection with highest confidence
                detection = max(results.detections, key=lambda d: d.score[0])
                
                # Extract bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                box_w = int(bboxC.width * w)
                box_h = int(bboxC.height * h)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                box_w = min(box_w, w - x)
                box_h = min(box_h, h - y)
                
                if box_w > 0 and box_h > 0:
                    # Crop face region from the color frame
                    face_roi = frame[y:y + box_h, x:x + box_w]
                    
                    # Convert cropped face to PIL Image (RGB)
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    # Use the infer function on the cropped face
                    prediction_text = infer(face_pil)
                    
                    # Print prediction and confidence to console
                    confidence = detection.score[0]
                    print(f"Prediction: {prediction_text} (Face confidence: {confidence:.2f})")
                    
                    # Draw rectangle and label on frame
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{prediction_text} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
            
            # Show the video feed
            cv2.imshow("Live Emotion Recognition (press 'q' to quit)", frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        face_detection.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
