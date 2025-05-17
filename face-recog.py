import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Initialize face detection model
face_detector = YOLO('model/yolov11n-face.pt')

def get_optimal_font_scale(text, width, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
    for scale in reversed(range(0, 60)):
        text_size = cv2.getTextSize(text, font, scale / 10, thickness)[0]
        if text_size[0] <= width:
            return scale / 10
    return 1  # default if nothing fits

# Function to predict face identity
def predict_face(face_img, model, label_encoder_classes):
    if face_img.size == 0:
        return None, 0
    
    # Preprocess face
    face = cv2.resize(face_img, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    
    # Make prediction
    predictions = model.predict(face, verbose=0)[0]
    j = np.argmax(predictions)
    probability = predictions[j]
    name = label_encoder_classes[j]
    
    return name, probability

# Function to recognize faces in an image
def recognize_faces(image_path, model_path='model_output/face_recognition_model.h5',
                   encoder_path='model_output/label_encoder.npy', confidence_threshold=0.5):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Load the model and label encoder
    model = load_model(model_path)
    label_encoder_classes = np.load(encoder_path, allow_pickle=True)
    
    # Make a copy for drawing on
    result_img = img.copy()
    
    # Detect faces
    results = face_detector(img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confidences):
            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                
                # Extract face
                face = img[y1:y2, x1:x2]
                
                # Predict face identity
                name, probability = predict_face(face, model, label_encoder_classes)
                
                if name is None:
                    continue
                
                # Draw results on image
                text = f"{name}: {probability:.2f}"
                box_width = x2 - x1
                font_scale = get_optimal_font_scale(text, box_width)
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_img, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (0, 255, 0), 2, cv2.LINE_AA)
    
    return result_img

# Function to process video
def process_video(video_path, model_path='model_output/face_recognition_model.h5',
                 encoder_path='model_output/label_encoder.npy', output_path=None):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load the model and label encoder
    model = load_model(model_path)
    label_encoder_classes = np.load(encoder_path, allow_pickle=True)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 3rd frame for speed (adjust as needed)
        if frame_count % 3 == 0:
            # Detect faces
            results = face_detector(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confidences):
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Extract face
                        face = frame[y1:y2, x1:x2]
                        
                        # Predict face identity
                        name, probability = predict_face(face, model, label_encoder_classes)
                        
                        if name is None:
                            continue
                        
                        # Draw results on frame
                        text = f"{name}: {probability:.2f}"
                        box_width = x2 - x1
                        font_scale = get_optimal_font_scale(text, box_width)
                        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Write to output file if specified
        if out:
            out.write(frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Function to use webcam for real-time face recognition
def use_webcam(model_path='model_output/face_recognition_model.h5',
              encoder_path='model_output/label_encoder.npy',
              camera_id=0, output_path=None):
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {camera_id})")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording to {output_path}")
    
    # Load model and encoder
    print("Loading model and label encoder...")
    model = load_model(model_path)
    label_encoder_classes = np.load(encoder_path, allow_pickle=True)
    print(f"Model loaded. Recognizing {len(label_encoder_classes)} people: {label_encoder_classes}")
    print("Starting webcam face recognition. Press 'q' to quit, 's' to save a snapshot.")
    
    frame_count = 0
    snapshot_count = 0
    last_detections = []  # 🟢 เก็บผลลัพธ์ล่าสุด
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
        
        process_frame = frame.copy()
        if frame_count % 2 == 0:
            new_detections = []
            results = face_detector(process_frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confidences):
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        face = process_frame[y1:y2, x1:x2]
                        
                        # Predict face identity
                        name, probability = predict_face(face, model, label_encoder_classes)
                        
                        if name is None:
                            continue
                            
                        new_detections.append((x1, y1, x2, y2, name, probability))
            
            if new_detections:
                last_detections = new_detections  # อัปเดตเฉพาะเมื่อเจอหน้าใหม่
        
        # วาดกรอบล่าสุดที่พบ (จะใช้เดิมถ้าไม่มีของใหม่)
        for x1, y1, x2, y2, name, prob in last_detections:
            text = f"{name}: {prob:.2f}"
            y_text = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(process_frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                       0.75, (0, 255, 0), 2, cv2.LINE_AA)
        
        # คำแนะนำ
        cv2.putText(process_frame, "Press 'q' to quit, 's' to save snapshot",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Webcam Face Recognition', process_frame)
        if out:
            out.write(process_frame)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("snapshots", exist_ok=True)
            snapshot_path = f"snapshots/snapshot_{snapshot_count}.jpg"
            cv2.imwrite(snapshot_path, process_frame)
            print(f"Snapshot saved to {snapshot_path}")
            snapshot_count += 1
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Webcam session ended")

# Main execution with command line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'video', 'webcam'],
                       help='Mode: image, video, or webcam')
    parser.add_argument('--path', type=str, default=None,
                       help='Path to image or video file (not needed for webcam mode)')
    parser.add_argument('--model', type=str, default='model_output/face_recognition_model.h5',
                       help='Path to trained model')
    parser.add_argument('--encoder', type=str, default='model_output/label_encoder.npy',
                       help='Path to label encoder')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output (optional)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam mode (default: 0)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output and os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'image':
        if not args.path:
            print("Error: Please provide an image path with --path")
        else:
            result_image = recognize_faces(args.path, args.model, args.encoder)
            if result_image is not None:
                # Display result
                cv2.imshow("Face Recognition", result_image)
                cv2.waitKey(0)
                # Save result if output path is specified
                if args.output:
                    cv2.imwrite(args.output, result_image)
                    print(f"Result saved to {args.output}")
    
    elif args.mode == 'video':
        if not args.path:
            print("Error: Please provide a video path with --path")
        else:
            process_video(args.path, args.model, args.encoder, args.output)
    
    elif args.mode == 'webcam':
        use_webcam(args.model, args.encoder, args.camera, args.output)
    
    cv2.destroyAllWindows()