import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mediapipe as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.lrn1(nn.functional.relu(self.conv1(x))))
        x = self.pool2(self.lrn2(nn.functional.relu(self.conv2(x))))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool3(nn.functional.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.drop1(nn.functional.relu(self.fc1(x)))
        x = self.drop2(nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Setup Model and Constants
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
model = CNN().to(device)
try:
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pth not found. Please run train.py first.")
    exit()

# Set to eval mode for Batch Norm behavior, but we will allow gradients for Grad-CAM
model.eval() 

# Preprocessing transform
transform_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Core Grad-CAM Logic
def compute_grad_cam(model, input_tensor, target_layer):
    # Hook storage
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    pred_class_name = class_names[pred_idx]
    
    # Backward pass
    score = output[0, pred_idx]
    score.backward()

    # Generate CAM
    act = activations[0].detach().cpu()
    grad = gradients[0].detach().cpu()
    
    # Global Average Pooling on gradients
    weights = torch.mean(grad, dim=(2, 3))[0]
    
    # Weighted combination
    cam = torch.zeros(act.shape[2:], dtype=torch.float32)
    for i in range(len(weights)):
        cam += weights[i] * act[0, i, :, :]
        
    # ReLU
    cam = torch.relu(cam).numpy()
    
    # Normalize
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    return cam, pred_class_name, output.softmax(dim=1).max().item()

def generate_visualization(input_tensor, cam_mask, layer_name):
    # Denormalize the original image for visualization
    img_np = input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Convert to uint8 for OpenCV
    img_uint8 = np.uint8(255 * img_np)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

    # Process heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    
    # Overlay
    overlay = cv2.addWeighted(heatmap, 0.4, img_bgr, 0.6, 0)
    
    # Add Text Label
    cv2.putText(overlay, layer_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return overlay

# 5. Main Loop
def main():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        return

    print("Running Live Grad-CAM. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Prepare frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        grad_cam_display = None # Placeholder for our stitched image

        if results.detections:
            # Pick the largest face
            detection = max(results.detections, key=lambda d: d.score[0])
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)
            
            # Bounds check
            x, y = max(0, x), max(0, y)
            box_w = min(box_w, w - x)
            box_h = min(box_h, h - y)

            if box_w > 0 and box_h > 0:
                # Crop and Preprocess
                face_roi = frame_rgb[y:y+box_h, x:x+box_w]
                face_pil = Image.fromarray(face_roi)
                input_tensor = transform_preprocess(face_pil).unsqueeze(0).to(device)
                
                input_tensor.requires_grad = True 

                # Loop through layers and generate maps
                layer_maps = []
                target_layers = [
                    ('Conv1', model.conv1),
                    ('Conv2', model.conv2),
                    ('Conv3', model.conv3),
                    ('Conv4', model.conv4),
                    ('Conv5', model.conv5)
                ]
                
                pred_text = ""
                conf_val = 0.0

                for name, layer in target_layers:
                    cam, pred_text, conf_val = compute_grad_cam(model, input_tensor, layer)
                    viz = generate_visualization(input_tensor, cam, name)
                    layer_maps.append(viz)

                # Stitch images horizontally
                grad_cam_display = cv2.hconcat(layer_maps)
                
                # Draw on main frame
                cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (0, 255, 0), 2)
                label_text = f"{pred_text} ({conf_val:.2f})"
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show Main Feed
        cv2.imshow('Live Feed', frame)
        
        # Show Grad-CAM Feed if available
        if grad_cam_display is not None:
            cv2.imshow('Layer Grad-CAMs (Conv1 - Conv5)', grad_cam_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

if __name__ == "__main__":
    main()
