from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import json
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load our trained model - update the path to your trained model
model = YOLO('runs/detect/demo_model/weights/best.pt')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection on the frame
            results = model(frame, conf=0.5)  
            
            # Draw boxes on the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Get class name and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Draw box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert frame to base64 for sending over websocket
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame with detections
            await websocket.send_text(json.dumps({
                "image": img_str,
            }))
            
            # inserted small delay to control frame rate
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 