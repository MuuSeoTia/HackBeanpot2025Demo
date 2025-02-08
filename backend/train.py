from ultralytics import YOLO

def train_model():
    """
    Train a YOLOv8 model for object detection.
    We're using the nano model for simplicity 
    """
    # Load the base model - YOLOv8 nano is perfect for demos
    model = YOLO('yolov8n.pt')
   
    
    # Train it on COCO128, a small dataset perfect for testing
    # You can replace this with your own custom dataset following YOLO format
    results = model.train(
        data='coco128.yaml',  
        epochs=15,           
        imgsz=640,           
        batch=16,            
        name='demo_model'    
    )
    
    # Export to ONNX format for faster inference
    model.export(format='onnx')
    print("Training complete! Model saved in runs/detect/demo_model")

if __name__ == "__main__":
    train_model() 