from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import io
import json
import os
from sklearn.cluster import KMeans
from collections import defaultdict

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Load the model
model = YOLO('hold_detector.pt')

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

def get_average_rgb(img, bbox):
    """Calculate average RGB values for a bounding box region."""
    x1, y1, x2, y2 = map(int, bbox)
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    
    # Extract the region
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return [0, 0, 0]
    
    # Calculate mean RGB values
    mean_rgb = np.mean(region, axis=(0, 1))
    return mean_rgb.tolist()

def group_holds_by_color(holds, rgb_values, n_clusters=3):
    """Group holds by their RGB values using K-means clustering."""
    if len(holds) < n_clusters:
        n_clusters = len(holds)
    
    if n_clusters <= 1:
        return {0: holds}
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(rgb_values)
    
    # Group holds by cluster
    grouped_holds = defaultdict(list)
    for hold, cluster_id in zip(holds, clusters):
        grouped_holds[int(cluster_id)].append(hold)
    
    return dict(grouped_holds)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model(img)
        
        # Process results and collect RGB values
        holds = []
        rgb_values = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                # Skip low confidence predictions
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Get average RGB values for the hold
                avg_rgb = get_average_rgb(img, [x1, y1, x2, y2])
                
                hold_data = {
                    "bbox": [x1, y1, x2, y2]
                }
                holds.append(hold_data)
                rgb_values.append(avg_rgb)
        
        # Group holds by color
        if holds:
            rgb_values_array = np.array(rgb_values)
            grouped_holds = group_holds_by_color(holds, rgb_values_array)
        else:
            grouped_holds = {}
        
        return {
            f"route_{i}": route_holds 
            for i, route_holds in grouped_holds.items()
        }
    finally:
        # Ensure the file is closed and cleaned up
        await file.close()

@app.get("/")
async def root():
    return {"message": "YOLOv8s API is running"} 