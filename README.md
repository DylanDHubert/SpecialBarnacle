# YOLOv8s API on Render.com

This repository contains a FastAPI application that serves a YOLOv8s model for object detection. The application is designed to be deployed on Render.com.

## Deployment on Render.com

1. Push this repository to a Git provider (GitHub, GitLab, etc.)
2. Create a new Web Service on Render.com
3. Connect your repository
4. Configure the service:
   - Name: Choose a name for your service
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Python Version: 3.10.0
   - Instance Type: Choose based on your needs (at least 1GB RAM recommended)

## API Usage

### Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Main endpoint for image inference

### Example Usage with curl

```bash
curl -X POST "https://your-render-url/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

### Example Response

```json
{
    "route_0": [
        {
            "bbox": [x1, y1, x2, y2]
        }
    ],
    "route_1": [
        {
            "bbox": [x1, y1, x2, y2]
        }
    ]
}
```

The API returns a simple JSON object where:
- Each key is a route identifier (route_0, route_1, etc.)
- Each value is an array of holds in that route
- Each hold contains only its bounding box coordinates

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000` # SpecialBarnacle
