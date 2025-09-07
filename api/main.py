from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import asyncio
import logging
from datetime import datetime
import tempfile
import json

# Import VisionFlow components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionflow import (
    ObjectDetector, 
    ImageClassifier, 
    DeepfakeDetector,
    VideoGenerator,
    VideoProcessor,
    VisualSearch,
    Config
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VisionFlow Pro API",
    description="Advanced Multi-Modal Computer Vision Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VisionFlow components
config = Config()
detector = ObjectDetector()
classifier = ImageClassifier()
deepfake_detector = DeepfakeDetector()
video_generator = VideoGenerator()
video_processor = VideoProcessor()
search_engine = VisualSearch()

# Pydantic models for API
class DetectionRequest(BaseModel):
    image_base64: str
    confidence_threshold: Optional[float] = 0.5
    model_name: Optional[str] = "yolov8n"

class DetectionResponse(BaseModel):
    detections: List[Dict[str, Any]]
    processing_time: float
    image_dimensions: List[int]

class ClassificationRequest(BaseModel):
    image_base64: str
    top_k: Optional[int] = 5
    model_name: Optional[str] = "efficientnet-b0"

class ClassificationResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    processing_time: float
    feature_vector: Optional[List[float]]

class DeepfakeRequest(BaseModel):
    image_base64: str
    confidence_threshold: Optional[float] = 0.8

class DeepfakeResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    real_probability: float
    fake_probability: float
    processing_time: float
    artifacts: Optional[Dict[str, Any]]

class VideoGenerationRequest(BaseModel):
    image_base64: Optional[str] = None
    text_prompt: Optional[str] = None
    duration: Optional[float] = 2.0
    fps: Optional[int] = 30
    style_transfer: Optional[bool] = False

class VideoGenerationResponse(BaseModel):
    video_path: str
    processing_time: float
    frame_count: int
    resolution: List[int]

class VideoProcessingRequest(BaseModel):
    video_base64: str
    stabilize: Optional[bool] = True
    enhance: Optional[bool] = True
    interpolate: Optional[bool] = False
    target_fps: Optional[int] = None

class VideoProcessingResponse(BaseModel):
    processed_video_path: str
    processing_stats: Dict[str, Any]
    processing_time: float

class SearchRequest(BaseModel):
    query: str
    search_type: str = "text"  # "text" or "image"
    top_k: Optional[int] = 10
    index_name: Optional[str] = "default"

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    total_results: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# Helper functions
def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    try:
        # Remove header if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """Convert numpy array to base64 string"""
    try:
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        image_data = buffer.getvalue()
        
        # Convert to base64
        base64_string = base64.b64encode(image_data).decode('utf-8')
        
        return f"data:image/{format.lower()};base64,{base64_string}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image conversion failed: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VisionFlow Pro API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services={
            "object_detection": "active",
            "image_classification": "active",
            "deepfake_detection": "active",
            "video_generation": "active",
            "video_processing": "active",
            "visual_search": "active"
        }
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """Object detection endpoint"""
    try:
        start_time = datetime.now()
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        
        # Perform detection
        detections = detector.detect(image)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DetectionResponse(
            detections=detections,
            processing_time=processing_time,
            image_dimensions=[image.shape[1], image.shape[0]]
        )
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(request: ClassificationRequest):
    """Image classification endpoint"""
    try:
        start_time = datetime.now()
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        
        # Perform classification
        predictions = classifier.classify(image, top_k=request.top_k)
        
        # Extract feature vector
        features = classifier.get_feature_vector(image)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClassificationResponse(
            predictions=predictions,
            processing_time=processing_time,
            feature_vector=features.tolist()
        )
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/deepfake", response_model=DeepfakeResponse)
async def detect_deepfake(request: DeepfakeRequest):
    """Deepfake detection endpoint"""
    try:
        start_time = datetime.now()
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        
        # Perform deepfake detection
        result = deepfake_detector.detect(image)
        
        # Analyze artifacts
        artifacts = deepfake_detector.analyze_artifacts(image)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DeepfakeResponse(
            is_deepfake=result['is_deepfake'],
            confidence=result['confidence'],
            real_probability=result['real_probability'],
            fake_probability=result['fake_probability'],
            processing_time=processing_time,
            artifacts=artifacts
        )
    except Exception as e:
        logger.error(f"Deepfake detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deepfake detection failed: {str(e)}")

@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Video generation endpoint"""
    try:
        start_time = datetime.now()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as input_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file:
                
                if request.image_base64:
                    # Save input image
                    image = base64_to_image(request.image_base64)
                    cv2.imwrite(input_file.name, image)
                    
                    # Generate video from image
                    video_path = video_generator.generate_from_image(
                        input_file.name,
                        duration=request.duration,
                        fps=request.fps,
                        output_path=output_file.name
                    )
                elif request.text_prompt:
                    # Generate video from text
                    video_path = video_generator.generate_from_text(
                        request.text_prompt,
                        duration=request.duration,
                        fps=request.fps,
                        output_path=output_file.name
                    )
                else:
                    raise HTTPException(status_code=400, detail="Either image_base64 or text_prompt must be provided")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return VideoGenerationResponse(
            video_path=video_path,
            processing_time=processing_time,
            frame_count=frame_count,
            resolution=[width, height]
        )
    except Exception as e:
        logger.error(f"Video generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

@app.post("/process-video", response_model=VideoProcessingResponse)
async def process_video(request: VideoProcessingRequest):
    """Video processing endpoint"""
    try:
        start_time = datetime.now()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file:
                
                # Save input video
                video_data = base64.b64decode(request.video_base64)
                with open(input_file.name, 'wb') as f:
                    f.write(video_data)
                
                # Process video
                stats = video_processor.process_video(
                    input_file.name,
                    output_file.name,
                    stabilize=request.stabilize,
                    enhance=request.enhance,
                    interpolate=request.interpolate,
                    target_fps=request.target_fps
                )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return VideoProcessingResponse(
            processed_video_path=output_file.name,
            processing_stats=stats,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """Visual search endpoint"""
    try:
        start_time = datetime.now()
        
        if request.search_type == "text":
            # Text-based search
            results = search_engine.search_by_text(
                request.query,
                top_k=request.top_k
            )
        elif request.search_type == "image":
            # Image-based search
            image = base64_to_image(request.query)
            results = search_engine.search_by_image(
                image,
                top_k=request.top_k
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid search type. Use 'text' or 'image'")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SearchResponse(
            results=results,
            processing_time=processing_time,
            total_results=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/batch-detect")
async def batch_detect_objects(images: List[UploadFile] = File(...)):
    """Batch object detection endpoint"""
    try:
        start_time = datetime.now()
        
        # Process images
        image_arrays = []
        for image_file in images:
            image_data = await image_file.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            image_arrays.append(image)
        
        # Perform batch detection
        detections_list = detector.detect_batch(image_arrays)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "detections": detections_list,
            "processing_time": processing_time,
            "total_images": len(image_arrays)
        }
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    try:
        search_stats = search_engine.get_performance_stats()
        
        return {
            "visionflow_stats": search_stats,
            "api_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    try:
        file_path = os.path.join("temp", filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, filename=filename)
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# Background task cleanup
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("VisionFlow Pro API starting up...")
    
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    logger.info("VisionFlow Pro API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("VisionFlow Pro API shutting down...")
    
    # Cleanup temp files
    import shutil
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    
    logger.info("VisionFlow Pro API shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )