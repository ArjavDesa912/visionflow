from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

# Base API Models
class BaseResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime = datetime.now()

# Object Detection Models
class ObjectDetectionRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    confidence_threshold: float = 0.5
    model_name: str = "yolov8n"
    return_image: bool = False

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str

class ObjectDetectionResponse(BaseModel):
    success: bool
    detections: List[Detection]
    processing_time: float
    image_dimensions: List[int]
    result_image_base64: Optional[str] = None

# Image Classification Models
class ClassificationRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    top_k: int = 5
    model_name: str = "efficientnet-b0"
    return_features: bool = False

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float

class ClassificationResponse(BaseModel):
    success: bool
    predictions: List[Prediction]
    processing_time: float
    feature_vector: Optional[List[float]] = None

# Deepfake Detection Models
class DeepfakeDetectionRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    confidence_threshold: float = 0.8
    return_heatmap: bool = False

class DeepfakeDetectionResponse(BaseModel):
    success: bool
    is_deepfake: bool
    confidence: float
    real_probability: float
    fake_probability: float
    processing_time: float
    artifacts: Optional[Dict[str, Any]] = None
    heatmap_base64: Optional[str] = None

# Video Generation Models
class VideoGenerationRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    text_prompt: Optional[str] = None
    duration: float = 2.0
    fps: int = 30
    style_transfer: bool = False
    enhance_quality: bool = True

class VideoGenerationResponse(BaseModel):
    success: bool
    video_url: str
    video_id: str
    processing_time: float
    frame_count: int
    resolution: List[int]
    file_size: int

# Video Processing Models
class VideoProcessingRequest(BaseModel):
    video_url: Optional[str] = None
    video_base64: Optional[str] = None
    stabilize: bool = True
    enhance: bool = True
    interpolate: bool = False
    target_fps: Optional[int] = None
    noise_reduction: bool = True
    color_correction: bool = True

class VideoProcessingResponse(BaseModel):
    success: bool
    processed_video_url: str
    video_id: str
    processing_stats: Dict[str, Any]
    processing_time: float

# Visual Search Models
class SearchRequest(BaseModel):
    query: str
    search_type: str = "text"  # "text" or "image"
    top_k: int = 10
    index_name: str = "default"
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    rank: int
    image_url: str
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResult]
    processing_time: float
    total_results: int
    query_info: Dict[str, Any]

# Batch Processing Models
class BatchDetectionRequest(BaseModel):
    images: List[str]  # List of base64 encoded images
    confidence_threshold: float = 0.5
    model_name: str = "yolov8n"

class BatchDetectionResponse(BaseModel):
    success: bool
    results: List[List[Detection]]
    processing_time: float
    total_images: int

class BatchClassificationRequest(BaseModel):
    images: List[str]
    top_k: int = 5
    model_name: str = "efficientnet-b0"

class BatchClassificationResponse(BaseModel):
    success: bool
    results: List[List[Prediction]]
    processing_time: float
    total_images: int

# User and Authentication Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

# Analytics and Monitoring Models
class APIUsage(BaseModel):
    endpoint: str
    method: str
    timestamp: datetime
    response_time: float
    status_code: int
    user_id: Optional[int]

class SystemStats(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    uptime: float
    active_users: int
    model_load_status: Dict[str, bool]

# Error Models
class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Webhook Models
class WebhookConfig(BaseModel):
    url: str
    events: List[str]
    secret: Optional[str] = None
    active: bool = True

class WebhookEvent(BaseModel):
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[int] = None

# Job Processing Models
class JobRequest(BaseModel):
    job_type: str
    parameters: Dict[str, Any]
    priority: int = 0
    callback_url: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# Configuration Models
class ModelConfig(BaseModel):
    model_name: str
    version: str
    parameters: Dict[str, Any]
    is_active: bool = True

class SystemConfig(BaseModel):
    max_file_size: int = 10485760  # 10MB
    max_concurrent_requests: int = 100
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # 1 hour
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    logging_level: str = "INFO"