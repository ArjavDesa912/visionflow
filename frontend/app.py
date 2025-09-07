import streamlit as st
import requests
import base64
import io
import cv2
import numpy as np
from PIL import Image
import json
import time
import os
from typing import Dict, List, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8080"  # Change to your API URL
STREAMLIT_SERVER_PORT = 8501
STREAMLIT_SERVER_ADDRESS = "0.0.0.0"

# Page configuration
st.set_page_config(
    page_title="VisionFlow Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# API Client Class
class VisionFlowAPIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def detect_objects(self, image_base64: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Object detection API call"""
        try:
            response = self.session.post(
                f"{self.base_url}/detect",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": confidence_threshold
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Object detection failed: {str(e)}")
            return None
    
    def classify_image(self, image_base64: str, top_k: int = 5) -> Dict[str, Any]:
        """Image classification API call"""
        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                json={
                    "image_base64": image_base64,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Image classification failed: {str(e)}")
            return None
    
    def detect_deepfake(self, image_base64: str, confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Deepfake detection API call"""
        try:
            response = self.session.post(
                f"{self.base_url}/deepfake",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": confidence_threshold
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Deepfake detection failed: {str(e)}")
            return None
    
    def search_images(self, query: str, search_type: str = "text", top_k: int = 10) -> Dict[str, Any]:
        """Visual search API call"""
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "search_type": search_type,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Visual search failed: {str(e)}")
            return None
    
    def generate_video(self, image_base64: str = None, text_prompt: str = None, duration: float = 2.0) -> Dict[str, Any]:
        """Video generation API call"""
        try:
            response = self.session.post(
                f"{self.base_url}/generate-video",
                json={
                    "image_base64": image_base64,
                    "text_prompt": text_prompt,
                    "duration": duration
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Video generation failed: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get stats: {str(e)}")
            return None

# Initialize API client
api_client = VisionFlowAPIClient()

# Helper functions
def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def draw_detections(image, detections):
    """Draw detection bounding boxes on image"""
    result_image = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ VisionFlow Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Multi-Modal Computer Vision Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Feature",
        ["🎯 Object Detection", "🏷️ Image Classification", "🔍 Deepfake Detection", 
         "🎬 Video Generation", "🔎 Visual Search", "📊 Dashboard"]
    )
    
    # API Configuration
    st.sidebar.subheader("API Configuration")
    api_url = st.sidebar.text_input("API URL", value=API_BASE_URL)
    if api_url != API_BASE_URL:
        api_client.base_url = api_url
    
    # Check API Health
    if st.sidebar.button("Check API Health"):
        try:
            response = requests.get(f"{api_client.base_url}/health")
            if response.status_code == 200:
                st.sidebar.success("✅ API is healthy")
            else:
                st.sidebar.error("❌ API is not responding")
        except:
            st.sidebar.error("❌ Cannot connect to API")
    
    # Main content area
    if page == "🎯 Object Detection":
        object_detection_page()
    elif page == "🏷️ Image Classification":
        image_classification_page()
    elif page == "🔍 Deepfake Detection":
        deepfake_detection_page()
    elif page == "🎬 Video Generation":
        video_generation_page()
    elif page == "🔎 Visual Search":
        visual_search_page()
    elif page == "📊 Dashboard":
        dashboard_page()

def object_detection_page():
    st.header("🎯 Object Detection")
    st.markdown("Detect objects in images using YOLO-v8")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to base64
            img_base64 = image_to_base64(image)
            
            # Detection parameters
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
            
            if st.button("Detect Objects"):
                with st.spinner("Detecting objects..."):
                    result = api_client.detect_objects(img_base64, confidence_threshold)
                    
                    if result:
                        with col2:
                            st.subheader("Results")
                            
                            # Display metrics
                            st.metric("Total Objects", len(result['detections']))
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            
                            # Display detections
                            for i, detection in enumerate(result['detections']):
                                with st.expander(f"Object {i+1}: {detection['class_name']}"):
                                    st.json({
                                        "Class": detection['class_name'],
                                        "Confidence": detection['confidence'],
                                        "Bounding Box": detection['bbox']
                                    })
                            
                            # Show detection results on image
                            if result['detections']:
                                # Convert PIL to OpenCV format
                                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                result_image = draw_detections(cv_image, result['detections'])
                                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                                st.image(result_image, caption="Detection Results", use_column_width=True)

def image_classification_page():
    st.header("🏷️ Image Classification")
    st.markdown("Classify images using EfficientNet-B7")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to base64
            img_base64 = image_to_base64(image)
            
            # Classification parameters
            top_k = st.slider("Top K Predictions", 1, 10, 5)
            
            if st.button("Classify Image"):
                with st.spinner("Classifying image..."):
                    result = api_client.classify_image(img_base64, top_k)
                    
                    if result:
                        with col2:
                            st.subheader("Results")
                            
                            # Display metrics
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            
                            # Display predictions as bar chart
                            predictions = result['predictions']
                            df = pd.DataFrame(predictions)
                            fig = px.bar(df, x='class_name', y='confidence', 
                                        title='Top Predictions')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display detailed predictions
                            st.subheader("Detailed Predictions")
                            for i, prediction in enumerate(predictions):
                                st.write(f"**{i+1}. {prediction['class_name']}**")
                                st.progress(prediction['confidence'])
                                st.write(f"Confidence: {prediction['confidence']:.4f}")

def deepfake_detection_page():
    st.header("🔍 Deepfake Detection")
    st.markdown("Detect deepfake images using Vision Transformers (94% accuracy)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to base64
            img_base64 = image_to_base64(image)
            
            # Detection parameters
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.8, 0.1)
            
            if st.button("Detect Deepfake"):
                with st.spinner("Analyzing image..."):
                    result = api_client.detect_deepfake(img_base64, confidence_threshold)
                    
                    if result:
                        with col2:
                            st.subheader("Results")
                            
                            # Display metrics
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            
                            # Display result
                            if result['is_deepfake']:
                                st.error("🚨 **DEEPFAKE DETECTED**")
                                st.metric("Fake Probability", f"{result['fake_probability']:.2%}")
                            else:
                                st.success("✅ **AUTHENTIC IMAGE**")
                                st.metric("Real Probability", f"{result['real_probability']:.2%}")
                            
                            st.metric("Overall Confidence", f"{result['confidence']:.2%}")
                            
                            # Display confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = result['confidence'] * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Confidence Level"},
                                delta = {'reference': 80},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"}],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90}}}))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display artifacts analysis
                            if result.get('artifacts'):
                                st.subheader("Artifact Analysis")
                                st.json(result['artifacts'])

def video_generation_page():
    st.header("🎬 Video Generation")
    st.markdown("Generate videos from images or text using Stable Video Diffusion")
    
    # Generation mode selection
    generation_mode = st.radio("Generation Mode", ["Image to Video", "Text to Video"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        if generation_mode == "Image to Video":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
            img_base64 = None
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Source Image", use_column_width=True)
                img_base64 = image_to_base64(image)
        
        else:  # Text to Video
            text_prompt = st.text_area("Enter text prompt:", height=100, 
                                     placeholder="A beautiful landscape with mountains and lakes...")
        
        # Generation parameters
        duration = st.slider("Duration (seconds)", 1.0, 10.0, 2.0, 0.5)
        fps = st.slider("FPS", 15, 60, 30, 5)
        
        if st.button("Generate Video"):
            if generation_mode == "Image to Video" and img_base64:
                with st.spinner("Generating video..."):
                    result = api_client.generate_video(image_base64=img_base64, duration=duration)
                    
                    if result:
                        with col2:
                            st.subheader("Results")
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            st.metric("Frame Count", result['frame_count'])
                            st.metric("Resolution", f"{result['resolution'][0]}x{result['resolution'][1]}")
                            
                            st.success(f"Video generated successfully!")
                            st.write(f"Video saved to: {result['video_path']}")
            
            elif generation_mode == "Text to Video" and text_prompt:
                with st.spinner("Generating video..."):
                    result = api_client.generate_video(text_prompt=text_prompt, duration=duration)
                    
                    if result:
                        with col2:
                            st.subheader("Results")
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            st.metric("Frame Count", result['frame_count'])
                            st.metric("Resolution", f"{result['resolution'][0]}x{result['resolution'][1]}")
                            
                            st.success(f"Video generated successfully!")
                            st.write(f"Video saved to: {result['video_path']}")

def visual_search_page():
    st.header("🔎 Visual Search")
    st.markdown("Search for similar images using CLIP embeddings and Faiss")
    
    # Search mode selection
    search_mode = st.radio("Search Mode", ["Text Search", "Image Search"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Search Query")
        
        if search_mode == "Text Search":
            query_text = st.text_area("Enter search query:", height=100, 
                                    placeholder="A red car on a sunny day...")
            search_type = "text"
            query = query_text
        else:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
            query = None
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Query Image", use_column_width=True)
                query = image_to_base64(image)
                search_type = "image"
        
        # Search parameters
        top_k = st.slider("Number of Results", 1, 50, 10, 1)
        
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    result = api_client.search_images(query, search_type, top_k)
                    
                    if result:
                        with col2:
                            st.subheader("Search Results")
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            st.metric("Total Results", result['total_results'])
                            
                            # Display search results
                            for i, search_result in enumerate(result['results']):
                                with st.expander(f"Result {i+1}"):
                                    st.write(f"**Image Path:** {search_result['image_path']}")
                                    st.write(f"**Similarity Score:** {search_result.get('similarity_score', 0):.4f}")
                                    if search_result.get('metadata'):
                                        st.write("**Metadata:**")
                                        st.json(search_result['metadata'])

def dashboard_page():
    st.header("📊 Dashboard")
    st.markdown("System statistics and performance metrics")
    
    # Get system stats
    stats = api_client.get_stats()
    
    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", stats.get('visionflow_stats', {}).get('total_queries', 0))
        
        with col2:
            avg_latency = stats.get('visionflow_stats', {}).get('average_latency', 0)
            st.metric("Avg Latency", f"{avg_latency:.3f}s")
        
        with col3:
            images_indexed = stats.get('visionflow_stats', {}).get('images_indexed', 0)
            st.metric("Images Indexed", images_indexed)
        
        with col4:
            index_type = stats.get('visionflow_stats', {}).get('index_type', 'N/A')
            st.metric("Index Type", index_type)
        
        # System information
        st.subheader("System Information")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**VisionFlow Stats:**")
            st.json(stats.get('visionflow_stats', {}))
        
        with col2:
            st.write("**API Information:**")
            st.json({
                "Version": stats.get('api_version', 'N/A'),
                "Timestamp": stats.get('timestamp', 'N/A')
            })
        
        # Performance charts (placeholder data)
        st.subheader("Performance Trends")
        
        # Query volume over time (placeholder)
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        query_volume = np.random.randint(50, 200, 30)
        
        fig1 = px.line(x=dates, y=query_volume, title="Query Volume Over Time")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Processing time distribution (placeholder)
        processing_times = np.random.normal(0.5, 0.1, 100)
        
        fig2 = px.histogram(x=processing_times, title="Processing Time Distribution")
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()