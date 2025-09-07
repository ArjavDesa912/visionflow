import unittest
import numpy as np
import cv2
import sys
import os

# Add the visionflow module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionflow.core.detector import ObjectDetector
from visionflow.core.classifier import ImageClassifier
from visionflow.core.deepfake_detector import DeepfakeDetector
from visionflow.video.generator import VideoGenerator
from visionflow.video.processor import VideoProcessor
from visionflow.search.engine import VisualSearch

class TestVisionFlow(unittest.TestCase):
    """Test cases for VisionFlow Pro components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some content to the test image
        self.test_image[100:200, 100:200] = [255, 0, 0]  # Blue rectangle
        self.test_image[300:400, 300:400] = [0, 255, 0]  # Green rectangle
    
    def test_object_detector_initialization(self):
        """Test object detector initialization"""
        detector = ObjectDetector()
        self.assertIsNotNone(detector.model)
        self.assertIsNotNone(detector.class_names)
    
    def test_object_detection(self):
        """Test object detection functionality"""
        detector = ObjectDetector()
        detections = detector.detect(self.test_image)
        self.assertIsInstance(detections, list)
    
    def test_image_classifier_initialization(self):
        """Test image classifier initialization"""
        classifier = ImageClassifier()
        self.assertIsNotNone(classifier.model)
        self.assertIsNotNone(classifier.transform)
    
    def test_image_classification(self):
        """Test image classification functionality"""
        classifier = ImageClassifier()
        predictions = classifier.classify(self.test_image, top_k=5)
        self.assertIsInstance(predictions, list)
        self.assertLessEqual(len(predictions), 5)
    
    def test_deepfake_detector_initialization(self):
        """Test deepfake detector initialization"""
        detector = DeepfakeDetector()
        self.assertIsNotNone(detector.model)
        self.assertIsNotNone(detector.transform)
    
    def test_deepfake_detection(self):
        """Test deepfake detection functionality"""
        detector = DeepfakeDetector()
        result = detector.detect(self.test_image)
        self.assertIsInstance(result, dict)
        self.assertIn('is_deepfake', result)
        self.assertIn('confidence', result)
    
    def test_video_generator_initialization(self):
        """Test video generator initialization"""
        generator = VideoGenerator()
        self.assertIsNotNone(generator.frame_rate)
        self.assertIsNotNone(generator.resolution)
    
    def test_video_processor_initialization(self):
        """Test video processor initialization"""
        processor = VideoProcessor()
        self.assertIsNotNone(processor.frame_rate)
        self.assertIsNotNone(processor.resolution)
    
    def test_visual_search_initialization(self):
        """Test visual search engine initialization"""
        search_engine = VisualSearch()
        self.assertIsNotNone(search_engine.embedding_dim)
        self.assertIsNotNone(search_engine.clip_model)
        self.assertIsNotNone(search_engine.clip_processor)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        # Create test images
        test_images = [self.test_image.copy() for _ in range(3)]
        
        # Test batch object detection
        detector = ObjectDetector()
        detections_list = detector.detect_batch(test_images)
        self.assertEqual(len(detections_list), 3)
        
        # Test batch classification
        classifier = ImageClassifier()
        predictions_list = classifier.classify_batch(test_images, top_k=3)
        self.assertEqual(len(predictions_list), 3)
        
        # Test batch deepfake detection
        deepfake_detector = DeepfakeDetector()
        results_list = deepfake_detector.detect_batch(test_images)
        self.assertEqual(len(results_list), 3)
    
    def test_feature_extraction(self):
        """Test feature extraction functionality"""
        classifier = ImageClassifier()
        features = classifier.get_feature_vector(self.test_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_image_similarity(self):
        """Test image similarity calculation"""
        classifier = ImageClassifier()
        similarity = classifier.get_similarity(self.test_image, self.test_image)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

class TestVideoProcessing(unittest.TestCase):
    """Test cases for video processing components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_frames = []
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add moving circle
            x = 100 + i * 40
            cv2.circle(frame, (x, 240), 30, (0, 255, 0), -1)
            self.test_frames.append(frame)
    
    def test_video_processor_functions(self):
        """Test video processing functions"""
        processor = VideoProcessor()
        
        # Test frame interpolation
        interpolated = processor.interpolate_frames(self.test_frames, 60)
        self.assertGreater(len(interpolated), len(self.test_frames))
        
        # Test scene detection
        scene_changes = processor.detect_scenes(self.test_frames)
        self.assertIsInstance(scene_changes, list)
        
        # Test keyframe extraction
        keyframes = processor.extract_keyframes(self.test_frames, method='uniform')
        self.assertIsInstance(keyframes, list)
        self.assertGreater(len(keyframes), 0)

class TestVisualSearch(unittest.TestCase):
    """Test cases for visual search components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.search_engine = VisualSearch()
        self.test_images = []
        for i in range(5):
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i]
            cv2.rectangle(image, (100 + i*20, 100 + i*20), (300 + i*20, 300 + i*20), color, -1)
            self.test_images.append(image)
    
    def test_embedding_extraction(self):
        """Test embedding extraction"""
        image_embedding = self.search_engine.extract_image_embedding(self.test_images[0])
        self.assertIsInstance(image_embedding, np.ndarray)
        
        text_embedding = self.search_engine.extract_text_embedding("test query")
        self.assertIsInstance(text_embedding, np.ndarray)
    
    def test_index_creation(self):
        """Test index creation"""
        self.search_engine.create_index('flat')
        self.assertIsNotNone(self.search_engine.index)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        stats = self.search_engine.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_queries', stats)
        self.assertIn('average_latency', stats)

def run_tests():
    """Run all tests"""
    print("Running VisionFlow Pro Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVisionFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestVideoProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualSearch))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)