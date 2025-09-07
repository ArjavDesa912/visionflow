import torch
import torch.nn as nn
import numpy as np
import faiss
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import os
import pickle
from tqdm import tqdm
from ..utils.config import config

class VisualSearch:
    """Distributed visual search engine with CLIP embeddings and HNSW indexing"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = config.get('models.device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Search configuration
        self.embedding_dim = config.get('search.embedding_dim', 512)
        self.index_type = config.get('search.index_type', 'hnsw')
        self.ef_construction = config.get('search.ef_construction', 100)
        self.ef_search = config.get('search.ef_search', 50)
        self.max_connections = config.get('search.max_connections', 32)
        self.max_elements = config.get('search.max_elements', 1000000)
        self.use_gpu = config.get('search.use_gpu', False)
        
        # Initialize models
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Initialize index
        self.index = None
        self.image_paths = []
        self.metadata = []
        
        # Performance metrics
        self.query_count = 0
        self.total_latency = 0
    
    def create_index(self, index_type: str = None):
        """
        Create FAISS index for efficient similarity search
        
        Args:
            index_type: Type of index ('hnsw', 'flat', 'ivf')
        """
        index_type = index_type or self.index_type
        
        if index_type == 'hnsw':
            # HNSW index for fast approximate nearest neighbor search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.max_connections)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
        
        elif index_type == 'flat':
            # Flat index for exact search
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        elif index_type == 'ivf':
            # IVF index for large datasets
            nlist = min(100, len(self.image_paths) // 10) if self.image_paths else 100
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
    
    def extract_image_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract CLIP embedding from image
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Image embedding as numpy array
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Process image with CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize embedding
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract text embedding using sentence transformers
        
        Args:
            text: Input text
            
        Returns:
            Text embedding as numpy array
        """
        with torch.no_grad():
            text_embedding = self.text_model.encode(text, convert_to_numpy=True)
            # Normalize embedding
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        return text_embedding
    
    def add_images(self, image_paths: List[str], metadata: List[Dict] = None):
        """
        Add images to the search index
        
        Args:
            image_paths: List of image file paths
            metadata: List of metadata dictionaries for each image
        """
        if metadata is None:
            metadata = [{} for _ in image_paths]
        
        # Extract embeddings for all images
        embeddings = []
        for image_path in tqdm(image_paths, desc="Extracting image embeddings"):
            try:
                embedding = self.extract_image_embedding(image_path)
                embeddings.append(embedding)
                self.image_paths.append(image_path)
                self.metadata.append(metadata[len(self.image_paths) - 1])
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Create index if not exists
            if self.index is None:
                self.create_index()
            
            # Add embeddings to index
            if hasattr(self.index, 'train') and not self.index.is_trained:
                self.index.train(embeddings)
            
            self.index.add(embeddings)
            
            print(f"Added {len(embeddings)} images to search index")
    
    def search_by_image(self, 
                       query_image: Union[str, np.ndarray, Image.Image],
                       top_k: int = 10,
                       return_scores: bool = True) -> List[Dict]:
        """
        Search for similar images
        
        Args:
            query_image: Query image
            top_k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of search results
        """
        if self.index is None:
            raise ValueError("Index not created. Add images first.")
        
        # Extract query embedding
        query_embedding = self.extract_image_embedding(query_image)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search index
        import time
        start_time = time.time()
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        latency = time.time() - start_time
        self.query_count += 1
        self.total_latency += latency
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.image_paths) and idx >= 0:
                result = {
                    'rank': i + 1,
                    'image_path': self.image_paths[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                }
                if return_scores:
                    result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search_by_text(self, 
                      query_text: str,
                      top_k: int = 10,
                      return_scores: bool = True) -> List[Dict]:
        """
        Search images by text query
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of search results
        """
        if self.index is None:
            raise ValueError("Index not created. Add images first.")
        
        # Extract text embedding and convert to image embedding space
        text_embedding = self.extract_text_embedding(query_text)
        
        # For CLIP, we need to use the text encoder
        inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        query_embedding = text_features.cpu().numpy()[0].reshape(1, -1).astype(np.float32)
        
        # Search index
        import time
        start_time = time.time()
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        latency = time.time() - start_time
        self.query_count += 1
        self.total_latency += latency
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.image_paths) and idx >= 0:
                result = {
                    'rank': i + 1,
                    'image_path': self.image_paths[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'query_text': query_text
                }
                if return_scores:
                    result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def hybrid_search(self,
                     query_image: Union[str, np.ndarray, Image.Image] = None,
                     query_text: str = None,
                     top_k: int = 10,
                     image_weight: float = 0.5,
                     text_weight: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining image and text queries
        
        Args:
            query_image: Query image (optional)
            query_text: Query text (optional)
            top_k: Number of results to return
            image_weight: Weight for image query
            text_weight: Weight for text query
            
        Returns:
            List of search results
        """
        if query_image is None and query_text is None:
            raise ValueError("At least one of query_image or query_text must be provided")
        
        results = []
        
        if query_image is not None:
            image_results = self.search_by_image(query_image, top_k * 2)
            results.extend(image_results)
        
        if query_text is not None:
            text_results = self.search_by_text(query_text, top_k * 2)
            results.extend(text_results)
        
        # Remove duplicates and re-rank
        unique_results = {}
        for result in results:
            image_path = result['image_path']
            if image_path not in unique_results:
                unique_results[image_path] = result
            else:
                # Combine scores
                if 'similarity_score' in result:
                    unique_results[image_path]['similarity_score'] = max(
                        unique_results[image_path].get('similarity_score', 0),
                        result['similarity_score']
                    )
        
        # Sort by combined score
        sorted_results = sorted(unique_results.values(), 
                               key=lambda x: x.get('similarity_score', 0), 
                               reverse=True)
        
        return sorted_results[:top_k]
    
    def save_index(self, index_path: str):
        """
        Save search index to disk
        
        Args:
            index_path: Path to save index
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = index_path.replace('.index', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'image_paths': self.image_paths,
                'metadata': self.metadata,
                'model_name': self.model_name,
                'config': {
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type,
                    'ef_construction': self.ef_construction,
                    'ef_search': self.ef_search,
                    'max_connections': self.max_connections
                }
            }, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """
        Load search index from disk
        
        Args:
            index_path: Path to saved index
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        # Load metadata
        metadata_path = index_path.replace('.index', '_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.image_paths = saved_data['image_paths']
        self.metadata = saved_data['metadata']
        self.model_name = saved_data['model_name']
        
        # Update config
        config_data = saved_data['config']
        self.embedding_dim = config_data['embedding_dim']
        self.index_type = config_data['index_type']
        self.ef_construction = config_data['ef_construction']
        self.ef_search = config_data['ef_search']
        self.max_connections = config_data['max_connections']
        
        print(f"Index loaded from {index_path}")
        print(f"Loaded {len(self.image_paths)} images")
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_latency = self.total_latency / self.query_count if self.query_count > 0 else 0
        
        return {
            'total_queries': self.query_count,
            'total_latency': self.total_latency,
            'average_latency': avg_latency,
            'images_indexed': len(self.image_paths),
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'use_gpu': self.use_gpu
        }
    
    def clear_index(self):
        """Clear the search index"""
        self.index = None
        self.image_paths = []
        self.metadata = []
        self.query_count = 0
        self.total_latency = 0
        print("Index cleared")
    
    def evaluate_index(self, test_queries: List[Dict], k_values: List[int] = [1, 5, 10]) -> Dict:
        """
        Evaluate search index performance
        
        Args:
            test_queries: List of test queries with expected results
            k_values: List of k values for evaluation
            
        Returns:
            Evaluation metrics
        """
        evaluation_results = {}
        
        for k in k_values:
            recall_at_k = 0
            precision_at_k = 0
            
            for query in test_queries:
                # Perform search
                if 'image' in query:
                    results = self.search_by_image(query['image'], top_k=k)
                elif 'text' in query:
                    results = self.search_by_text(query['text'], top_k=k)
                else:
                    continue
                
                # Calculate metrics
                retrieved_paths = [r['image_path'] for r in results]
                relevant_paths = query['relevant_images']
                
                # Recall@K
                relevant_retrieved = set(retrieved_paths) & set(relevant_paths)
                recall_at_k += len(relevant_retrieved) / len(relevant_paths)
                
                # Precision@K
                precision_at_k += len(relevant_retrieved) / k
            
            evaluation_results[f'recall@{k}'] = recall_at_k / len(test_queries)
            evaluation_results[f'precision@{k}'] = precision_at_k / len(test_queries)
        
        return evaluation_results