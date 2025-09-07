"""
Cloud Storage Integration for VisionFlow Pro

This module provides integration with Google Cloud Storage for storing
and retrieving models, images, videos, and other files.
"""

import os
import json
import base64
import io
import pickle
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging

# Google Cloud Storage
try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, GoogleCloudError
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    storage = None

# Local storage fallback
import shutil
import hashlib

logger = logging.getLogger(__name__)

class CloudStorageManager:
    """Manages cloud storage operations for VisionFlow Pro"""
    
    def __init__(self, 
                 bucket_name: str = None,
                 credentials_path: str = None,
                 use_local_fallback: bool = True):
        """
        Initialize cloud storage manager
        
        Args:
            bucket_name: Name of the GCS bucket
            credentials_path: Path to service account credentials
            use_local_fallback: Whether to use local storage if GCP is unavailable
        """
        self.bucket_name = bucket_name or os.getenv('STORAGE_BUCKET', 'visionflow-storage')
        self.credentials_path = credentials_path
        self.use_local_fallback = use_local_fallback
        
        # Initialize clients
        self.client = None
        self.bucket = None
        self.local_storage_path = './storage'
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage client and bucket"""
        try:
            if storage is None:
                raise ImportError("Google Cloud Storage library not installed")
            
            # Initialize GCS client
            if self.credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Test connection
            self.bucket.reload()
            logger.info(f"Connected to GCS bucket: {self.bucket_name}")
            
        except (ImportError, DefaultCredentialsError, NotFound) as e:
            logger.warning(f"Failed to initialize GCS: {e}")
            if self.use_local_fallback:
                self._initialize_local_storage()
            else:
                raise
    
    def _initialize_local_storage(self):
        """Initialize local storage fallback"""
        self.local_storage_path = os.path.abspath(self.local_storage_path)
        os.makedirs(self.local_storage_path, exist_ok=True)
        logger.info(f"Using local storage: {self.local_storage_path}")
    
    def _get_local_path(self, blob_name: str) -> str:
        """Get local file path for blob"""
        # Replace GCS path separators with local path separators
        local_path = os.path.join(self.local_storage_path, blob_name.replace('/', os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return local_path
    
    def upload_file(self, 
                   local_path: str, 
                   blob_name: str = None,
                   metadata: Dict[str, Any] = None,
                   content_type: str = None) -> str:
        """
        Upload file to cloud storage
        
        Args:
            local_path: Path to local file
            blob_name: Name for the blob (defaults to filename)
            metadata: Optional metadata dictionary
            content_type: Content type of the file
            
        Returns:
            Path to the uploaded file
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
        
        blob_name = blob_name or os.path.basename(local_path)
        
        if self.bucket is not None:
            try:
                # Upload to GCS
                blob = self.bucket.blob(blob_name)
                
                if content_type:
                    blob.content_type = content_type
                
                if metadata:
                    blob.metadata = metadata
                
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")
                return f"gs://{self.bucket_name}/{blob_name}"
                
            except GoogleCloudError as e:
                logger.error(f"GCS upload failed: {e}")
                if self.use_local_fallback:
                    return self._upload_local(local_path, blob_name, metadata)
                else:
                    raise
        else:
            return self._upload_local(local_path, blob_name, metadata)
    
    def _upload_local(self, local_path: str, blob_name: str, metadata: Dict[str, Any] = None) -> str:
        """Upload file to local storage"""
        local_storage_path = self._get_local_path(blob_name)
        
        # Copy file
        shutil.copy2(local_path, local_storage_path)
        
        # Save metadata
        if metadata:
            metadata_path = local_storage_path + '.metadata'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        logger.info(f"Uploaded {local_path} to local storage: {local_storage_path}")
        return local_storage_path
    
    def download_file(self, 
                     blob_name: str, 
                     local_path: str = None,
                     create_dirs: bool = True) -> str:
        """
        Download file from cloud storage
        
        Args:
            blob_name: Name of the blob to download
            local_path: Local path to save file (defaults to current directory)
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            Path to downloaded file
        """
        if local_path is None:
            local_path = os.path.basename(blob_name)
        
        if create_dirs:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        if self.bucket is not None:
            try:
                # Download from GCS
                blob = self.bucket.blob(blob_name)
                blob.download_to_filename(local_path)
                logger.info(f"Downloaded gs://{self.bucket_name}/{blob_name} to {local_path}")
                return local_path
                
            except NotFound:
                logger.warning(f"Blob not found in GCS: {blob_name}")
                if self.use_local_fallback:
                    return self._download_local(blob_name, local_path)
                else:
                    raise
            except GoogleCloudError as e:
                logger.error(f"GCS download failed: {e}")
                if self.use_local_fallback:
                    return self._download_local(blob_name, local_path)
                else:
                    raise
        else:
            return self._download_local(blob_name, local_path)
    
    def _download_local(self, blob_name: str, local_path: str) -> str:
        """Download file from local storage"""
        local_storage_path = self._get_local_path(blob_name)
        
        if not os.path.exists(local_storage_path):
            raise FileNotFoundError(f"File not found in local storage: {blob_name}")
        
        shutil.copy2(local_storage_path, local_path)
        logger.info(f"Downloaded {blob_name} from local storage to {local_path}")
        return local_path
    
    def upload_bytes(self, 
                    data: bytes, 
                    blob_name: str,
                    metadata: Dict[str, Any] = None,
                    content_type: str = None) -> str:
        """
        Upload bytes data to cloud storage
        
        Args:
            data: Bytes data to upload
            blob_name: Name for the blob
            metadata: Optional metadata dictionary
            content_type: Content type of the data
            
        Returns:
            Path to the uploaded data
        """
        if self.bucket is not None:
            try:
                # Upload to GCS
                blob = self.bucket.blob(blob_name)
                
                if content_type:
                    blob.content_type = content_type
                
                if metadata:
                    blob.metadata = metadata
                
                blob.upload_from_string(data)
                logger.info(f"Uploaded bytes to gs://{self.bucket_name}/{blob_name}")
                return f"gs://{self.bucket_name}/{blob_name}"
                
            except GoogleCloudError as e:
                logger.error(f"GCS upload failed: {e}")
                if self.use_local_fallback:
                    return self._upload_bytes_local(data, blob_name, metadata)
                else:
                    raise
        else:
            return self._upload_bytes_local(data, blob_name, metadata)
    
    def _upload_bytes_local(self, data: bytes, blob_name: str, metadata: Dict[str, Any] = None) -> str:
        """Upload bytes data to local storage"""
        local_storage_path = self._get_local_path(blob_name)
        
        # Save data
        with open(local_storage_path, 'wb') as f:
            f.write(data)
        
        # Save metadata
        if metadata:
            metadata_path = local_storage_path + '.metadata'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        logger.info(f"Uploaded bytes to local storage: {local_storage_path}")
        return local_storage_path
    
    def download_bytes(self, blob_name: str) -> bytes:
        """
        Download file as bytes from cloud storage
        
        Args:
            blob_name: Name of the blob to download
            
        Returns:
            Bytes data
        """
        if self.bucket is not None:
            try:
                # Download from GCS
                blob = self.bucket.blob(blob_name)
                data = blob.download_as_bytes()
                logger.info(f"Downloaded gs://{self.bucket_name}/{blob_name} as bytes")
                return data
                
            except NotFound:
                logger.warning(f"Blob not found in GCS: {blob_name}")
                if self.use_local_fallback:
                    return self._download_bytes_local(blob_name)
                else:
                    raise
            except GoogleCloudError as e:
                logger.error(f"GCS download failed: {e}")
                if self.use_local_fallback:
                    return self._download_bytes_local(blob_name)
                else:
                    raise
        else:
            return self._download_bytes_local(blob_name)
    
    def _download_bytes_local(self, blob_name: str) -> bytes:
        """Download bytes data from local storage"""
        local_storage_path = self._get_local_path(blob_name)
        
        if not os.path.exists(local_storage_path):
            raise FileNotFoundError(f"File not found in local storage: {blob_name}")
        
        with open(local_storage_path, 'rb') as f:
            data = f.read()
        
        logger.info(f"Downloaded {blob_name} from local storage as bytes")
        return data
    
    def list_files(self, prefix: str = "", delimiter: str = None) -> List[Dict[str, Any]]:
        """
        List files in cloud storage
        
        Args:
            prefix: Prefix to filter files
            delimiter: Delimiter for hierarchical listing
            
        Returns:
            List of file information dictionaries
        """
        if self.bucket is not None:
            try:
                # List from GCS
                blobs = self.bucket.list_blobs(prefix=prefix, delimiter=delimiter)
                
                files = []
                for blob in blobs:
                    files.append({
                        'name': blob.name,
                        'size': blob.size,
                        'content_type': blob.content_type,
                        'time_created': blob.time_created,
                        'updated': blob.updated,
                        'metadata': blob.metadata or {}
                    })
                
                return files
                
            except GoogleCloudError as e:
                logger.error(f"GCS list failed: {e}")
                if self.use_local_fallback:
                    return self._list_files_local(prefix)
                else:
                    raise
        else:
            return self._list_files_local(prefix)
    
    def _list_files_local(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in local storage"""
        local_prefix = self._get_local_path(prefix)
        
        if not os.path.exists(local_prefix):
            return []
        
        files = []
        for root, dirs, filenames in os.walk(local_prefix):
            for filename in filenames:
                if not filename.endswith('.metadata'):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.local_storage_path)
                    
                    # Load metadata if exists
                    metadata_path = file_path + '.metadata'
                    metadata = {}
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass
                    
                    stat = os.stat(file_path)
                    files.append({
                        'name': rel_path.replace(os.sep, '/'),
                        'size': stat.st_size,
                        'content_type': metadata.get('content_type', 'application/octet-stream'),
                        'time_created': datetime.fromtimestamp(stat.st_ctime),
                        'updated': datetime.fromtimestamp(stat.st_mtime),
                        'metadata': metadata
                    })
        
        return files
    
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete file from cloud storage
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.bucket is not None:
            try:
                # Delete from GCS
                blob = self.bucket.blob(blob_name)
                blob.delete()
                logger.info(f"Deleted gs://{self.bucket_name}/{blob_name}")
                return True
                
            except NotFound:
                logger.warning(f"Blob not found in GCS: {blob_name}")
                if self.use_local_fallback:
                    return self._delete_file_local(blob_name)
                else:
                    return False
            except GoogleCloudError as e:
                logger.error(f"GCS delete failed: {e}")
                if self.use_local_fallback:
                    return self._delete_file_local(blob_name)
                else:
                    return False
        else:
            return self._delete_file_local(blob_name)
    
    def _delete_file_local(self, blob_name: str) -> bool:
        """Delete file from local storage"""
        local_storage_path = self._get_local_path(blob_name)
        
        if os.path.exists(local_storage_path):
            os.remove(local_storage_path)
            
            # Remove metadata file
            metadata_path = local_storage_path + '.metadata'
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            logger.info(f"Deleted {blob_name} from local storage")
            return True
        
        return False
    
    def get_file_info(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Get file information
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            File information dictionary or None if not found
        """
        if self.bucket is not None:
            try:
                # Get from GCS
                blob = self.bucket.blob(blob_name)
                blob.reload()
                
                if blob.exists():
                    return {
                        'name': blob.name,
                        'size': blob.size,
                        'content_type': blob.content_type,
                        'time_created': blob.time_created,
                        'updated': blob.updated,
                        'metadata': blob.metadata or {}
                    }
                else:
                    return None
                    
            except GoogleCloudError as e:
                logger.error(f"GCS get info failed: {e}")
                if self.use_local_fallback:
                    return self._get_file_info_local(blob_name)
                else:
                    return None
        else:
            return self._get_file_info_local(blob_name)
    
    def _get_file_info_local(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get file information from local storage"""
        local_storage_path = self._get_local_path(blob_name)
        
        if not os.path.exists(local_storage_path):
            return None
        
        # Load metadata
        metadata = {}
        metadata_path = local_storage_path + '.metadata'
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        stat = os.stat(local_storage_path)
        return {
            'name': blob_name,
            'size': stat.st_size,
            'content_type': metadata.get('content_type', 'application/octet-stream'),
            'time_created': datetime.fromtimestamp(stat.st_ctime),
            'updated': datetime.fromtimestamp(stat.st_mtime),
            'metadata': metadata
        }
    
    def copy_file(self, source_blob_name: str, destination_blob_name: str) -> bool:
        """
        Copy file within cloud storage
        
        Args:
            source_blob_name: Source blob name
            destination_blob_name: Destination blob name
            
        Returns:
            True if successful, False otherwise
        """
        if self.bucket is not None:
            try:
                # Copy in GCS
                source_blob = self.bucket.blob(source_blob_name)
                destination_blob = self.bucket.blob(destination_blob_name)
                
                destination_blob.upload_from_string(source_blob.download_as_bytes())
                
                # Copy metadata
                if source_blob.metadata:
                    destination_blob.metadata = source_blob.metadata.copy()
                    destination_blob.patch()
                
                logger.info(f"Copied gs://{self.bucket_name}/{source_blob_name} to gs://{self.bucket_name}/{destination_blob_name}")
                return True
                
            except GoogleCloudError as e:
                logger.error(f"GCS copy failed: {e}")
                if self.use_local_fallback:
                    return self._copy_file_local(source_blob_name, destination_blob_name)
                else:
                    return False
        else:
            return self._copy_file_local(source_blob_name, destination_blob_name)
    
    def _copy_file_local(self, source_blob_name: str, destination_blob_name: str) -> bool:
        """Copy file in local storage"""
        source_path = self._get_local_path(source_blob_name)
        destination_path = self._get_local_path(destination_blob_name)
        
        if not os.path.exists(source_path):
            return False
        
        shutil.copy2(source_path, destination_path)
        
        # Copy metadata
        source_metadata_path = source_path + '.metadata'
        destination_metadata_path = destination_path + '.metadata'
        
        if os.path.exists(source_metadata_path):
            shutil.copy2(source_metadata_path, destination_metadata_path)
        
        logger.info(f"Copied {source_blob_name} to {destination_blob_name} in local storage")
        return True
    
    def get_signed_url(self, blob_name: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate signed URL for blob access
        
        Args:
            blob_name: Name of the blob
            expiration: URL expiration time in seconds
            
        Returns:
            Signed URL or None if not available
        """
        if self.bucket is not None:
            try:
                blob = self.bucket.blob(blob_name)
                url = blob.generate_signed_url(expiration=expiration)
                return url
            except Exception as e:
                logger.error(f"Failed to generate signed URL: {e}")
                return None
        else:
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Storage statistics dictionary
        """
        if self.bucket is not None:
            try:
                # Get GCS stats
                stats = {
                    'provider': 'google_cloud_storage',
                    'bucket_name': self.bucket_name,
                    'total_files': 0,
                    'total_size': 0,
                    'last_updated': datetime.now()
                }
                
                # Count files and calculate total size
                blobs = self.bucket.list_blobs()
                for blob in blobs:
                    stats['total_files'] += 1
                    if blob.size:
                        stats['total_size'] += blob.size
                
                return stats
                
            except GoogleCloudError as e:
                logger.error(f"GCS stats failed: {e}")
                if self.use_local_fallback:
                    return self._get_local_storage_stats()
                else:
                    return {'provider': 'error', 'error': str(e)}
        else:
            return self._get_local_storage_stats()
    
    def _get_local_storage_stats(self) -> Dict[str, Any]:
        """Get local storage statistics"""
        stats = {
            'provider': 'local_filesystem',
            'storage_path': self.local_storage_path,
            'total_files': 0,
            'total_size': 0,
            'last_updated': datetime.now()
        }
        
        if os.path.exists(self.local_storage_path):
            for root, dirs, files in os.walk(self.local_storage_path):
                for file in files:
                    if not file.endswith('.metadata'):
                        stats['total_files'] += 1
                        file_path = os.path.join(root, file)
                        try:
                            stats['total_size'] += os.path.getsize(file_path)
                        except:
                            pass
        
        return stats