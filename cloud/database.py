"""
Database Integration for VisionFlow Pro

This module provides integration with PostgreSQL and other databases
for storing user data, analytics, and application state.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import hashlib
import uuid

# Database libraries
try:
    import sqlalchemy
    from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, JSON, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session, relationship
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.dialects.postgresql import UUID
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    sqlalchemy = None

# Local fallback
import sqlite3
import pickle

logger = logging.getLogger(__name__)

# Database models (if SQLAlchemy is available)
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class User(Base):
        __tablename__ = 'users'
        
        id = Column(Integer, primary_key=True)
        username = Column(String(50), unique=True, nullable=False)
        email = Column(String(100), unique=True, nullable=False)
        password_hash = Column(String(255), nullable=False)
        full_name = Column(String(100))
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Relationships
        api_keys = relationship("APIKey", back_populates="user")
        usage_records = relationship("UsageRecord", back_populates="user")
        search_queries = relationship("SearchQuery", back_populates="user")
    
    class APIKey(Base):
        __tablename__ = 'api_keys'
        
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, ForeignKey('users.id'))
        key = Column(String(255), unique=True, nullable=False)
        name = Column(String(100), nullable=False)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        expires_at = Column(DateTime)
        last_used = Column(DateTime)
        
        # Relationships
        user = relationship("User", back_populates="api_keys")
        usage_records = relationship("UsageRecord", back_populates="api_key")
    
    class UsageRecord(Base):
        __tablename__ = 'usage_records'
        
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, ForeignKey('users.id'))
        api_key_id = Column(Integer, ForeignKey('api_keys.id'))
        endpoint = Column(String(100), nullable=False)
        method = Column(String(10), nullable=False)
        response_time = Column(Float)
        status_code = Column(Integer)
        request_size = Column(Integer)
        response_size = Column(Integer)
        timestamp = Column(DateTime, default=datetime.utcnow)
        metadata = Column(JSON)
        
        # Relationships
        user = relationship("User", back_populates="usage_records")
        api_key = relationship("APIKey", back_populates="usage_records")
    
    class SearchQuery(Base):
        __tablename__ = 'search_queries'
        
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, ForeignKey('users.id'))
        query_type = Column(String(20), nullable=False)  # 'text' or 'image'
        query_text = Column(Text)
        query_image_path = Column(String(255))
        results_count = Column(Integer, default=0)
        processing_time = Column(Float)
        timestamp = Column(DateTime, default=datetime.utcnow)
        metadata = Column(JSON)
        
        # Relationships
        user = relationship("User", back_populates="search_queries")
    
    class ModelPerformance(Base):
        __tablename__ = 'model_performance'
        
        id = Column(Integer, primary_key=True)
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50))
        accuracy = Column(Float)
        precision = Column(Float)
        recall = Column(Float)
        f1_score = Column(Float)
        latency = Column(Float)
        throughput = Column(Float)
        timestamp = Column(DateTime, default=datetime.utcnow)
        metadata = Column(JSON)
    
    class SystemMetrics(Base):
        __tablename__ = 'system_metrics'
        
        id = Column(Integer, primary_key=True)
        metric_name = Column(String(100), nullable=False)
        metric_value = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow)
        metadata = Column(JSON)
    
    class JobRecord(Base):
        __tablename__ = 'job_records'
        
        id = Column(Integer, primary_key=True)
        job_id = Column(String(100), unique=True, nullable=False)
        job_type = Column(String(50), nullable=False)
        status = Column(String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
        progress = Column(Float, default=0.0)
        parameters = Column(JSON)
        result = Column(JSON)
        error_message = Column(Text)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        started_at = Column(DateTime)
        completed_at = Column(DateTime)

class DatabaseManager:
    """Manages database operations for VisionFlow Pro"""
    
    def __init__(self, 
                 database_url: str = None,
                 use_local_fallback: bool = True,
                 echo: bool = False):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
            use_local_fallback: Whether to use local SQLite if PostgreSQL is unavailable
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///visionflow.db')
        self.use_local_fallback = use_local_fallback
        self.echo = echo
        
        # Initialize database
        self.engine = None
        self.SessionLocal = None
        self.local_db_path = None
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available, using local fallback")
            self._initialize_local_database()
            return
        
        try:
            # Create engine
            self.engine = create_engine(self.database_url, echo=self.echo)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Test connection
            with self.get_session() as session:
                session.execute(sqlalchemy.text("SELECT 1"))
            
            logger.info(f"Connected to database: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            if self.use_local_fallback:
                self._initialize_local_database()
            else:
                raise
    
    def _initialize_local_database(self):
        """Initialize local SQLite database"""
        # Extract database name from URL
        if self.database_url.startswith('sqlite:///'):
            self.local_db_path = self.database_url[10:]  # Remove 'sqlite:///'
        else:
            self.local_db_path = 'visionflow_local.db'
        
        # Create tables in SQLite
        self._create_local_tables()
        logger.info(f"Using local SQLite database: {self.local_db_path}")
    
    def _create_local_tables(self):
        """Create tables in local SQLite database"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # Create tables
        tables = {
            'users': '''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'api_keys': '''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    last_used DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''',
            'usage_records': '''
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    api_key_id INTEGER,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    response_time REAL,
                    status_code INTEGER,
                    request_size INTEGER,
                    response_size INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (api_key_id) REFERENCES api_keys (id)
                )
            ''',
            'search_queries': '''
                CREATE TABLE IF NOT EXISTS search_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    query_type TEXT NOT NULL,
                    query_text TEXT,
                    query_image_path TEXT,
                    results_count INTEGER DEFAULT 0,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''',
            'model_performance': '''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    latency REAL,
                    throughput REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''',
            'system_metrics': '''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''',
            'job_records': '''
                CREATE TABLE IF NOT EXISTS job_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    parameters TEXT,
                    result TEXT,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    started_at DATETIME,
                    completed_at DATETIME
                )
            '''
        }
        
        for table_name, query in tables.items():
            cursor.execute(query)
        
        conn.commit()
        conn.close()
    
    @contextmanager
    def get_session(self):
        """Get database session context manager"""
        if self.SessionLocal:
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()
        else:
            # Use local SQLite connection
            conn = sqlite3.connect(self.local_db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def create_user(self, username: str, email: str, password: str, full_name: str = None) -> Dict[str, Any]:
        """
        Create a new user
        
        Args:
            username: Username
            email: Email address
            password: Password (will be hashed)
            full_name: Full name
            
        Returns:
            User information dictionary
        """
        password_hash = self._hash_password(password)
        
        if self.SessionLocal:
            with self.get_session() as session:
                user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    full_name=full_name
                )
                session.add(user)
                session.flush()
                
                return {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'is_active': user.is_active,
                    'created_at': user.created_at
                }
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
                    (username, email, password_hash, full_name)
                )
                user_id = cursor.lastrowid
                
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                return dict(row)
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username
        
        Args:
            username: Username
            
        Returns:
            User information dictionary or None
        """
        if self.SessionLocal:
            with self.get_session() as session:
                user = session.query(User).filter(User.username == username).first()
                if user:
                    return {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'full_name': user.full_name,
                        'is_active': user.is_active,
                        'created_at': user.created_at
                    }
                return None
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                return dict(row) if row else None
    
    def create_api_key(self, user_id: int, name: str, expires_in_days: int = 30) -> Dict[str, Any]:
        """
        Create API key for user
        
        Args:
            user_id: User ID
            name: API key name
            expires_in_days: Number of days until key expires
            
        Returns:
            API key information dictionary
        """
        api_key = self._generate_api_key()
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None
        
        if self.SessionLocal:
            with self.get_session() as session:
                key = APIKey(
                    user_id=user_id,
                    key=api_key,
                    name=name,
                    expires_at=expires_at
                )
                session.add(key)
                session.flush()
                
                return {
                    'id': key.id,
                    'key': key.key,
                    'name': key.name,
                    'is_active': key.is_active,
                    'created_at': key.created_at,
                    'expires_at': key.expires_at
                }
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO api_keys (user_id, key, name, expires_at) VALUES (?, ?, ?, ?)",
                    (user_id, api_key, name, expires_at)
                )
                key_id = cursor.lastrowid
                
                cursor.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,))
                row = cursor.fetchone()
                
                return dict(row)
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            API key information dictionary or None if invalid
        """
        if self.SessionLocal:
            with self.get_session() as session:
                key = session.query(APIKey).filter(
                    APIKey.key == api_key,
                    APIKey.is_active == True,
                    (APIKey.expires_at.is_(None) | (APIKey.expires_at > datetime.utcnow()))
                ).first()
                
                if key:
                    # Update last used timestamp
                    key.last_used = datetime.utcnow()
                    session.commit()
                    
                    return {
                        'id': key.id,
                        'user_id': key.user_id,
                        'key': key.key,
                        'name': key.name,
                        'is_active': key.is_active,
                        'created_at': key.created_at,
                        'expires_at': key.expires_at,
                        'last_used': key.last_used
                    }
                return None
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM api_keys WHERE key = ? AND is_active = 1 AND (expires_at IS NULL OR expires_at > ?)",
                    (api_key, datetime.utcnow())
                )
                row = cursor.fetchone()
                
                if row:
                    # Update last used timestamp
                    cursor.execute(
                        "UPDATE api_keys SET last_used = ? WHERE id = ?",
                        (datetime.utcnow(), row['id'])
                    )
                    conn.commit()
                    
                    return dict(row)
                return None
    
    def record_usage(self, user_id: int, api_key_id: int, endpoint: str, method: str,
                    response_time: float = None, status_code: int = None,
                    request_size: int = None, response_size: int = None,
                    metadata: Dict[str, Any] = None) -> None:
        """
        Record API usage
        
        Args:
            user_id: User ID
            api_key_id: API key ID
            endpoint: API endpoint
            method: HTTP method
            response_time: Response time in seconds
            status_code: HTTP status code
            request_size: Request size in bytes
            response_size: Response size in bytes
            metadata: Additional metadata
        """
        if self.SessionLocal:
            with self.get_session() as session:
                record = UsageRecord(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    endpoint=endpoint,
                    method=method,
                    response_time=response_time,
                    status_code=status_code,
                    request_size=request_size,
                    response_size=response_size,
                    metadata=metadata
                )
                session.add(record)
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO usage_records 
                       (user_id, api_key_id, endpoint, method, response_time, status_code, 
                        request_size, response_size, metadata) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, api_key_id, endpoint, method, response_time, status_code,
                     request_size, response_size, json.dumps(metadata) if metadata else None)
                )
    
    def record_search_query(self, user_id: int, query_type: str, query_text: str = None,
                          query_image_path: str = None, results_count: int = 0,
                          processing_time: float = None, metadata: Dict[str, Any] = None) -> None:
        """
        Record search query
        
        Args:
            user_id: User ID
            query_type: Query type ('text' or 'image')
            query_text: Query text
            query_image_path: Query image path
            results_count: Number of results
            processing_time: Processing time in seconds
            metadata: Additional metadata
        """
        if self.SessionLocal:
            with self.get_session() as session:
                query = SearchQuery(
                    user_id=user_id,
                    query_type=query_type,
                    query_text=query_text,
                    query_image_path=query_image_path,
                    results_count=results_count,
                    processing_time=processing_time,
                    metadata=metadata
                )
                session.add(query)
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO search_queries 
                       (user_id, query_type, query_text, query_image_path, results_count, 
                        processing_time, metadata) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, query_type, query_text, query_image_path, results_count,
                     processing_time, json.dumps(metadata) if metadata else None)
                )
    
    def record_model_performance(self, model_name: str, model_version: str = None,
                               accuracy: float = None, precision: float = None,
                               recall: float = None, f1_score: float = None,
                               latency: float = None, throughput: float = None,
                               metadata: Dict[str, Any] = None) -> None:
        """
        Record model performance metrics
        
        Args:
            model_name: Model name
            model_version: Model version
            accuracy: Accuracy metric
            precision: Precision metric
            recall: Recall metric
            f1_score: F1 score metric
            latency: Latency metric
            throughput: Throughput metric
            metadata: Additional metadata
        """
        if self.SessionLocal:
            with self.get_session() as session:
                performance = ModelPerformance(
                    model_name=model_name,
                    model_version=model_version,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    latency=latency,
                    throughput=throughput,
                    metadata=metadata
                )
                session.add(performance)
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO model_performance 
                       (model_name, model_version, accuracy, precision, recall, f1_score, 
                        latency, throughput, metadata) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (model_name, model_version, accuracy, precision, recall, f1_score,
                     latency, throughput, json.dumps(metadata) if metadata else None)
                )
    
    def record_system_metric(self, metric_name: str, metric_value: float,
                           metadata: Dict[str, Any] = None) -> None:
        """
        Record system metric
        
        Args:
            metric_name: Metric name
            metric_value: Metric value
            metadata: Additional metadata
        """
        if self.SessionLocal:
            with self.get_session() as session:
                metric = SystemMetrics(
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metadata=metadata
                )
                session.add(metric)
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO system_metrics (metric_name, metric_value, metadata) VALUES (?, ?, ?)",
                    (metric_name, metric_value, json.dumps(metadata) if metadata else None)
                )
    
    def create_job(self, job_id: str, job_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new job record
        
        Args:
            job_id: Job ID
            job_type: Job type
            parameters: Job parameters
            
        Returns:
            Job information dictionary
        """
        if self.SessionLocal:
            with self.get_session() as session:
                job = JobRecord(
                    job_id=job_id,
                    job_type=job_type,
                    parameters=parameters
                )
                session.add(job)
                session.flush()
                
                return {
                    'id': job.id,
                    'job_id': job.job_id,
                    'job_type': job.job_type,
                    'status': job.status,
                    'progress': job.progress,
                    'parameters': job.parameters,
                    'created_at': job.created_at
                }
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO job_records (job_id, job_type, parameters) VALUES (?, ?, ?)",
                    (job_id, job_type, json.dumps(parameters) if parameters else None)
                )
                job_db_id = cursor.lastrowid
                
                cursor.execute("SELECT * FROM job_records WHERE id = ?", (job_db_id,))
                row = cursor.fetchone()
                
                return dict(row)
    
    def update_job(self, job_id: str, status: str = None, progress: float = None,
                  result: Dict[str, Any] = None, error_message: str = None) -> bool:
        """
        Update job record
        
        Args:
            job_id: Job ID
            status: Job status
            progress: Job progress (0.0 to 1.0)
            result: Job result
            error_message: Error message
            
        Returns:
            True if successful, False otherwise
        """
        update_fields = []
        update_values = []
        
        if status is not None:
            update_fields.append("status = ?")
            update_values.append(status)
        
        if progress is not None:
            update_fields.append("progress = ?")
            update_values.append(progress)
        
        if result is not None:
            update_fields.append("result = ?")
            update_values.append(json.dumps(result))
        
        if error_message is not None:
            update_fields.append("error_message = ?")
            update_values.append(error_message)
        
        if not update_fields:
            return True
        
        update_fields.append("updated_at = ?")
        update_values.append(datetime.utcnow())
        update_values.append(job_id)
        
        if self.SessionLocal:
            with self.get_session() as session:
                rows_affected = session.query(JobRecord).filter(
                    JobRecord.job_id == job_id
                ).update({
                    field: value for field, value in [
                        ('status', status),
                        ('progress', progress),
                        ('result', result),
                        ('error_message', error_message),
                        ('updated_at', datetime.utcnow())
                    ] if value is not None
                })
                return rows_affected > 0
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                query = f"UPDATE job_records SET {', '.join(update_fields)} WHERE job_id = ?"
                cursor.execute(query, update_values)
                return cursor.rowcount > 0
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job record
        
        Args:
            job_id: Job ID
            
        Returns:
            Job information dictionary or None
        """
        if self.SessionLocal:
            with self.get_session() as session:
                job = session.query(JobRecord).filter(JobRecord.job_id == job_id).first()
                if job:
                    return {
                        'id': job.id,
                        'job_id': job.job_id,
                        'job_type': job.job_type,
                        'status': job.status,
                        'progress': job.progress,
                        'parameters': job.parameters,
                        'result': job.result,
                        'error_message': job.error_message,
                        'created_at': job.created_at,
                        'updated_at': job.updated_at,
                        'started_at': job.started_at,
                        'completed_at': job.completed_at
                    }
                return None
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM job_records WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
    
    def get_usage_stats(self, user_id: int = None, days: int = 30) -> Dict[str, Any]:
        """
        Get usage statistics
        
        Args:
            user_id: User ID (optional, for user-specific stats)
            days: Number of days to include in stats
            
        Returns:
            Usage statistics dictionary
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        if self.SessionLocal:
            with self.get_session() as session:
                query = session.query(UsageRecord).filter(
                    UsageRecord.timestamp >= start_date
                )
                
                if user_id:
                    query = query.filter(UsageRecord.user_id == user_id)
                
                records = query.all()
                
                stats = {
                    'total_requests': len(records),
                    'average_response_time': sum(r.response_time for r in records if r.response_time) / len(records) if records else 0,
                    'success_rate': len([r for r in records if r.status_code and 200 <= r.status_code < 300]) / len(records) if records else 0,
                    'total_data_transferred': sum(r.response_size for r in records if r.response_size),
                    'endpoints': {}
                }
                
                # Group by endpoint
                for record in records:
                    if record.endpoint not in stats['endpoints']:
                        stats['endpoints'][record.endpoint] = {
                            'count': 0,
                            'avg_response_time': 0,
                            'success_rate': 0
                        }
                    
                    stats['endpoints'][record.endpoint]['count'] += 1
                
                return stats
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute(
                        "SELECT * FROM usage_records WHERE user_id = ? AND timestamp >= ?",
                        (user_id, start_date)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM usage_records WHERE timestamp >= ?",
                        (start_date,)
                    )
                
                records = cursor.fetchall()
                
                stats = {
                    'total_requests': len(records),
                    'average_response_time': sum(r['response_time'] for r in records if r['response_time']) / len(records) if records else 0,
                    'success_rate': len([r for r in records if r['status_code'] and 200 <= r['status_code'] < 300]) / len(records) if records else 0,
                    'total_data_transferred': sum(r['response_size'] for r in records if r['response_size']),
                    'endpoints': {}
                }
                
                # Group by endpoint
                for record in records:
                    endpoint = record['endpoint']
                    if endpoint not in stats['endpoints']:
                        stats['endpoints'][endpoint] = {
                            'count': 0,
                            'avg_response_time': 0,
                            'success_rate': 0
                        }
                    
                    stats['endpoints'][endpoint]['count'] += 1
                
                return stats
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"vf_{uuid.uuid4().hex}{uuid.uuid4().hex}"
    
    def cleanup_old_records(self, days: int = 90) -> int:
        """
        Clean up old records
        
        Args:
            days: Number of days to keep records
            
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        if self.SessionLocal:
            with self.get_session() as session:
                # Delete old usage records
                deleted_usage = session.query(UsageRecord).filter(
                    UsageRecord.timestamp < cutoff_date
                ).delete()
                
                # Delete old search queries
                deleted_search = session.query(SearchQuery).filter(
                    SearchQuery.timestamp < cutoff_date
                ).delete()
                
                # Delete old system metrics
                deleted_metrics = session.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < cutoff_date
                ).delete()
                
                return deleted_usage + deleted_search + deleted_metrics
        else:
            # Use local SQLite
            with self.get_session() as conn:
                cursor = conn.cursor()
                
                # Delete old usage records
                cursor.execute(
                    "DELETE FROM usage_records WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted_usage = cursor.rowcount
                
                # Delete old search queries
                cursor.execute(
                    "DELETE FROM search_queries WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted_search = cursor.rowcount
                
                # Delete old system metrics
                cursor.execute(
                    "DELETE FROM system_metrics WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted_metrics = cursor.rowcount
                
                return deleted_usage + deleted_search + deleted_metrics