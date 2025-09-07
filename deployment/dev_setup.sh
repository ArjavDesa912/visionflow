#!/bin/bash

# VisionFlow Pro Local Development Setup Script
# This script sets up VisionFlow Pro for local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Python 3.8+ is installed
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.8"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log "Python version ${PYTHON_VERSION} is compatible"
    else
        error "Python 3.8 or higher is required. Found version ${PYTHON_VERSION}"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is not installed. Please install pip."
        exit 1
    fi
    
    # Check if virtualenv is installed
    if ! command -v virtualenv &> /dev/null; then
        warn "virtualenv is not installed. Installing..."
        pip3 install virtualenv
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        warn "Docker is not installed. Some features may not be available."
    fi
    
    # Check if Git is installed
    if ! command -v git &> /dev/null; then
        warn "Git is not installed. Version control features may not be available."
    fi
    
    log "Prerequisites check completed."
}

# Create virtual environment
create_virtualenv() {
    log "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log "Virtual environment created."
    else
        log "Virtual environment already exists."
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    log "Virtual environment activated."
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install main requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # Install API requirements
    if [ -f "api/requirements.txt" ]; then
        pip install -r api/requirements.txt
    fi
    
    # Install frontend requirements
    if [ -f "frontend/requirements.txt" ]; then
        pip install -r frontend/requirements.txt
    fi
    
    # Install development dependencies
    if [ -f "deployment/dev-requirements.txt" ]; then
        pip install -r deployment/dev-requirements.txt
    fi
    
    log "Dependencies installed."
}

# Setup configuration files
setup_configuration() {
    log "Setting up configuration files..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# VisionFlow Pro Development Configuration
DATABASE_URL=postgresql://visionflow:visionflow@localhost:5432/visionflow
REDIS_URL=redis://localhost:6379
STORAGE_BUCKET=visionflow-storage-local
MODELS_BUCKET=visionflow-models-local
API_BASE_URL=http://localhost:8080
DEBUG=True
LOG_LEVEL=DEBUG
EOF
        log ".env file created."
    else
        log ".env file already exists."
    fi
    
    # Create config.yaml for visionflow
    if [ ! -f "visionflow/config/config.yaml" ]; then
        mkdir -p visionflow/config
        cat > visionflow/config/config.yaml << EOF
# VisionFlow Pro Configuration
models:
  device: auto
  object_detection:
    model: yolov8n
    confidence_threshold: 0.5
    nms_threshold: 0.4
  classification:
    model: efficientnet-b0
    num_classes: 1000
  deepfake_detection:
    model: vit_base_patch16_224
    confidence_threshold: 0.8

video:
  frame_rate: 30
  resolution: [640, 480]
  batch_size: 4
  temporal_consistency: true

search:
  embedding_dim: 512
  index_type: hnsw
  ef_construction: 100
  ef_search: 50
  max_connections: 32
  max_elements: 10000
  use_gpu: false

paths:
  models_dir: ./models
  data_dir: ./data
  cache_dir: ./cache
  output_dir: ./output
  logs_dir: ./logs
EOF
        log "visionflow configuration file created."
    fi
    
    # Create development database configuration
    if [ ! -f "deployment/dev-db.conf" ]; then
        mkdir -p deployment
        cat > deployment/dev-db.conf << EOF
# Development Database Configuration
db_host=localhost
db_port=5432
db_name=visionflow
db_user=visionflow
db_password=visionflow
EOF
        log "Database configuration file created."
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "models"
        "data"
        "cache"
        "output"
        "logs"
        "temp"
        "uploads"
        "downloads"
        "deployment/backups"
        "deployment/scripts"
        "tests/fixtures"
        "docs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
    
    # Create .gitkeep files to preserve empty directories
    for dir in "${directories[@]}"; do
        if [ ! -f "$dir/.gitkeep" ]; then
            touch "$dir/.gitkeep"
        fi
    done
    
    log "Directories created."
}

# Setup database
setup_database() {
    log "Setting up database..."
    
    # Check if PostgreSQL is running
    if command -v pg_isready &> /dev/null; then
        if pg_isready -h localhost -p 5432; then
            log "PostgreSQL is running."
            
            # Create database if it doesn't exist
            psql -h localhost -p 5432 -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'visionflow'" | grep -q 1 || psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE visionflow"
            
            # Create user if it doesn't exist
            psql -h localhost -p 5432 -U postgres -tc "SELECT 1 FROM pg_roles WHERE rolname = 'visionflow'" | grep -q 1 || psql -h localhost -p 5432 -U postgres -c "CREATE USER visionflow WITH PASSWORD 'visionflow'"
            
            # Grant privileges
            psql -h localhost -p 5432 -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE visionflow TO visionflow"
            
            log "Database setup completed."
        else
            warn "PostgreSQL is not running. Please start PostgreSQL and run setup again."
        fi
    else
        warn "PostgreSQL is not installed. Please install PostgreSQL for full functionality."
    fi
}

# Setup Redis
setup_redis() {
    log "Setting up Redis..."
    
    # Check if Redis is running
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping > /dev/null 2>&1; then
            log "Redis is running."
        else
            warn "Redis is not running. Please start Redis for caching functionality."
        fi
    else
        warn "Redis is not installed. Please install Redis for caching functionality."
    fi
}

# Download models
download_models() {
    log "Downloading models..."
    
    # Create models directory
    mkdir -p models
    
    # This would typically download models from HuggingFace or other sources
    # For now, we'll create placeholder files
    log "Note: Models will be downloaded automatically when first used."
    log "Placeholder files created in models directory."
}

# Run initial setup
run_initial_setup() {
    log "Running initial setup..."
    
    # Run the setup script
    if [ -f "setup.py" ]; then
        python setup.py
    fi
    
    # Run models setup
    if [ -f "setup_models.py" ]; then
        python setup_models.py
    fi
    
    log "Initial setup completed."
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Run unit tests
    if [ -f "tests/test_visionflow.py" ]; then
        python -m pytest tests/test_visionflow.py -v
    fi
    
    # Run integration tests
    if [ -d "tests/integration" ]; then
        python -m pytest tests/integration/ -v
    fi
    
    log "Tests completed."
}

# Create startup scripts
create_startup_scripts() {
    log "Creating startup scripts..."
    
    # Create API startup script
    cat > start_api.sh << 'EOF'
#!/bin/bash
# Start VisionFlow Pro API

source venv/bin/activate
cd api
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
EOF
    chmod +x start_api.sh
    
    # Create frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
# Start VisionFlow Pro Frontend

source venv/bin/activate
cd frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
EOF
    chmod +x start_frontend.sh
    
    # Create development startup script
    cat > dev_start.sh << 'EOF'
#!/bin/bash
# Start VisionFlow Pro Development Environment

echo "Starting VisionFlow Pro Development Environment..."

# Start API in background
echo "Starting API..."
./start_api.sh &
API_PID=$!

# Start Frontend in background
echo "Starting Frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

# Wait for both processes
wait $API_PID
wait $FRONTEND_PID
EOF
    chmod +x dev_start.sh
    
    log "Startup scripts created."
}

# Main setup function
setup() {
    log "Starting VisionFlow Pro local development setup..."
    
    check_prerequisites
    create_virtualenv
    install_dependencies
    setup_configuration
    create_directories
    setup_database
    setup_redis
    download_models
    run_initial_setup
    create_startup_scripts
    
    log "Setup completed successfully!"
    
    echo -e "\n${GREEN}=== Setup Information ===${NC}"
    echo -e "Virtual environment: ./venv"
    echo -e "Configuration files: .env, visionflow/config/config.yaml"
    echo -e "Start API: ./start_api.sh"
    echo -e "Start Frontend: ./start_frontend.sh"
    echo -e "Start Development: ./dev_start.sh"
    echo -e "API URL: http://localhost:8080"
    echo -e "Frontend URL: http://localhost:8501"
    echo -e "API Documentation: http://localhost:8080/docs"
    echo -e "${GREEN}======================${NC}"
    
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Start the development environment: ./dev_start.sh"
    echo "3. Open http://localhost:8501 in your browser"
    echo "4. Run tests: python -m pytest tests/"
}

# Handle command line arguments
case "${1:-}" in
    "setup")
        setup
        ;;
    "test")
        run_tests
        ;;
    "clean")
        log "Cleaning up..."
        rm -rf venv/
        rm -rf __pycache__/
        rm -rf .pytest_cache/
        rm -rf *.egg-info/
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -exec rm -rf {} +
        log "Cleanup completed."
        ;;
    "reset")
        warn "This will reset the entire setup. Are you sure? (y/N)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            ./dev_setup.sh clean
            rm -rf models/
            rm -rf data/
            rm -rf cache/
            rm -rf output/
            rm -rf logs/
            rm -f .env
            rm -f config.yaml
            setup
        else
            log "Reset cancelled."
        fi
        ;;
    *)
        echo "Usage: $0 {setup|test|clean|reset}"
        exit 1
        ;;
esac