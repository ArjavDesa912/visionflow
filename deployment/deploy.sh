#!/bin/bash

# VisionFlow Pro Deployment Script
# This script deploys VisionFlow Pro to Google Cloud Platform

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"visionflow-pro-$(date +%s)"}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"us-central1-a"}
SERVICE_NAME=${SERVICE_NAME:-"visionflow-api"}
FRONTEND_NAME=${FRONTEND_NAME:-"visionflow-frontend"}
IMAGE_NAME=${IMAGE_NAME:-"visionflow-pro"}
REPOSITORY=${REPOSITORY:-"visionflow-repo"}

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
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        error "gcloud is not installed. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error "docker is not installed. Please install Docker."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        warn "kubectl is not installed. Kubernetes features will not be available."
    fi
    
    log "Prerequisites check completed."
}

# Initialize GCP project
initialize_project() {
    log "Initializing GCP project..."
    
    # Set project
    gcloud config set project ${PROJECT_ID}
    
    # Enable required APIs
    log "Enabling required APIs..."
    gcloud services enable \
        run.googleapis.com \
        sql-component.googleapis.com \
        storage.googleapis.com \
        pubsub.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        container.googleapis.com \
        cloudbuild.googleapis.com \
        secretmanager.googleapis.com \
        iam.googleapis.com \
        --project=${PROJECT_ID}
    
    log "GCP project initialized."
}

# Create service account
create_service_account() {
    log "Creating service account..."
    
    SERVICE_ACCOUNT="${SERVICE_NAME}-sa"
    
    # Create service account
    gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
        --display-name="VisionFlow Pro Service Account" \
        --project=${PROJECT_ID}
    
    # Grant necessary roles
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/editor"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/run.admin"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/storage.admin"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/cloudsql.admin"
    
    # Create and download service account key
    gcloud iam service-accounts keys create key.json \
        --iam-account="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    log "Service account created."
}

# Setup Cloud SQL
setup_database() {
    log "Setting up Cloud SQL..."
    
    DATABASE_INSTANCE="visionflow-db"
    DATABASE_NAME="visionflow"
    DATABASE_USER="visionflow_user"
    DATABASE_PASSWORD=$(openssl rand -base64 32)
    
    # Create Cloud SQL instance
    gcloud sql instances create ${DATABASE_INSTANCE} \
        --database-version=POSTGRES_13 \
        --tier=db-g1-small \
        --region=${REGION} \
        --project=${PROJECT_ID}
    
    # Create database
    gcloud sql databases create ${DATABASE_NAME} \
        --instance=${DATABASE_INSTANCE} \
        --project=${PROJECT_ID}
    
    # Create user
    gcloud sql users create ${DATABASE_USER} \
        --password=${DATABASE_PASSWORD} \
        --instance=${DATABASE_INSTANCE} \
        --project=${PROJECT_ID}
    
    # Store database password in Secret Manager
    echo "${DATABASE_PASSWORD}" | gcloud secrets create database-password --data-file=-
    
    log "Cloud SQL setup completed."
}

# Setup Cloud Storage
setup_storage() {
    log "Setting up Cloud Storage..."
    
    STORAGE_BUCKET="visionflow-storage-$(date +%s)"
    MODELS_BUCKET="visionflow-models-$(date +%s)"
    
    # Create storage buckets
    gsutil mb -l ${REGION} gs://${STORAGE_BUCKET}
    gsutil mb -l ${REGION} gs://${MODELS_BUCKET}
    
    # Set bucket permissions
    gsutil iam ch allUsers:objectViewer gs://${STORAGE_BUCKET}
    gsutil iam ch allUsers:objectViewer gs://${MODELS_BUCKET}
    
    # Store bucket names in Secret Manager
    echo "${STORAGE_BUCKET}" | gcloud secrets create storage-bucket --data-file=-
    echo "${MODELS_BUCKET}" | gcloud secrets create models-bucket --data-file=-
    
    log "Cloud Storage setup completed."
}

# Build and push Docker images
build_and_push_images() {
    log "Building and pushing Docker images..."
    
    # Enable Container Registry API
    gcloud services enable containerregistry.googleapis.com
    
    # Configure Docker
    gcloud auth configure-docker
    
    # Build API image
    log "Building API image..."
    docker build -t gcr.io/${PROJECT_ID}/${IMAGE_NAME}-api:latest ./api
    
    # Build frontend image
    log "Building frontend image..."
    docker build -t gcr.io/${PROJECT_ID}/${IMAGE_NAME}-frontend:latest ./frontend
    
    # Push images
    log "Pushing images to Container Registry..."
    docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}-api:latest
    docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}-frontend:latest
    
    log "Images built and pushed successfully."
}

# Deploy to Cloud Run
deploy_to_cloud_run() {
    log "Deploying to Cloud Run..."
    
    # Get secrets
    DATABASE_URL=$(gcloud secrets versions access latest --secret=database-url)
    STORAGE_BUCKET=$(gcloud secrets versions access latest --secret=storage-bucket)
    MODELS_BUCKET=$(gcloud secrets versions access latest --secret=models-bucket)
    
    # Deploy API
    log "Deploying API..."
    gcloud run deploy ${SERVICE_NAME} \
        --image=gcr.io/${PROJECT_ID}/${IMAGE_NAME}-api:latest \
        --platform=managed \
        --region=${REGION} \
        --allow-unauthenticated \
        --memory=4Gi \
        --cpu=2 \
        --max-instances=10 \
        --min-instances=1 \
        --set-env-vars="DATABASE_URL=${DATABASE_URL}" \
        --set-env-vars="STORAGE_BUCKET=${STORAGE_BUCKET}" \
        --set-env-vars="MODELS_BUCKET=${MODELS_BUCKET}" \
        --set-env-vars="REDIS_URL=redis://redis:6379" \
        --set-env-vars="API_HOST=0.0.0.0" \
        --set-env-vars="API_PORT=8080" \
        --service-account="${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Deploy Frontend
    log "Deploying Frontend..."
    API_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
    
    gcloud run deploy ${FRONTEND_NAME} \
        --image=gcr.io/${PROJECT_ID}/${IMAGE_NAME}-frontend:latest \
        --platform=managed \
        --region=${REGION} \
        --allow-unauthenticated \
        --memory=2Gi \
        --cpu=1 \
        --max-instances=5 \
        --min-instances=1 \
        --set-env-vars="API_BASE_URL=${API_URL}" \
        --set-env-vars="STREAMLIT_SERVER_PORT=8501" \
        --set-env-vars="STREAMLIT_SERVER_ADDRESS=0.0.0.0" \
        --service-account="${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    
    log "Deployment completed successfully."
}

# Setup monitoring and logging
setup_monitoring() {
    log "Setting up monitoring and logging..."
    
    # Create dashboard
    gcloud monitoring dashboards create --config-from-file=deployment/dashboard.json
    
    # Create alert policies
    gcloud alpha monitoring policies create --policy-from-file=deployment/alerts.json
    
    # Create log-based metrics
    gcloud logging metrics create api_error_count \
        --description="Count of API errors" \
        --log-filter="resource.type=\"cloud_run_revision\" AND jsonPayload.level=\"ERROR\""
    
    log "Monitoring setup completed."
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Run unit tests
    python -m pytest tests/ -v
    
    # Run integration tests
    python -m pytest tests/integration/ -v
    
    log "Tests completed."
}

# Main deployment function
deploy() {
    log "Starting VisionFlow Pro deployment..."
    
    check_prerequisites
    initialize_project
    create_service_account
    setup_database
    setup_storage
    build_and_push_images
    deploy_to_cloud_run
    setup_monitoring
    
    log "Deployment completed successfully!"
    
    # Print deployment information
    API_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
    FRONTEND_URL=$(gcloud run services describe ${FRONTEND_NAME} --region=${REGION} --format='value(status.url)')
    
    echo -e "\n${GREEN}=== Deployment Information ===${NC}"
    echo -e "Project ID: ${PROJECT_ID}"
    echo -e "API URL: ${API_URL}"
    echo -e "Frontend URL: ${FRONTEND_URL}"
    echo -e "Region: ${REGION}"
    echo -e "${GREEN}==============================${NC}"
}

# Handle command line arguments
case "${1:-}" in
    "deploy")
        deploy
        ;;
    "test")
        run_tests
        ;;
    "init")
        check_prerequisites
        initialize_project
        ;;
    "build")
        build_and_push_images
        ;;
    "monitoring")
        setup_monitoring
        ;;
    *)
        echo "Usage: $0 {deploy|test|init|build|monitoring}"
        exit 1
        ;;
esac