# VisionFlow Pro GCP Deployment Setup

This directory contains configuration files and scripts for deploying VisionFlow Pro to Google Cloud Platform.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Cloud SQL     │
│   Frontend      │◄──►│   Backend       │◄──►│   Database      │
│   (Cloud Run)   │    │   (Cloud Run)   │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cloud Storage │    │   Cloud Pub/Sub │    │   Cloud         │
│   (Models/Files)│    │   (Events)      │    │   Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Services

### Core Services
- **Cloud Run**: Serverless container deployment for API and frontend
- **Cloud SQL**: Managed PostgreSQL database
- **Cloud Storage**: Object storage for models and files
- **Cloud Pub/Sub**: Event-driven architecture
- **Cloud Monitoring**: Application monitoring and logging

### Machine Learning Services
- **Vertex AI**: Model training and deployment
- **AI Platform**: Model serving and inference
- **Cloud TPUs/GPUs**: Accelerated computing

## Setup Instructions

1. **Initialize GCP Project**
   ```bash
   gcloud auth login
   gcloud config set project ${PROJECT_ID}
   gcloud services enable run.googleapis.com
   gcloud services enable sql-component.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

2. **Create Service Account**
   ```bash
   gcloud iam service-accounts create ${SERVICE_ACCOUNT}
   gcloud projects add-iam-policy-binding ${PROJECT_ID} \
       --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
       --role="roles/editor"
   ```

3. **Setup Database**
   ```bash
   gcloud sql instances create ${DATABASE_INSTANCE} \
       --database-version=POSTGRES_13 \
       --tier=db-g1-small \
       --region=${REGION}
   ```

4. **Create Storage Buckets**
   ```bash
   gsutil mb -l ${REGION} gs://${STORAGE_BUCKET}
   gsutil mb -l ${REGION} gs://${MODELS_BUCKET}
   ```

5. **Deploy Services**
   ```bash
   # Deploy API
   gcloud run deploy ${SERVICE_NAME} \
       --image=gcr.io/${PROJECT_ID}/${IMAGE_NAME} \
       --platform=managed \
       --region=${REGION} \
       --allow-unauthenticated

   # Deploy Frontend
   gcloud run deploy visionflow-frontend \
       --image=gcr.io/${PROJECT_ID}/visionflow-frontend \
       --platform=managed \
       --region=${REGION} \
       --allow-unauthenticated
   ```

## Configuration Files

- `app.yaml`: App Engine configuration
- `cloudbuild.yaml`: Cloud Build configuration
- `Dockerfile`: Container configuration
- `requirements.txt`: Python dependencies
- `main.py`: FastAPI application entry point

## Environment Variables

All environment variables are defined in `.env` file:
- Database credentials
- API keys
- Storage bucket names
- Service configuration

## Monitoring and Logging

- **Cloud Monitoring**: Application metrics and dashboards
- **Cloud Logging**: Structured logging
- **Error Reporting**: Error tracking and alerting
- **Cloud Trace**: Performance monitoring

## Security

- **IAM Roles**: Service account permissions
- **API Keys**: Secure API access
- **VPC Service Controls**: Network security
- **Cloud KMS**: Encryption key management