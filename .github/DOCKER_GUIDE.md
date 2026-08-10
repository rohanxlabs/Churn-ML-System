# Docker Guide for Churn ML System

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start the API
docker-compose up --build

# API will be available at http://localhost:8000
# Swagger UI: http://localhost:8000/docs

# Train the model in Docker
docker-compose run --rm churn-api python -m src.pipeline.train_pipeline

# Stop the API
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build the image
docker build -t churn-ml-system .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  churn-ml-system

# Train in Docker
docker run --rm \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  churn-ml-system \
  python -m src.pipeline.train_pipeline
```

## Environment Variables

Pass environment variables to Docker:

```bash
docker run -p 8000:8000 \
  -e CHURN_LOG_LEVEL=DEBUG \
  -e CHURN_ARTIFACT_DIR=/app/artifacts \
  -v $(pwd)/artifacts:/app/artifacts \
  churn-ml-system
```

Or with docker-compose, edit the `environment` section in `docker-compose.yml`.

## Volume Mounts

The docker-compose setup mounts two directories:

- `./artifacts` → `/app/artifacts` - Persists trained models
- `./data` → `/app/data` - Provides access to training data

This allows:
- Training models that persist after container restart
- Using local data files
- Inspecting generated artifacts

## Health Checks

The Docker image includes a health check that runs every 30 seconds:

```bash
# Check container health
docker ps

# View health check logs
docker inspect --format='{{json .State.Health}}' churn-ml-system | jq
```

## Production Deployment

For production, consider:

1. **Multi-stage build** for smaller images
2. **Non-root user** for security
3. **Secret management** via environment variables or secret managers
4. **Resource limits** via docker-compose or orchestrator
5. **Logging** to stdout/stderr for container log aggregation
6. **Monitoring** with health checks and metrics endpoints

Example production docker-compose snippet:

```yaml
services:
  churn-api:
    image: churn-ml-system:latest
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        max_attempts: 3
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs churn-api

# Run interactively
docker-compose run --rm churn-api bash
```

### Port already in use

```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Artifacts not persisting

Ensure volume mounts are correct:

```bash
# Windows PowerShell
docker run -v ${PWD}/artifacts:/app/artifacts ...

# Linux/Mac
docker run -v $(pwd)/artifacts:/app/artifacts ...
```

### Health check failing

```bash
# Check if API is responding
docker exec churn-ml-system curl http://localhost:8000/health

# View full health status
docker inspect churn-ml-system
```
