# Deployment Checklist

Use this checklist when deploying the Churn ML System to production.

## Pre-Deployment

### Code Quality
- [ ] All tests pass locally (`python -m pytest`)
- [ ] Training pipeline completes successfully
- [ ] API starts without errors
- [ ] No hardcoded secrets or credentials
- [ ] `.env.example` is up to date
- [ ] README reflects current functionality

### Model Artifacts
- [ ] Model is trained on appropriate dataset
- [ ] Evaluation metrics are acceptable (ROC-AUC > 0.80)
- [ ] Artifacts are generated (model.pkl, preprocessor.pkl, feature_names.pkl)
- [ ] Validation report shows no data quality issues
- [ ] Model inference works correctly on test data

### Configuration
- [ ] `config/config.yaml` has correct paths
- [ ] Environment variables are documented
- [ ] Deployment-specific config is separated from code
- [ ] Resource limits are appropriate for expected load

### Security
- [ ] No secrets in repository
- [ ] Environment variables use secure injection
- [ ] Dependencies have no known critical vulnerabilities
- [ ] API error messages don't leak sensitive information
- [ ] CORS is configured appropriately (if needed)

## Deployment Options

### Option 1: Render

1. [ ] Push code to GitHub
2. [ ] Connect repository to Render
3. [ ] Verify `render.yaml` configuration
4. [ ] Set environment variables in Render dashboard
5. [ ] Train model before or during first deploy:
   ```yaml
   buildCommand: pip install -r requirements.txt && python -m src.pipeline.train_pipeline
   ```
6. [ ] Monitor first deployment logs
7. [ ] Test health endpoint: `https://your-app.onrender.com/health`
8. [ ] Test prediction endpoint with sample data
9. [ ] Verify metrics in Render dashboard

### Option 2: Docker (Cloud)

1. [ ] Build image: `docker build -t churn-ml-system:prod .`
2. [ ] Tag for registry: `docker tag churn-ml-system:prod registry.example.com/churn-ml-system:latest`
3. [ ] Push to registry: `docker push registry.example.com/churn-ml-system:latest`
4. [ ] Train model in container or mount pre-trained artifacts
5. [ ] Deploy to container orchestrator (ECS, Kubernetes, etc.)
6. [ ] Configure health checks and auto-scaling
7. [ ] Set up logging and monitoring
8. [ ] Test deployed endpoints

### Option 3: Local Docker

1. [ ] Train model: `python -m src.pipeline.train_pipeline`
2. [ ] Start with docker-compose: `docker-compose up -d`
3. [ ] Verify health: `curl http://localhost:8000/health`
4. [ ] Test prediction: Use sample request from README

## Post-Deployment

### Verification
- [ ] Health endpoint returns 200 with `model_loaded: true`
- [ ] Prediction endpoint returns valid responses
- [ ] Invalid input returns 422 with clear error messages
- [ ] Missing model returns 503 (if applicable)
- [ ] Response times are acceptable (< 1s for prediction)
- [ ] OpenAPI docs accessible at `/docs`

### Monitoring Setup
- [ ] Configure application logging
- [ ] Set up error alerting
- [ ] Monitor response times
- [ ] Track prediction volume
- [ ] Monitor resource usage (CPU, memory)
- [ ] Set up uptime monitoring

### Documentation
- [ ] Document deployed URL
- [ ] Update API documentation with production examples
- [ ] Document deployment-specific configuration
- [ ] Create runbook for common issues
- [ ] Document rollback procedure

## Production Considerations

### Performance
- [ ] Load test API with expected traffic
- [ ] Verify model inference latency
- [ ] Configure connection pooling if needed
- [ ] Consider caching for frequent requests (if applicable)
- [ ] Monitor memory usage under load

### Reliability
- [ ] Set up health checks (already configured in Docker/Render)
- [ ] Configure auto-restart on failure
- [ ] Implement graceful shutdown
- [ ] Test recovery from model loading failures
- [ ] Plan for zero-downtime deployments

### Scalability
- [ ] Determine scaling strategy (horizontal vs vertical)
- [ ] Configure auto-scaling thresholds
- [ ] Test behavior under high load
- [ ] Plan for artifact storage as model versions increase
- [ ] Consider read replicas if needed

### Security
- [ ] Enable HTTPS (handled by Render/cloud provider)
- [ ] Implement rate limiting if public-facing
- [ ] Configure CORS policies
- [ ] Review and harden error messages
- [ ] Set up security scanning in CI/CD

### Data
- [ ] Plan model retraining schedule
- [ ] Set up artifact versioning strategy
- [ ] Configure backup for critical artifacts
- [ ] Plan for data drift monitoring (future)
- [ ] Document prediction logging strategy (if implemented)

## Rollback Plan

If deployment fails:

1. [ ] Check deployment logs for errors
2. [ ] Verify environment variables are set correctly
3. [ ] Test model artifacts are present
4. [ ] Roll back to previous version if needed
5. [ ] Investigate root cause before redeploying

## Common Issues

### Model not loading
- Verify artifacts exist in expected location
- Check file permissions
- Ensure artifact directory is correctly mounted (Docker)
- Verify preprocessor and model versions are compatible

### High latency
- Check model complexity
- Verify resource allocation
- Monitor database connections (if applicable)
- Consider preprocessing optimization
- Check for memory swapping

### 503 errors
- Model artifacts missing or corrupted
- Insufficient memory to load model
- Permissions issue accessing artifact files
- Check logs for specific error messages

### Docker deployment issues
- Verify volume mounts are correct
- Check container logs: `docker logs churn-ml-system`
- Ensure port mappings are correct
- Verify health check configuration

## Maintenance Schedule

- [ ] Weekly: Review logs for errors
- [ ] Monthly: Update dependencies (security patches)
- [ ] Quarterly: Retrain model with new data
- [ ] Quarterly: Review and update metrics thresholds
- [ ] Yearly: Major dependency upgrades
