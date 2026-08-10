# CI/CD Guide for Churn ML System

## GitHub Actions Workflow

The project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/ci.yml`.

## Workflow Overview

### Triggers

The CI pipeline runs on:
- Push to `main`, `master`, or `develop` branches
- Pull requests targeting those branches

### Jobs

#### 1. **Test Job**

Runs the test suite and verifies the training pipeline:

```yaml
- Install Python 3.11
- Install dependencies from requirements.txt
- Run pytest with verbose output
- Execute training pipeline
- Verify all artifacts are generated
```

**What it checks:**
- All 16 tests pass
- Training completes without errors
- Required artifacts (model.pkl, preprocessor.pkl, etc.) are created
- No import errors or dependency issues

#### 2. **Lint Job**

Checks code quality with ruff:

```yaml
- Install ruff
- Run basic linting (E, F, W categories)
- Ignore line length (E501)
```

**Note:** This job uses `continue-on-error: true`, so linting warnings won't fail the build.

#### 3. **Docker Job**

Verifies Docker build:

```yaml
- Set up Docker Buildx
- Build Docker image
- Test that app module imports successfully
```

## Local CI Simulation

Run the same checks locally before pushing:

### Test Suite
```bash
python -m pytest tests/ -v --tb=short
```

### Training Verification
```bash
python -m src.pipeline.train_pipeline
```

### Artifact Check
```bash
ls artifacts/model.pkl
ls artifacts/preprocessor.pkl
ls artifacts/feature_names.pkl
ls artifacts/metrics.json
ls artifacts/validation_report.json
```

### Docker Build (if Docker installed)
```bash
docker build -t churn-ml-system:test .
docker run --rm churn-ml-system:test python -c "import app; print('OK')"
```

### Linting (optional)
```bash
pip install ruff
ruff check . --select E,F,W --ignore E501
```

## Workflow Status

View workflow runs:
1. Go to your repository on GitHub
2. Click the "Actions" tab
3. View workflow runs and logs

## Common CI Failures and Fixes

### Test failures

```bash
# Run tests locally to debug
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_api.py::test_health_endpoint_reports_model_ready -v
```

### Import errors

```bash
# Verify dependencies are correct
pip install -r requirements.txt
python -c "import app"
```

### Training pipeline failures

```bash
# Check if data file exists
ls data/raw/churn.csv

# Check config
cat config/config.yaml

# Run with debug logging
CHURN_LOG_LEVEL=DEBUG python -m src.pipeline.train_pipeline
```

### Docker build failures

```bash
# Build locally to see detailed error
docker build -t churn-ml-system:debug .

# Check .dockerignore isn't excluding needed files
cat .dockerignore
```

## Extending the CI Pipeline

### Add Coverage Reporting

```yaml
- name: Run tests with coverage
  run: |
    pip install pytest-cov
    python -m pytest tests/ --cov=src --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
```

### Add Deployment Step

```yaml
deploy:
  needs: [test, lint, docker]
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  steps:
    - name: Deploy to Render
      # Add deployment steps
```

### Add Security Scanning

```yaml
- name: Run Bandit security scanner
  run: |
    pip install bandit
    bandit -r src/ -f json
```

### Add Dependency Checking

```yaml
- name: Check for vulnerabilities
  run: |
    pip install safety
    safety check
```

## Best Practices

1. **Keep pipelines fast** - Current pipeline takes ~2-3 minutes
2. **Cache dependencies** - Uses `actions/setup-python` with cache
3. **Fail fast** - Tests run before expensive Docker builds
4. **Clear feedback** - Verbose test output with `--tb=short`
5. **Reproducible** - Pinned dependency versions ensure consistency

## Troubleshooting CI Issues

### "Tests pass locally but fail in CI"

- Check Python version matches (3.11)
- Verify environment variables
- Check for platform-specific code
- Review CI logs for environment differences

### "CI is slow"

- Enable dependency caching (already configured)
- Run fewer jobs in parallel
- Use matrix strategy for multiple Python versions only if needed

### "Docker build times out"

- Optimize Dockerfile layers
- Use smaller base image
- Enable BuildKit for faster builds (already using `setup-buildx-action`)

## GitHub Secrets

For deployment workflows, add secrets:

1. Go to repository Settings → Secrets and variables → Actions
2. Add secrets like:
   - `RENDER_API_KEY` (for Render deployments)
   - `DOCKER_USERNAME` / `DOCKER_PASSWORD` (for Docker Hub)

Access in workflow:
```yaml
env:
  API_KEY: ${{ secrets.RENDER_API_KEY }}
```
