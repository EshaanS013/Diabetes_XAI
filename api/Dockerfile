FROM python:3.11-slim

WORKDIR /app
COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/base.txt

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY api /app/api
COPY configs /app/configs
COPY artifacts/production /app/artifacts/production

ENV MODEL_ARTIFACT_PATH=artifacts/production/model.joblib
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
