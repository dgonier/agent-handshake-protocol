FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps first (better layer caching).
COPY examples/viewer/requirements.txt /tmp/viewer-requirements.txt
RUN pip install --no-cache-dir -r /tmp/viewer-requirements.txt

# Copy the AHP package + the viewer example.
COPY pyproject.toml README.md /app/
COPY ahp /app/ahp
COPY examples /app/examples

# Editable install of AHP itself so `import ahp` picks up local changes
# when the source is bind-mounted in compose.
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000
ENV VIEWER_DATA_DIR=/data
VOLUME ["/data"]

CMD ["uvicorn", "examples.viewer.app:app", "--host", "0.0.0.0", "--port", "8000"]
