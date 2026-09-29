FROM python:3.10-slim

# Install system dependencies required by Open3D and SciPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY backend/ ./backend/

# Document target port (dynamic at runtime)
EXPOSE 8765

# Start orchestrator
CMD ["python", "-m", "backend.main"]
