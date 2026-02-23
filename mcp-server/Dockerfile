# mcp-server/Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True \
    PORT=8080

# Set container working directory
WORKDIR /app

# Copy dependency map
COPY requirements.txt .

# Install dependencies natively to ensure grpcio compiles flawlessly
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP Server logic
COPY . .

# Start the FastAPI server directly via Uvicorn as prescribed by your __main__ block
CMD ["python", "main.py"]
