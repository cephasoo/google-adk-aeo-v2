# worker-feedback/Dockerfile
FROM python:3.11-slim

# Set environment variables for clean output and cloud-run port compliance
ENV PYTHONUNBUFFERED=True \
    PORT=8080

# Set container working directory
WORKDIR /app

# Copy dependency map
COPY requirements.txt .

# Install dependencies natively on Debian stable (Fixing gRPC C++ Bug)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (including the 'shared/' module synced via deploy script)
COPY . .

# Boot using the Google Functions Framework perfectly sandboxed
CMD exec functions-framework --target=process_feedback_logic --port=$PORT --host=0.0.0.0
