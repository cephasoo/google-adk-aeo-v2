# dispatcher-story/Dockerfile
FROM python:3.11-slim

# Set environment variables for clean output and cloud-run port compliance
ENV PYTHONUNBUFFERED=True \
    PORT=8080

# Set container working directory
WORKDIR /app

# Copy dependency map
COPY requirements.txt .

# Install dependencies using standard package manager (bypassing Google's buildpack cache)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Boot using the Google Functions Framework exactly as it runs natively,
# but now fully sandboxed within our own OS image.
CMD exec functions-framework --target=start_story_workflow --port=$PORT --host=0.0.0.0
