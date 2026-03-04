FROM python:3.11-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Shared directory logic for Cloud Run (handled via build context)
# COPY shared ./shared

CMD exec functions-framework --target=process_schema_logic --port=$PORT
