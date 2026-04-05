# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install dependencies (add more if your envs need them)
RUN pip install --no-cache-dir fastapi uvicorn pydantic dataclasses

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
