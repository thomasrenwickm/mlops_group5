FROM python:3.11-slim

# Setting environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Setting working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying all project files (include your app folder)
COPY . .

# Exposing the FastAPI port
EXPOSE 8000

# Starting FastAPI app 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
