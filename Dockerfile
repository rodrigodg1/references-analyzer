# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV HOST=0.0.0.0
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]