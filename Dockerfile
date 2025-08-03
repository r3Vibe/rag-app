# Use Python 3.12 slim image as base for efficiency
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files to the container
COPY . .

# Expose port 8501 (default Streamlit port)
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "display.py", "--server.port=8501", "--server.address=0.0.0.0"]