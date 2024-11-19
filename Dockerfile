FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY random_forest_model.pkl .
COPY templates/ ./templates/
COPY static/ ./static/

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]