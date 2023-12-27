# Use the official lightweight Python image.
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        tesseract-ocr \
        libgl1-mesa-glx \
        PyMuPDF \
        frontend \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies.
RUN pip install -r requirements.txt

# Install tesseract
RUN apt-get update -qqy && apt-get install -qqy \
    tesseract-ocr \
    libtesseract-dev

# Install tesseract
RUN apt-get update -qqy && apt-get install -qqy \
    tesseract-ocr \
    libtesseract-dev

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
