# Use the official Python image as the base image
FROM python:3.10.6-slim

# Set environment variable to ensure Python output is unbuffered
ENV PYTHONUNBUFFERED 1

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the container image
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files to the container image
COPY . .


# Expose the port that the app will run on
EXPOSE 8080

# Start the app
CMD panel serve app.py --address 0.0.0.0 --port 8080 --allow-websocket-origin "*" --log-level debug
