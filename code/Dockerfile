# syntax=docker/dockerfile:1

# ARG PYTHON_VERSION=3.9
FROM python:3.9-slim

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the source code into the container.
COPY . /app

RUN pip install -U torch==2.1.1 torchvision==0.16.1
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN mim install mmdet
RUN pip install fastapi
RUN pip install uvicorn
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN pip install python-multipart

# Expose the port that the application listens on.
EXPOSE 80

# Run the application.
CMD uvicorn 'main:app' --host=0.0.0.0 --port=80
