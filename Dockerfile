# syntax=docker/dockerfile:1
FROM python:3.9.12-slim-bullseye
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .