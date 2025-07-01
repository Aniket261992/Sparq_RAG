#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if API_KEY is passed as an argument
if [ -z "$1" ]; then
  echo "Usage: ./start.sh <API_KEY>"
  exit 1
fi

API_KEY=$1

echo "======================================"
echo "🔑 Setting API_KEY in .env"
echo "======================================"

# Write the API key to .env file
echo "API_KEY=${API_KEY}" > .env

echo "======================================"
echo "🐳 Building Docker image: xyz_rag"
echo "======================================"

docker build -t xyz_rag .

echo "======================================"
echo "🚀 Running Docker container: xyz_rag"
echo "======================================"

docker run -p 8000:8000 xyz_rag
