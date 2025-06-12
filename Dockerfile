FROM ubuntu:22.04

# Install system deps
RUN apt-get update && apt-get install -y \
    curl git python3 python3-pip libssl-dev \
    && apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy your app
WORKDIR /app
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint script to run ollama + pull model + start FastAPI
CMD bash -c "ollama serve & sleep 5 && ollama pull mistral && uvicorn app.main:app --host 0.0.0.0 --port 8000"
