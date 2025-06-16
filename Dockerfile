FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 git curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download model (could be done manually too)
RUN mkdir -p models && curl -L -o models/TinyLlama-1.1B-Chat.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]