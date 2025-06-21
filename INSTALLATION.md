# 🚀 Installation and Setup Guide

## System Requirements

### Minimum Requirements
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.10 or higher
- **Internet**: Required for initial model downloads

### Recommended Requirements
- **RAM**: 16GB+ for optimal performance
- **CPU**: Multi-core processor (Apple Silicon/Intel/AMD)
- **Storage**: SSD for faster vector operations
- **GPU**: Optional (Ollama can utilize GPU acceleration)

## Step-by-Step Installation

### 1. Install Python and Pip

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Verify installation
python3 --version
pip3 --version
```

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-pip python3.11-venv

# Verify installation
python3.11 --version
pip3 --version
```

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and check "Add Python to PATH"
3. Verify in Command Prompt:
```cmd
python --version
pip --version
```

### 2. Install Ollama

#### macOS
```bash
# Using Homebrew
brew install ollama

# Or download from website
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux
```bash
# Install script
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama
```

#### Windows
1. Download Ollama installer from [ollama.ai](https://ollama.ai/download)
2. Run the installer
3. Ollama will start automatically

### 3. Download Required Models

```bash
# Start Ollama service (if not running)
ollama serve

# In a new terminal, download models
ollama pull llama3.2
ollama pull mxbai-embed-large

# Verify models are downloaded
ollama list
```

**Expected Output:**
```
NAME                    ID              SIZE      MODIFIED
llama3.2:latest         a4d6b80b56d2    2.0GB     2 hours ago
mxbai-embed-large:latest 468836c2f4d2   669MB     2 hours ago
```

### 4. Clone the Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd "Run Ai Agents Locally Using OLLAMA"

# Or download ZIP and extract
```

### 5. Create Virtual Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python3 -m venv agentenv

# Activate virtual environment
# macOS/Linux:
source agentenv/bin/activate

# Windows:
agentenv\Scripts\activate

# Verify activation (should show (agentenv) in prompt)
which python
```

#### Using conda (Alternative)
```bash
# Create conda environment
conda create -n agentenv python=3.11 -y

# Activate environment
conda activate agentenv
```

### 6. Install Python Dependencies

```bash
# Ensure virtual environment is activated
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installations
pip list
```

### 7. Verify Installation

#### Test Ollama Connection
```bash
# Test LLM model
ollama run llama3.2 "Hello, how are you?"

# Test embedding model
ollama run mxbai-embed-large "Test embedding"
```

#### Test Python Dependencies
```bash
python -c "
import pandas as pd
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
print('✅ All dependencies imported successfully')
"
```

#### Test Vector Store Creation
```bash
python -c "
from vector import vector_store, retriever
print(f'✅ Vector store created with {vector_store._collection.count()} documents')
"
```

### 8. Run the Application

```bash
# Ensure virtual environment is activated
python main.py
```

**Expected Output:**
```
Vector store already contains 130 documents. Skipping document addition.

==================================================

Welcome to the Nepali Authentic Cuisine Restaurant Q&A!
Enter your question (or type 'q' to quit):
```

## Configuration Options

### Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=mxbai-embed-large
LLM_MODEL=llama3.2
VECTOR_DB_PATH=./chroma_db
MAX_RETRIEVED_DOCS=5
```

### Model Configuration

#### Switching LLM Models
```python
# In main.py, modify:
model = OllamaLLM(model="llama3.1")  # or "codellama", "mistral"
```

#### Switching Embedding Models
```python
# In vector.py, modify:
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Alternative model
```

### Performance Tuning

#### Memory Optimization
```python
# Reduce batch size for lower memory usage
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
    # Add these for memory optimization
    collection_metadata={"hnsw:space": "cosine", "hnsw:M": 16}
)
```

#### Retrieval Tuning
```python
# Adjust number of retrieved documents
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10,  # Retrieve more documents
        "score_threshold": 0.7  # Minimum similarity threshold
    }
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Not Found
**Error**: `ollama: command not found`

**Solution**:
```bash
# macOS - Add to PATH
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Linux - Restart terminal or
source ~/.bashrc

# Windows - Restart Command Prompt after installation
```

#### 2. Model Download Fails
**Error**: `Error pulling model`

**Solutions**:
```bash
# Check internet connection
ping ollama.ai

# Clear Ollama cache
ollama rm llama3.2
ollama rm mxbai-embed-large

# Re-download models
ollama pull llama3.2
ollama pull mxbai-embed-large

# Check disk space
df -h  # Linux/macOS
dir    # Windows
```

#### 3. ChromaDB Permission Error
**Error**: `PermissionError: [Errno 13] Permission denied`

**Solutions**:
```bash
# Fix directory permissions
chmod -R 755 ./chroma_db

# Or remove and recreate
rm -rf ./chroma_db
python vector.py
```

#### 4. Memory Issues
**Error**: `Out of memory` or slow performance

**Solutions**:
```bash
# Reduce model context
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# Use smaller models
ollama pull llama3.2:1b  # Smaller variant
```

#### 5. Vector Store Empty
**Error**: Retriever returns no results

**Solution**:
```bash
# Check document count
python -c "from vector import vector_store; print(vector_store._collection.count())"

# If 0, force document addition
rm -rf ./chroma_db
python vector.py
```

### Performance Monitoring

#### Check Resource Usage
```bash
# Monitor CPU and memory
top -p $(pgrep -f ollama)

# Monitor disk usage
du -sh ./chroma_db/

# Check GPU usage (if available)
nvidia-smi  # NVIDIA GPUs
```

#### Benchmark Performance
```python
# benchmark.py
import time
from vector import retriever

queries = [
    "What do people say about momos?",
    "How is the service?",
    "What's the atmosphere like?",
    "Are the prices reasonable?",
    "Which dishes are recommended?"
]

for query in queries:
    start_time = time.time()
    results = retriever.invoke(query)
    end_time = time.time()
    
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print(f"Time: {end_time - start_time:.3f}s")
    print("-" * 50)
```

### Development Setup

#### Installing Development Dependencies
```bash
# Additional dev dependencies
pip install pytest black flake8 mypy jupyter

# Create dev requirements
pip freeze > requirements-dev.txt
```

#### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
EOF

# Install hooks
pre-commit install
```

## Docker Setup (Optional)

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Ollama
EXPOSE 11434

# Start script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    
  rag-agent:
    build: .
    depends_on:
      - ollama
    volumes:
      - ./chroma_db:/app/chroma_db
    environment:
      - OLLAMA_HOST=http://ollama:11434

volumes:
  ollama_data:
```

This comprehensive installation guide should help users set up the project successfully across different platforms and troubleshoot common issues.
