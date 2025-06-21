# 🔬 Technical Deep Dive: RAG Implementation

## Mathematical Foundations

### Vector Embeddings Theory

#### What are Embeddings?
Embeddings transform discrete tokens (words, sentences) into continuous vector representations in high-dimensional space:

```
f: Text → ℝᵈ
```

Where `d` is the embedding dimension (1024 in our case).

#### Word2Vec to Modern Embeddings
Evolution of embedding techniques:

1. **Word2Vec (2013)**:
   ```
   P(w_context | w_target) = exp(v'_context · v_target) / Σ exp(v'_i · v_target)
   ```

2. **BERT-style (2018)**:
   ```
   h_i = Transformer(x_1, ..., x_n)_i
   ```

3. **Modern (mxbai-embed-large)**:
   - Trained on massive text corpora
   - Optimized for retrieval tasks
   - Dimension: 1024
   - Context length: 512 tokens

### Similarity Metrics

#### Cosine Similarity (Used in ChromaDB)
```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
```

**Properties**:
- Range: [-1, 1]
- Geometric interpretation: Angle between vectors
- Scale-invariant: Only direction matters

#### Alternative Metrics
```
# Euclidean Distance
euclidean(A, B) = ||A - B||₂

# Manhattan Distance  
manhattan(A, B) = ||A - B||₁

# Dot Product
dot_product(A, B) = A · B
```

### HNSW Algorithm (ChromaDB Index)

#### Hierarchical Navigable Small World
ChromaDB uses HNSW for efficient approximate nearest neighbor search:

```
Algorithm: HNSW Search
Input: query q, graph G, entry point ep, layers L
1. For layer = L down to 1:
   2. Greedy search from current best candidates
   3. Update candidates using layer-specific edges
4. At layer 0: Return k nearest neighbors
```

**Complexity**:
- Construction: O(N log N)
- Search: O(log N)
- Memory: O(N × M) where M is max connections

## RAG Architecture Deep Dive

### 1. Document Processing Pipeline

#### Text Preprocessing
```python
def preprocess_text(text):
    # Our implementation
    return title + " " + review  # Simple concatenation
    
    # Advanced preprocessing could include:
    # - Normalization: text.lower().strip()
    # - Tokenization: word_tokenize(text)
    # - Stop word removal: [w for w in tokens if w not in stopwords]
    # - Stemming/Lemmatization
```

#### Chunking Strategy
For larger documents, chunking is crucial:

```python
# Fixed-size chunking
def fixed_chunk(text, size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

# Semantic chunking (advanced)
def semantic_chunk(text, model):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        if len(' '.join(current_chunk + [sentence])) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)
    
    return chunks
```

### 2. Vector Storage Architecture

#### ChromaDB Internal Structure
```sql
-- Simplified ChromaDB schema
CREATE TABLE collections (
    id TEXT PRIMARY KEY,
    name TEXT,
    metadata JSON
);

CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    collection_id TEXT,
    embedding BLOB,  -- Serialized vector
    document TEXT,
    metadata JSON,
    FOREIGN KEY (collection_id) REFERENCES collections(id)
);

CREATE INDEX idx_embeddings_collection ON embeddings(collection_id);
```

#### Vector Quantization
For memory efficiency, ChromaDB may use quantization:

```python
# Product Quantization (conceptual)
def quantize_vector(vector, codebooks):
    subvectors = split_vector(vector, len(codebooks))
    quantized = []
    for i, subvec in enumerate(subvectors):
        closest_centroid = find_nearest(subvec, codebooks[i])
        quantized.append(closest_centroid)
    return quantized
```

### 3. Retrieval Mathematics

#### Similarity Search Algorithm
```python
def similarity_search(query_vector, document_vectors, k=5):
    similarities = []
    
    for doc_id, doc_vector in document_vectors.items():
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc_id, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]
```

#### Approximate vs Exact Search

**Exact Search**: O(N) complexity
```python
def exact_search(query, vectors):
    scores = [cosine_sim(query, v) for v in vectors]
    return heapq.nlargest(k, enumerate(scores), key=lambda x: x[1])
```

**Approximate Search (HNSW)**: O(log N) complexity
- Trade-off: Speed vs Accuracy
- Recall@k typically > 95% for well-tuned parameters

### 4. LLM Integration

#### Prompt Engineering
Our template structure:
```python
template = """
System Context: {role_definition}
Knowledge Base: {retrieved_documents}
User Query: {user_question}
Instructions: {generation_guidelines}
"""
```

#### Context Window Management
Llama 3.2 context window: ~128k tokens

```python
def manage_context(retrieved_docs, max_tokens=4000):
    current_tokens = count_tokens(base_prompt)
    included_docs = []
    
    for doc in retrieved_docs:
        doc_tokens = count_tokens(doc.page_content)
        if current_tokens + doc_tokens < max_tokens:
            included_docs.append(doc)
            current_tokens += doc_tokens
        else:
            break
    
    return included_docs
```

## Performance Optimization

### Caching Strategies

#### Embedding Caching
```python
import hashlib
import pickle

class EmbeddingCache:
    def __init__(self, cache_dir="./embedding_cache"):
        self.cache_dir = cache_dir
    
    def get_embedding(self, text):
        # Create hash of text for cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_path = f"{self.cache_dir}/{text_hash}.pkl"
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Generate new embedding
        embedding = self.model.embed_query(text)
        
        # Cache for future use
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        return embedding
```

#### Query Result Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieval(query_hash):
    return retriever.invoke(query_hash)
```

### Batch Processing

#### Batch Embedding Generation
```python
def batch_embed_documents(documents, batch_size=32):
    embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        
        # Single API call for batch
        batch_embeddings = embedding_model.embed_documents(batch_texts)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

## Evaluation Metrics

### Retrieval Quality

#### Precision@k
```python
def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    return len(retrieved_k & relevant_set) / k
```

#### Recall@k
```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    return len(retrieved_k & relevant_set) / len(relevant_set)
```

#### Mean Reciprocal Rank (MRR)
```python
def mean_reciprocal_rank(queries_results):
    mrr_sum = 0
    
    for retrieved, relevant in queries_results:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                mrr_sum += 1 / (i + 1)
                break
    
    return mrr_sum / len(queries_results)
```

### Generation Quality

#### ROUGE Scores
```python
from rouge_score import rouge_scorer

def evaluate_rouge(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(generated, reference)
    return scores
```

#### Semantic Similarity
```python
def semantic_similarity(generated, reference, model):
    gen_embedding = model.encode(generated)
    ref_embedding = model.encode(reference)
    
    return cosine_similarity(gen_embedding, ref_embedding)
```

## Scaling Considerations

### Horizontal Scaling

#### Distributed Vector Search
```python
# Conceptual distributed setup
class DistributedVectorStore:
    def __init__(self, shards):
        self.shards = shards  # List of VectorStore instances
    
    def search(self, query, k):
        # Search all shards in parallel
        shard_results = []
        for shard in self.shards:
            results = shard.search(query, k)
            shard_results.extend(results)
        
        # Merge and re-rank
        return sorted(shard_results, key=lambda x: x.score, reverse=True)[:k]
```

#### Load Balancing
```python
import random

class LoadBalancer:
    def __init__(self, embedding_services):
        self.services = embedding_services
    
    def get_embedding(self, text):
        # Round-robin or random selection
        service = random.choice(self.services)
        return service.embed_query(text)
```

### Memory Optimization

#### Memory-Mapped Files
```python
import mmap

def load_embeddings_mmap(file_path):
    with open(file_path, 'rb') as f:
        # Memory-map the file
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Load embeddings without loading entire file into memory
        return mmapped_file
```

#### Streaming Processing
```python
def stream_process_documents(file_path, batch_size=1000):
    with open(file_path, 'r') as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            
            if len(batch) >= batch_size:
                yield process_batch(batch)
                batch = []
        
        if batch:
            yield process_batch(batch)
```

## Security Considerations

### Input Sanitization
```python
import re

def sanitize_input(user_input):
    # Remove potential injection attempts
    cleaned = re.sub(r'[<>{}\\]', '', user_input)
    
    # Limit length
    cleaned = cleaned[:1000]
    
    # Check for malicious patterns
    malicious_patterns = ['exec', 'eval', 'import', '__']
    for pattern in malicious_patterns:
        if pattern in cleaned.lower():
            raise ValueError("Potentially malicious input detected")
    
    return cleaned
```

### Rate Limiting
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=100, time_window=3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests
        user_requests[:] = [req_time for req_time in user_requests 
                          if now - req_time < self.time_window]
        
        if len(user_requests) >= self.max_requests:
            return False
        
        user_requests.append(now)
        return True
```

This technical documentation provides the mathematical foundations and implementation details behind the RAG system, helping developers understand the underlying mechanisms and optimize performance.
