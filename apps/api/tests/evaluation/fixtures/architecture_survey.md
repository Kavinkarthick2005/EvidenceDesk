# Survey of Information Retrieval Technologies and Vector Systems

## 1. Cloud-Hosted Vector Databases
Cloud-native vector database solutions such as Pinecone, Milvus, and Qdrant offer managed indexing and multi-node sharding. These platforms store embeddings on remote clusters rather than local PostgreSQL instances.

## 2. High-Dimensional Embedding Models
Commercial API providers often use high-dimensional representations. For example, OpenAI text-embedding-3-small and text-embedding-ada-002 produce 1536-dimensional vectors, requiring substantially greater storage than 384-dimensional models.

## 3. Approximate Nearest Neighbor Indexing Algorithms
Approximate nearest neighbor algorithms like HNSW construct hierarchical graph layers using hyperparameters M=16 and ef_construction=64. While HNSW accelerates search on millions of vectors, it introduces recall trade-offs compared to exact flat scanning.

## 4. Lexical Search with BM25
Lexical search engines like Elasticsearch and Tantivy calculate document relevance using the BM25 probabilistic ranking formula. BM25 relies on exact term frequencies and inverse document frequencies rather than semantic embeddings.

## 5. Reciprocal Rank Fusion
Hybrid retrieval pipelines frequently merge dense vector rankings with BM25 keyword rankings using Reciprocal Rank Fusion with smoothing constant k=60. This balances lexical precision with dense semantic recall.

## 6. Distributed Session Caching
Distributed systems employ Redis clusters to cache query results and session states with a default time-to-live of 3600 seconds. In-memory caching mitigates redundant vector distance calculations for frequent queries.
