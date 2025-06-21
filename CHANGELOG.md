# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-21

### Added
- Initial release of Nepali Restaurant Review AI Agent
- RAG (Retrieval-Augmented Generation) pipeline implementation
- Vector store using ChromaDB for efficient semantic search
- Integration with Ollama for local LLM processing
- Support for Llama 3.2 language model
- mxbai-embed-large embedding model integration
- Interactive command-line interface
- Restaurant review dataset with 130 entries
- Comprehensive documentation and setup guides

### Features
- **Semantic Search**: Vector-based similarity search for relevant review retrieval
- **Local Processing**: Complete offline operation using Ollama
- **Contextual Responses**: LLM-powered answers grounded in review data
- **Persistent Storage**: ChromaDB for vector embeddings persistence
- **Metadata Support**: Review ratings and dates for enhanced context

### Documentation
- Complete README with project overview and quick start
- Technical deep dive with mathematical foundations
- Installation guide for multiple platforms
- AI Agents and RAG architecture explanation
- Contributing guidelines for open source collaboration

### Technical Specifications
- Python 3.10+ support
- LangChain framework integration
- Pandas for data processing
- Vector dimensions: 1024 (mxbai-embed-large)
- Similarity metric: Cosine similarity
- Search algorithm: HNSW (Hierarchical Navigable Small World)

### Dependencies
- langchain-ollama: ^0.1.0
- langchain-chroma: ^0.1.0
- langchain-core: ^0.3.0
- pandas: ^2.0.0
- chromadb: ^0.5.0

## [Unreleased]

### Planned Features
- [ ] Web interface using Streamlit/Gradio
- [ ] Multi-turn conversation support
- [ ] Intent classification system
- [ ] User feedback collection and learning
- [ ] Performance metrics and monitoring
- [ ] Docker deployment support
- [ ] API endpoint creation
- [ ] Multi-language support
- [ ] Advanced query preprocessing
- [ ] Sentiment analysis integration

### Planned Improvements
- [ ] Response caching for faster queries
- [ ] Batch processing for large datasets
- [ ] Memory optimization for resource-constrained environments
- [ ] Enhanced error handling and logging
- [ ] Configuration file support
- [ ] Unit test coverage expansion
- [ ] Integration test suite
- [ ] Performance benchmarking tools

### Planned Documentation
- [ ] API reference documentation
- [ ] Video tutorials and demos
- [ ] Architecture decision records (ADRs)
- [ ] Deployment guides for cloud platforms
- [ ] Performance tuning guide
- [ ] Troubleshooting FAQ expansion

---

### Version History Format

#### [Version] - Date

**Added**
- New features

**Changed** 
- Changes in existing functionality

**Deprecated**
- Soon-to-be removed features

**Removed**
- Removed features

**Fixed**
- Bug fixes

**Security**
- Security vulnerability fixes
