# Contributing to Nepali Restaurant Review AI Agent

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the codebase.

## 🤝 How to Contribute

### Types of Contributions

1. **Bug Reports**: Found a bug? Let us know!
2. **Feature Requests**: Have an idea for improvement?
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve documentation and examples
5. **Testing**: Add tests or improve test coverage

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/nepali-restaurant-ai-agent.git
   cd nepali-restaurant-ai-agent
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv agentenv
   source agentenv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

## 📝 Development Guidelines

### Code Style

We use the following tools for code formatting and linting:

```bash
# Format code with Black
black *.py

# Check linting with flake8
flake8 *.py

# Type checking with mypy
mypy *.py
```

### Code Standards

1. **Follow PEP 8** Python style guidelines
2. **Use type hints** for function parameters and return values
3. **Write docstrings** for all functions and classes
4. **Keep functions small** and focused on single responsibility
5. **Use meaningful variable names**

### Example Code Style

```python
from typing import List, Dict, Optional
import pandas as pd
from langchain_core.documents import Document

def process_reviews(
    reviews_df: pd.DataFrame,
    min_rating: Optional[int] = None
) -> List[Document]:
    """
    Process restaurant reviews and convert to Document objects.
    
    Args:
        reviews_df: DataFrame containing review data
        min_rating: Optional minimum rating filter
        
    Returns:
        List of Document objects with processed reviews
        
    Raises:
        ValueError: If reviews_df is empty or invalid
    """
    if reviews_df.empty:
        raise ValueError("Reviews DataFrame cannot be empty")
    
    documents = []
    
    for idx, row in reviews_df.iterrows():
        if min_rating and row["Rating"] < min_rating:
            continue
            
        document = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={
                "rating": row["Rating"],
                "date": row["Date"],
                "id": str(idx)
            }
        )
        documents.append(document)
    
    return documents
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_vector.py
```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_vector.py
import pytest
import pandas as pd
from vector import process_reviews

def test_process_reviews_valid_input():
    """Test processing valid review data."""
    sample_data = pd.DataFrame({
        'Title': ['Great Food', 'Poor Service'],
        'Review': ['Excellent momos', 'Slow service'],
        'Rating': [5, 2],
        'Date': ['2024-01-01', '2024-01-02']
    })
    
    documents = process_reviews(sample_data)
    
    assert len(documents) == 2
    assert documents[0].metadata['rating'] == 5
    assert 'Great Food Excellent momos' in documents[0].page_content

def test_process_reviews_empty_dataframe():
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="Reviews DataFrame cannot be empty"):
        process_reviews(empty_df)
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the bug
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (OS, Python version, dependencies)
6. **Error messages** and stack traces

### Bug Report Template

```markdown
## Bug Description
A clear description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. macOS 12.0]
- Python: [e.g. 3.11.0]
- Ollama: [e.g. 0.1.17]

## Additional Context
Add any other context about the problem here.
```

## ✨ Feature Requests

For feature requests, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches you've thought about
4. **Additional context**: Examples, mockups, etc.

### Feature Request Template

```markdown
## Problem Statement
Is your feature request related to a problem? Please describe.

## Proposed Solution
Describe the solution you'd like.

## Alternatives Considered
Describe alternatives you've considered.

## Additional Context
Add any other context or screenshots about the feature request here.
```

## 🔄 Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add or update tests** for your changes
3. **Ensure all tests pass**
4. **Run code formatting tools**
5. **Update CHANGELOG.md** if applicable

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly complex areas
- [ ] Documentation updated (if applicable)
- [ ] Tests added/updated and passing
- [ ] No merge conflicts with main branch

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information reviewers should know.
```

## 📁 Project Structure

Understanding the project structure helps with contributions:

```
├── main.py                 # Application entry point
├── vector.py              # Vector store management
├── reviews.csv            # Sample data
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── tests/                 # Test files
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_vector.py
│   └── conftest.py       # Pytest configuration
├── docs/                  # Additional documentation
├── examples/              # Usage examples
└── scripts/               # Utility scripts
```

## 🚀 Development Workflow

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/nepali-restaurant-ai-agent.git
cd nepali-restaurant-ai-agent

# Add upstream remote
git remote add upstream https://github.com/original/nepali-restaurant-ai-agent.git

# Create development environment
python -m venv agentenv
source agentenv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make your changes
# ... edit files ...

# Run tests
pytest

# Format code
black *.py
flake8 *.py

# Commit changes
git add .
git commit -m "Add: descriptive commit message"

# Push to your fork
git push origin feature/your-feature
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Switch to main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Update your fork
git push origin main
```

## 📚 Documentation

### Updating Documentation

- **README.md**: Main project documentation
- **TECHNICAL_DETAILS.md**: Technical implementation details
- **INSTALLATION.md**: Setup and installation guide
- **AGENTS_AND_RAG.md**: AI concepts and architecture
- **API.md**: API documentation (if applicable)

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep information up-to-date

## 🎯 Areas for Contribution

### Beginner-Friendly

- [ ] Add more example queries and responses
- [ ] Improve error messages
- [ ] Add input validation
- [ ] Write additional tests
- [ ] Update documentation

### Intermediate

- [ ] Implement query intent classification
- [ ] Add conversation history
- [ ] Create web interface
- [ ] Add configuration management
- [ ] Implement caching system

### Advanced

- [ ] Multi-agent architecture
- [ ] Real-time learning from feedback
- [ ] Advanced retrieval strategies
- [ ] Performance optimization
- [ ] Distributed deployment

## 📞 Getting Help

If you need help:

1. **Check existing issues** and documentation
2. **Ask questions** in GitHub Discussions
3. **Join community channels** (if available)
4. **Contact maintainers** through GitHub

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes (for significant contributions)

Thank you for contributing to making this project better! 🎉
