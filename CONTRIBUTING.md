# Contributing to Tax Chatbot

We welcome contributions to the Tax Chatbot project! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended: RTX 2080 Ti or better)
- 16GB+ system RAM
- Git

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tax-chatbot.git
   cd tax-chatbot
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   pip install pytest black isort flake8  # Development tools
   ```

5. **Test your setup**:
   ```bash
   python test_environment.py
   ```

## 🔄 Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/your-feature-name`: Feature development
- `bugfix/issue-description`: Bug fixes

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run formatting
   black src/
   isort src/
   
   # Run linting
   flake8 src/
   
   # Run tests
   python examples/system_tests.py
   pytest  # If you add unit tests
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide
- **PEP 8**: Follow Python style guidelines
- **Black**: Use Black for code formatting
- **isort**: Use isort for import sorting
- **Type Hints**: Add type hints for function parameters and returns
- **Docstrings**: Use Google-style docstrings

### Code Quality
- **Maximum line length**: 88 characters (Black default)
- **Complexity**: Keep functions focused and simple
- **Comments**: Write clear, necessary comments
- **Variable names**: Use descriptive names

### Example Code Style
```python
from typing import List, Optional, Dict, Any

def process_tax_query(
    query: str, 
    user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a tax-related query with optional user context.
    
    Args:
        query: The user's tax question
        user_context: Optional user information for personalization
        
    Returns:
        Dictionary containing response and metadata
        
    Raises:
        ValueError: If query is empty or invalid
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Implementation here...
    return {"response": "...", "query_type": "general_tax"}
```

## 🧪 Testing Guidelines

### Test Categories
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions  
3. **System Tests**: End-to-end functionality tests
4. **Performance Tests**: Memory usage and response time tests

### Writing Tests
- Place tests in `tests/` directory
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies when appropriate

### Running Tests
```bash
# System tests (provided)
python examples/system_tests.py

# Component tests
python test_router.py
python test_hierarchical_retrieval.py

# Unit tests (if you add pytest tests)
pytest tests/
```

## 🏗️ Project Structure

```
tax-chatbot/
├── src/
│   ├── core/           # Core AI/ML components
│   ├── interfaces/     # CLI and web interfaces
│   ├── utils/          # Utility modules
│   ├── evaluation/     # Evaluation framework
│   └── monitoring/     # Observability components
├── examples/           # Usage examples
├── tests/             # Test files
├── docs/              # Additional documentation
└── resources/         # Data and configuration files
```

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, include:
- **Environment**: OS, Python version, GPU info
- **Steps to reproduce**: Clear reproduction steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error logs
- **System specs**: Memory, GPU model, etc.

### Feature Requests
For new features, provide:
- **Use case**: Why is this feature needed?
- **Description**: What should the feature do?
- **Examples**: How would users interact with it?
- **Implementation ideas**: Optional technical suggestions

## 🚀 Types of Contributions

### Documentation
- Fix typos and improve clarity
- Add usage examples
- Update API documentation
- Create tutorials and guides

### Code Improvements
- Fix bugs and issues
- Optimize performance
- Add new features
- Improve error handling

### Testing
- Add unit tests
- Improve test coverage
- Add integration tests
- Performance benchmarks

### Examples and Tutorials
- Create new usage examples
- Add Jupyter notebooks
- Write how-to guides
- Record demo videos

## 🤝 Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages are descriptive
- [ ] No merge conflicts with main

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Existing tests pass
- [ ] Added new tests if needed
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process
1. **Automated checks**: CI pipeline must pass
2. **Code review**: At least one maintainer review
3. **Testing**: Verify functionality works as expected
4. **Merge**: Squash merge to main branch

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📜 Code of Conduct

### Our Standards
- **Respectful**: Be respectful of differing viewpoints
- **Constructive**: Provide constructive feedback
- **Inclusive**: Welcome newcomers and diverse perspectives
- **Professional**: Maintain professional conduct

### Unacceptable Behavior
- Harassment or discriminatory language
- Personal attacks or insults
- Public or private harassment
- Inappropriate sexual content

## 🆘 Getting Help

- **Questions**: Open a GitHub issue with the `question` label
- **Documentation**: Check existing docs and examples
- **Discussions**: Use GitHub Discussions for broader topics
- **Bug Reports**: Use GitHub issues with bug report template

## 🎯 Development Priorities

### Current Focus Areas
1. **Performance Optimization**: Memory usage and response times
2. **Additional Jurisdictions**: Support for more tax jurisdictions
3. **Evaluation Framework**: Better accuracy measurement
4. **Documentation**: Comprehensive guides and examples

### Future Roadmap
- Multi-modal document support (images, tables)
- Real-time learning capabilities
- Advanced evaluation metrics
- Cloud deployment guides

## 📞 Contact

For questions about contributing, please:
1. Check existing issues and documentation
2. Open a GitHub issue with the `question` label
3. Join GitHub Discussions for broader topics

Thank you for contributing to Tax Chatbot! 🚀