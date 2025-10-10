# Contributing to VideoAnnotator

Thank you for your interest in contributing to VideoAnnotator! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing Standards](#testing-standards)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- FFmpeg (for video processing)
- Optional: Docker for containerized development

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/VideoAnnotator.git
   cd VideoAnnotator
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/VideoAnnotator.git
   ```

## Development Setup

### Quick Setup

Use the Makefile for quick development setup:

```bash
make dev-setup
```

This will:

- Install the package in development mode
- Set up pre-commit hooks
- Install all development dependencies

### Manual Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -e .[dev]
   ```

3. Set up pre-commit hooks:

   ```bash
   pre-commit install
   ```

4. Verify installation:
   ```bash
   make test-unit
   ```

### Docker Development

For Docker-based development:

```bash
docker-compose --profile dev up --build
```

## Testing Standards

Please follow our comprehensive testing standards outlined in [TESTING_STANDARDS.md](docs/TESTING_STANDARDS.md).

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Requirements

- All new features must include unit tests
- Integration tests for pipeline components
- Performance tests for critical paths
- Tests must pass in all supported Python versions (3.8-3.11)
- Minimum code coverage: 80%

## Code Style

### Formatting

We use automated code formatting:

```bash
# Format code
make format

# Check formatting
black --check src tests examples
isort --check-only src tests examples
```

### Linting

```bash
# Run linting
make lint

# Type checking
make type-check

# All quality checks
make quality-check
```

### Style Guidelines

- Follow PEP 8 style guide
- Use type hints for all functions and methods
- Write docstrings in Google style
- Maximum line length: 88 characters
- Use meaningful variable and function names

### Example Code Structure

```python
"""Module docstring describing the module's purpose."""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExampleClass:
    """Example class demonstrating style guidelines.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """

    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        """Initialize the example class."""
        self.param1 = param1
        self.param2 = param2

    def process_data(self, data: List[Dict[str, str]]) -> List[str]:
        """Process input data and return processed results.

        Args:
            data: List of dictionaries containing input data

        Returns:
            List of processed string results

        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")

        results = []
        for item in data:
            processed = self._process_item(item)
            results.append(processed)

        return results

    def _process_item(self, item: Dict[str, str]) -> str:
        """Private method to process individual items."""
        # Implementation details
        pass
```

## Pull Request Process

### Before Creating a PR

1. Ensure your fork is up to date:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes following the guidelines above

4. Run the full test suite:
   ```bash
   make dev-cycle
   ```

### PR Requirements

- **Title**: Use a clear, descriptive title
- **Description**: Provide a detailed description of changes
- **Tests**: Include appropriate tests for new features
- **Documentation**: Update documentation if needed
- **Breaking Changes**: Clearly mark any breaking changes

### PR Template

```markdown
## Description

Brief description of the changes made.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Code coverage maintained/improved

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] Changes tested on different platforms (if applicable)
```

## Issue Reporting

### Bug Reports

Use the bug report template with:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or screenshots

### Feature Requests

Use the feature request template with:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Alternatives considered

### Performance Issues

Include:

- Detailed performance metrics
- System specifications
- Profiling results (if available)
- Comparison with expected performance

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation updates

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:

```
feat(face_analysis): add emotion detection with DeepFace
fix(audio_pipeline): resolve memory leak in batch processing
docs(installation): update Docker setup instructions
```

### Code Review Process

1. **Automated Checks**: All CI checks must pass
2. **Manual Review**: Code review by maintainers
3. **Testing**: Reviewer tests the changes
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

## Documentation

### API Documentation

- Use Google-style docstrings
- Include examples in docstrings
- Document all public APIs
- Update documentation for changes

### User Documentation

- Update README.md for significant changes
- Add examples to the examples/ directory
- Update configuration documentation
- Create tutorials for new features

### Building Documentation

```bash
make docs
make serve-docs
```

## Release Process

### Version Numbering

We follow semantic versioning (SemVer):

- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Publish to PyPI (automated)
7. Create GitHub release

## Getting Help

- **General Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Issues
- **Security Issues**: Email maintainers directly

## Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes for significant contributions

## Development Tools

### Recommended IDE Setup

**VS Code Extensions:**

- Python
- Black Formatter
- isort
- Pylance
- GitLens
- Docker

**PyCharm:**

- Configure Black as external tool
- Enable type checking
- Set up run configurations

### Debugging

```bash
# Run with debug logging
python main.py --log-level DEBUG

# Debug specific pipeline
python -m pdb examples/test_individual_pipelines.py
```

### Performance Profiling

```bash
# Profile performance
python -m cProfile -o profile.stats main.py

# Analyze with snakeviz
snakeviz profile.stats
```

## Common Tasks

### Adding a New Pipeline

1. Create pipeline class in `src/pipelines/`
2. Implement `BasePipeline` interface
3. Add configuration schema
4. Create unit tests
5. Add integration tests
6. Update documentation
7. Add example usage

### Adding Dependencies

1. Add to `pyproject.toml`
2. Update `requirements.txt`
3. Test installation
4. Update documentation

### Updating Models

1. Test new model performance
2. Update configuration options
3. Add backward compatibility
4. Update documentation
5. Add performance benchmarks

Thank you for contributing to VideoAnnotator! Your contributions help make video analysis more accessible and powerful for everyone.
