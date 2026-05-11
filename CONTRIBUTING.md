# Contributing to Halo-VLA

Thank you for your interest in contributing to Halo-VLA! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Getting Started

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/halo-vla.git
   cd halo-vla
   ```

3. Set up your development environment:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting and import sorting
- **isort**: Import organization
- **mypy**: Static type checking

Run formatting and linting:
```bash
make format
make lint
make type-check
```

### Writing Tests

- Add tests for all new features in the `tests/` directory
- Use pytest fixtures for reusable test components
- Aim for high code coverage

Run tests:
```bash
make test          # Run all tests
make test-cov      # Run with coverage report
```

### Commit Messages

Follow these guidelines for commit messages:
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally after the first line

Example:
```
Add Vision Transformer module

This commit introduces the ViT implementation with:
- Patch embedding layer
- Multi-head attention
- Feed-forward networks

Closes #123
```

### Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes and commit them
3. Push to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```

4. Open a Pull Request with a clear title and description
5. Ensure all CI checks pass

## Documentation

- Update docstrings for all public functions and classes
- Use Google-style docstrings
- Include type hints for all function parameters and returns

Example:
```python
def compute_embedding(image: torch.Tensor, model: ViT) -> torch.Tensor:
    """Compute image embedding using Vision Transformer.
    
    Args:
        image: Input image tensor of shape (batch_size, channels, height, width).
        model: Vision Transformer model instance.
        
    Returns:
        Image embedding tensor of shape (batch_size, embedding_dim).
    """
    return model(image)
```

## Reporting Issues

Before creating bug reports, please check the issue list as you might find out that you don't need to create one.

When creating a bug report, please include:
- Clear title and description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)
- Code samples if applicable

## Feature Requests

Feature requests are welcome! Please provide:
- Clear description of the feature
- Use case and motivation
- Possible implementation approach (optional)

## License

By contributing to Halo-VLA, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion for any questions about contributing.
