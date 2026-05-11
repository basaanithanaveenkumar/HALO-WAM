# Halo-VLA: Vision-Language Assistant

A PyTorch-based Vision-Language Model (VLA) implementation combining visual and linguistic understanding.

## Features

- **Vision Transformer (ViT)**: State-of-the-art image encoding
- **Transformer Backbone**: Multi-head attention mechanism for sequence modeling
- **Language Model Head**: Causal language modeling for text generation
- **Mixture of Experts**: Efficient multi-expert model architecture
- **Positional Embeddings**: Learnable positional encoding for sequences
- **Image Projection**: Efficient image-to-embedding projection layer

## Project Structure

```
Halo-VLA/
├── models/
│   ├── __init__.py
│   ├── ha_vlm.py              # Main VLA model
│   ├── vlm.py                 # Base VLM class
│   ├── vit.py                 # Vision Transformer
│   ├── transformer.py         # Transformer encoder/decoder
│   ├── lm_head.py            # Language model head
│   ├── image_proj.py         # Image projection layer
│   ├── moe.py                # Mixture of Experts
│   └── positional_embeddings.py  # Positional encoding
├── tests/
│   ├── __init__.py
│   └── test_models.py        # Model tests
├── pyproject.toml            # Project configuration
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Installation

### Using UV (Recommended)

```bash
# Install the package in development mode
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
pip install -e ".[dev]"
```

## Quick Start

```python
from models import VLM, ViT, Transformer

# Initialize components
vit = ViT(...)
transformer = Transformer(...)
vla_model = VLM(vision_encoder=vit, language_model=transformer)

# Forward pass
output = vla_model(images, text_tokens)
```

## Development

### Setup Development Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
pytest --cov=models  # With coverage
```

### Code Formatting and Linting

```bash
black .           # Format code
ruff check .      # Lint
isort .           # Sort imports
mypy models       # Type checking
```

## Requirements

- Python ≥ 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- numpy >= 1.24.0

See [pyproject.toml](pyproject.toml) for complete dependency list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
