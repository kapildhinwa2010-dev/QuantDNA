# QuantDNA

[![CI](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**QuantDNA** is a quantized weight generation framework that combines DNA-based weight generation with quantization, enabling efficient storage and computation for neural networks with limited precision.

### Key Features

- ✅ **Quantized Weights**: Generate weights with reduced precision (8-bit, 4-bit, 2-bit, binary)
- ✅ **DNA-Based**: Uses Fourier features for weight generation
- ✅ **Memory Efficient**: Quantized storage reduces memory usage
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models
- ✅ **Configurable**: Adjustable quantization levels

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/QuantDNA.git
cd QuantDNA
pip install -e .
```

## Usage

### Basic Quantized Weight Generation

```python
from quant_dna import QuantDNA

# Create QuantDNA instance
quant_dna = QuantDNA(num_freqs=64, hidden_size=128, bits=8)

# Generate quantized weights
W = quant_dna.generate_quantized_weights(1000, 1000)
print(f"Generated quantized matrix: {W.shape}")
```

### Different Quantization Levels

```python
# 8-bit quantization (default)
W8 = quant_dna.generate_quantized_weights(100, 100, bits=8)

# 4-bit quantization
W4 = quant_dna.generate_quantized_weights(100, 100, bits=4)

# 2-bit (ternary) quantization
W2 = quant_dna.generate_quantized_weights(100, 100, bits=2)

# Binary quantization
W1 = quant_dna.generate_quantized_weights(100, 100, bits=1)
```

### Generate for Multiple Layers

```python
# Generate quantized weights for multiple layers
layer_config = {
    'fc1': (256, 128),
    'fc2': (128, 256),
}
weights = quant_dna.generate_quantized_layer_weights(layer_config, bits=8)

for name, W in weights.items():
    print(f"{name}: {W.shape}")
```

### Change Quantization Configuration

```python
# Change quantization settings
quant_dna.set_quantization(bits=4, symmetric=True)

# Get current quantization info
info = quant_dna.get_quantization_info()
print(f"Bits: {info['bits']}, Symmetric: {info['symmetric']}")
```

## Quantization Levels

| Bits | Type | Unique Values | Storage Savings | Use Case |
|------|------|---------------|-----------------|----------|
| 1 | Binary | 2 | ~32x | Extreme compression |
| 2 | Ternary | 3 | ~16x | Very high compression |
| 4 | Quaternary | 16 | ~8x | High compression |
| 8 | 8-bit | 256 | ~4x | General purpose |

## Architecture

### Quantized Weight Generation

QuantDNA combines DNA-based weight generation with quantization:

```
Coordinates → Fourier Features → DNA Network → Dense Weights → Quantization → Quantized Weights
```

### Quantization Methods

- **Binary (1-bit)**: Sign function (values: -1, 0, 1)
- **Ternary (2-bit)**: Threshold-based (values: -1, 0, 1)
- **Multi-bit**: Rounding to nearest representable value

### Advantages

- **Memory Efficiency**: Reduced precision = reduced memory
- **Computational Efficiency**: Quantized operations can be faster
- **Hardware-Friendly**: Works well with specialized hardware
- **Regularization**: Quantization acts as a form of regularization

## API Reference

### QuantDNA

```python
QuantDNA(
    num_freqs: int = 64,
    hidden_size: int = 128,
    bits: int = 8,
    symmetric: bool = True
)
```

**Methods:**
- `generate_weights(out_dim: int, in_dim: int) -> Tensor`: Generate dense weights
- `quantize(W: Tensor, bits: int, symmetric: bool) -> Tensor`: Quantize a tensor
- `generate_quantized_weights(out_dim: int, in_dim: int, bits: int) -> Tensor`: Generate quantized weights
- `generate_quantized_layer_weights(layer_config: Dict, bits: int) -> Dict`: Generate quantized weights for layers
- `set_quantization(bits: int, symmetric: bool) -> None`: Set quantization configuration
- `count_parameters() -> int`: Count DNA network parameters
- `get_quantization_info() -> Dict`: Get quantization configuration

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Neural network quantization
- Binary and ternary networks
- Fourier feature networks
- Efficient deep learning
