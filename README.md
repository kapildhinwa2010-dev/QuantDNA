# QuantDNA

[![CI](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/QuantDNA/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Fourier-feature weight generator with built-in quantization (1/2/4/8-bit).

## What it does

QuantDNA generates a dense weight matrix from a small coordinate-MLP
(same Fourier-feature family as FourierNet) and then quantizes the
result: binary sign, ternary threshold, or multi-bit rounding, with an
optional symmetric range.

## Installation

    pip install torch numpy pytest

## Usage

    from quant_dna import QuantDNA

    quant_dna = QuantDNA(num_freqs=64, hidden_size=128, bits=8)
    W = quant_dna.generate_quantized_weights(1000, 1000)

    W2 = quant_dna.generate_quantized_weights(100, 100, bits=2)
    quant_dna.set_quantization(bits=4, symmetric=True)
    info = quant_dna.get_quantization_info()

## API

- QuantDNA(num_freqs=64, hidden_size=128, bits=8, symmetric=True)
- generate_weights / quantize / generate_quantized_weights /
  generate_quantized_layer_weights / set_quantization /
  count_parameters / get_quantization_info

## Honest results and limitations

- The storage-savings column of any ratio table here is arithmetic
  (32x for binary means 32-bit float to 1 bit). It is not a measured
  end-to-end result.
- The generator output is a smooth random field unless fitted to a
  target; quantizing an unfitted generator does not give useful
  pretrained weights.
- Quantization is applied post-hoc to dense floats; there is no
  quantized-matrix kernel, so computation speed does not improve in
  this repo.
- No accuracy evaluation is included: quantization error on real
  trained weights is unmeasured here.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that generated-weight
  reconstructions of trained models were not usable without per-layer
  fitting, and the related carrier-mode recall failed about half of
  noise trials. The robust design kept in the toolkit is the
  Reed-Solomon slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
