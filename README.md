# QuantDNA

QuantDNA is a small Python/PyTorch prototype for generating dense matrices from a coordinate-based neural network with Fourier features, then applying simple tensor quantization. Despite the project name and source comments, the implementation does not encode data as DNA; it is a neural weight-field generator.

## Status

This repository contains one source module and unit tests. The core generation and quantization routines are implemented, but this is not a validated weight-compression or model-quantization system. There is no training or fitting workflow, trained-model integration, packed low-bit storage, quantized compute kernel, or accuracy/performance evaluation. Treat generated matrices as outputs from a randomly initialized generator, not as useful pretrained weights.

## Implemented functionality

- `QuantDNA` builds a two-coordinate Fourier-feature representation and a small MLP, then evaluates it over a coordinate grid to produce a dense `(out_dim, in_dim)` tensor.
- The quantizer provides sign-based 1-bit output, mean-absolute-threshold ternary output for `bits=2`, and round/scale behavior for other bit values. The result remains a regular PyTorch tensor; values are not bit-packed, and the multi-bit path does not clamp values to a quantized range.
- `generate_quantized_layer_weights` returns a dictionary of generated matrices for supplied layer dimensions. The module also exposes parameter counting and basic quantization-configuration helpers.

The `symmetric` option changes the scale used by the multi-bit path; the 1-bit and ternary paths do not use it. The `QuantizationConfig` dataclass is defined but is not integrated into the `QuantDNA` methods, and its `per_channel` field does not implement per-channel quantization.

## Requirements and use from a checkout

Python 3.10 or later is declared in `pyproject.toml`. The requirements file lists PyTorch 2.0 or later, NumPy 1.24 or later, and pytest 7 or later.

From the repository root, install those dependencies and run the included example:

```sh
python -m pip install -r requirements.txt
python quant_dna.py
```

The module can also be imported directly while the repository root is on Python's import path:

```python
from quant_dna import QuantDNA

model = QuantDNA(num_freqs=64, hidden_size=128, bits=8)
weights = model.generate_quantized_weights(out_dim=100, in_dim=100)
print(weights.shape)
```

There is no dedicated console-script command. Although `pyproject.toml` declares project metadata, a wheel built from this revision contains metadata but omits `quant_dna.py`; installing the built wheel therefore does not provide the importable module. Use the source checkout rather than relying on `pip install .` until packaging is corrected.

## Tests and checks

On the inspected commit, `python -m pytest -q` completed with **14 passed**. The configured CI compile check (`python -m compileall -q quant_dna.py tests`) and its import/generation smoke check also passed locally. The workflow configuration requests the pytest job on Python 3.10, 3.11, and 3.12, but no hosted CI run status is asserted here. The repository includes a CodeQL workflow; its result was not checked as part of this review.

The tests cover initialization, parameter counting, output shapes and dtypes, basic quantization behavior, layer-dictionary generation, and configuration helpers. They do not measure model accuracy, quantization error on trained weights, actual storage savings, or runtime speed. No benchmark or research result is reported here.

## Limitations

- The generator is randomly initialized; there is no fitting/training routine or evidence that its outputs approximate a target model's weights.
- Quantization is applied after dense floating-point matrix generation. It does not reduce the returned tensor's storage or provide faster inference.
- The multi-bit path rounds and rescales without clipping, and the API does not validate that `bits` is a supported or meaningful value.
- Layer support is limited to producing a name-to-tensor dictionary; there is no conversion or integration with actual PyTorch model layers.
- No accuracy, compression, memory, or speed results are supplied.
- The built wheel currently omits the source module, as noted above.

## License

The repository's `LICENSE` file contains the MIT License, copyright 2026 Kapil Dhinwa. See [LICENSE](LICENSE) for the full terms.
