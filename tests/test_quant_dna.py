"""
Tests for QuantDNA module
"""

import pytest
import torch
from quant_dna import QuantDNA, QuantizationConfig


class TestQuantDNAInitialization:
    """Test QuantDNA initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        quant_dna = QuantDNA()
        assert quant_dna.num_freqs == 64
        assert quant_dna.hidden_size == 128
        assert quant_dna.bits == 8
        assert quant_dna.symmetric == True
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        quant_dna = QuantDNA(num_freqs=32, hidden_size=64, bits=4, symmetric=False)
        assert quant_dna.num_freqs == 32
        assert quant_dna.hidden_size == 64
        assert quant_dna.bits == 4
        assert quant_dna.symmetric == False
    
    def test_buffer_registration(self):
        """Test that B is registered as buffer"""
        quant_dna = QuantDNA()
        assert hasattr(quant_dna, 'B')
        assert isinstance(quant_dna.B, torch.Tensor)
        assert quant_dna.B.shape == (64, 2)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        quant_dna = QuantDNA(num_freqs=32, hidden_size=64)
        count = quant_dna.count_parameters()
        # net = Linear(64->64) + Linear(64->64) + Linear(64->1), plus scale
        expected = (64*64 + 64) + (64*64 + 64) + (64*1 + 1) + 1
        assert count == expected


class TestQuantDNAWeightGeneration:
    """Test weight generation"""
    
    def test_generate_dense_weights(self):
        """Test generating dense weights"""
        quant_dna = QuantDNA()
        W = quant_dna.generate_weights(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_quantized_weights(self):
        """Test generating quantized weights"""
        quant_dna = QuantDNA(bits=8)
        W = quant_dna.generate_quantized_weights(10, 10)
        assert W.shape == (10, 10)


class TestQuantDNAQuantization:
    """Test quantization"""
    
    def test_binary_quantization(self):
        """Test binary quantization"""
        quant_dna = QuantDNA()
        W = torch.randn(10, 10)
        W_quant = quant_dna.quantize(W, bits=1)
        unique = torch.unique(W_quant)
        assert len(unique) <= 3  # -1, 0, 1
    
    def test_ternary_quantization(self):
        """Test ternary quantization"""
        quant_dna = QuantDNA()
        W = torch.randn(10, 10)
        W_quant = quant_dna.quantize(W, bits=2)
        unique = torch.unique(W_quant)
        assert len(unique) <= 3  # -1, 0, 1
    
    def test_multi_bit_quantization(self):
        """Test multi-bit quantization"""
        quant_dna = QuantDNA()
        W = torch.randn(10, 10)
        W_quant = quant_dna.quantize(W, bits=8)
        assert W_quant.shape == W.shape


class TestQuantDNALayerWeights:
    """Test layer weight generation"""
    
    def test_generate_quantized_layer_weights(self):
        """Test generating quantized weights for multiple layers"""
        quant_dna = QuantDNA()
        layer_config = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = quant_dna.generate_quantized_layer_weights(layer_config, bits=8)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)


class TestQuantDNAConfiguration:
    """Test configuration"""
    
    def test_set_quantization(self):
        """Test setting quantization"""
        quant_dna = QuantDNA(bits=8)
        quant_dna.set_quantization(bits=4, symmetric=False)
        assert quant_dna.bits == 4
        assert quant_dna.symmetric == False
    
    def test_get_quantization_info(self):
        """Test getting quantization info"""
        quant_dna = QuantDNA(bits=8, symmetric=True)
        info = quant_dna.get_quantization_info()
        assert info['bits'] == 8
        assert info['symmetric'] == True


class TestQuantDNADeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        quant_dna = QuantDNA()
        W = quant_dna.generate_quantized_weights(10, 10)
        assert W.device == quant_dna.B.device
        assert W.device == quant_dna.scale.device
        for param in quant_dna.net.parameters():
            assert param.device == quant_dna.B.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        quant_dna = QuantDNA()
        W = quant_dna.generate_quantized_weights(10, 10)
        assert W.dtype == quant_dna.B.dtype
        assert W.dtype == quant_dna.scale.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
