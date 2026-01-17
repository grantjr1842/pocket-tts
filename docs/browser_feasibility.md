# Browser/WASM Feasibility Assessment

This document outlines the feasibility of running pocket-tts in the browser via WebAssembly.

## Current Status: **Research Phase**

### Challenges

1. **Model Size**: The full model is ~600MB, which is prohibitive for browser download
2. **Streaming State**: pocket-tts uses streaming/stateful transformers with KV caching that don't export cleanly to ONNX
3. **Audio Codec**: The Mimi audio encoder/decoder uses complex convolutions that may not have WASM kernels
4. **Memory**: Browsers have memory limits (~2-4GB) that may be insufficient for full model

### Potential Approaches

#### 1. ONNX.js / ONNX Runtime Web (Recommended)

- Export model components to ONNX format
- Use ONNX Runtime Web for inference
- **Blocker**: Streaming state management is complex in ONNX

#### 2. TensorFlow.js

- Convert PyTorch model to TensorFlow format
- Use TFLite or WebGL backend
- **Blocker**: Conversion is complex and may lose functionality

#### 3. Server-Assisted Hybrid

- Run lightweight portions in browser
- Heavy computation on server via WebSocket/WebRTC
- **Best for**: Low-latency streaming applications

### Recommended Next Steps

1. Try exporting a simplified, non-streaming version of the model to ONNX
2. Benchmark ONNX Runtime Web performance on a single forward pass
3. If successful, implement state management in JavaScript

### Files Added

- `pocket_tts/utils/export_model.py` - TorchScript export (foundation for ONNX)
- This research document

### Future Work

- Quantization may help reduce model size (see Issue #7)
- Consider progressive model loading for browser
- Explore WebGPU for better performance (emerging standard)
