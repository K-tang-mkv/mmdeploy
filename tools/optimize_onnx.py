import onnx
from onnxoptimizer import optimize_model


if __name__ == "__main__":
    model = onnx.load('onnx/end2end.onnx')
    optimized_model = optimize_model(model)
    onnx.save(optimized_model, 'onnx/optimized.onnx')