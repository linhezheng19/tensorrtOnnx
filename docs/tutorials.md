## Simple Tutorials
All task implement two basic parts, pipeline and post process. Post could be programmed in CUDA, simple cuda examples in [here](https://github.com/00hz/cuda-rt-examples.git).

### CLS
Generate onnx model by torch.onnx.export, move it to `models` folder, and modified configuration needed in `cfg/tasks/cls.yaml`.

### Semantic Segmentation
Generate onnx model by torch.onnx.export, move it to `models` folder, and modified configuration needed in `cfg/tasks/semseg.yaml`.
#### Note
Add an argmax layer at last of the model, since the cuda post porcess not implement yet, if do it on CPU, it will be slow.

### YOLOv5
Convert yolov5.pt to onnx model following original repo, post process is implement on CPU, CUDA version not implement yet.

### FCOS
Convert fcos output as 9, 3 center_ness, 3 reg, 3 classification.

### F_Track
Track network with FCOS detection and Re-ID branch. Just replace detector in FairMOT.

### FairMOT
FairMOT.
