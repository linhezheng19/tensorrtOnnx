# tensorrtOnnx
Implementation of some methods or tasks with TensorRT and ONNX.

### Requirements

- TensorRT >= 7.0.0.11
- OpenCV >= 3.4.0
- yaml-cpp
- cuda

### Changelog

Please refer to [changelog.md](docs/changelog.md) for details and change history.

### Getting Start

First clone this repo to your workspace, and enter this folder, then execute commands following below. After the executable file is generated, run it. You need to modified the `cfgs/main.yaml` to turn on the task you want to excute.

```
$ cp ${ONNX_MODEL_PATH} models
$ mkdir build && cd build
$ cmake .. && make -j
$ ./engine
```

### Performance
`# Waiting for update.`

### Notes

- All parameters can change in `cfgs/*.yaml` and `cfgs/tasks/*.yaml`. You can change it as your requirement.
- If you deploy this project on NX or Nano devices, you may need to modified the **compute arch** in  `CMakeLists.txt`, more details in [Nvidia](https://developer.nvidia.com/cuda-gpus) official webs.
- If you deploy this project on GPU server, make sure you install **CUDA** and **Cudnn** with corresponding version with TensorRT, and modified `{TRT_ROOT}` add **cuda include directory** in `CMakeLists.txt`.
