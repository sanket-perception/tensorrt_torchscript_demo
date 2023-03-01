# Deployment Tools 
In this repository, we have summarized several deployment pathways with small-scale examples. Deployment here, strictly refers to converting a Pytorch Model, into an executable (or a series of executables) that can be integrated in an already existing C++ based project, on a specific hardware. In practice, even though PyTorch based development is done in python, one should not simply deploy such models in python. Instead, these models should be converted to C++ executables, after adding further inference centric optimizations. 


## A. System Setup
Before we dive further into specifics, we must first get our system ready. 
1. First install required libraries using ``` requirements.txt```.


    ```
    pip3 install -r requirements.txt
    ```
2. Then, clone the repository locally :

    ```
    git clone https://github.com/sanket-perception/tensorrt_torchscript_demo.git
    ```



## B. Optimizing Classification Model in ONNX + TensorRT

There are several pathways one can choose from, in order to take a model from development to production. A clear winner amongst all existing methods is TensorRT, NVIDIA's flagship model optimization framework and inference engine generator. In simple words, TensorRT takes a Torch/TensorFlow model and converts it into an "engine" such that it makes most of the existing hardware resource. We will first look at a simple example of converting a classification model into TensorRT engine. Refer to the notebook inside [this notebook](/notebooks/Simple%20Classification%20Model%20-%20TensorRT%20%2B%20ONNX.ipynb).

## C. Optimizing Classification Model in TorchScript

TensorRT is an unparalleled solution for model deployment,however, for Pytorch or ONNX-based models it has incomplete support and suffers from poor portability. It is difficult to integrate TensorRT with custom torch operations, which is the case more often than not. There is a plugin system to add custom layers and postprocessing, but this low-level work is out of reach for groups without specialized deployment teams. TensorRT also doesn’t support cross-compilation so models must be optimized directly on the target hardware — not great for embedded platforms or highly diverse compute ecosystems. Thus, in conlusion, TensorRT is not the ultimate answer to the deployment puzzle. Another candidate for solving this problem is TorchScript. 

TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

It is a tool to incrementally transition a model from a pure Python program to a TorchScript program that can be run independently from Python, such as in a standalone C++ program. This makes it possible to train models in PyTorch using familiar tools in Python and then export the model via TorchScript to a production environment where Python programs may be disadvantageous for performance and multi-threading reasons.

Let us look at the same model as we explored with TensorRT, using TorchScript to understand how this framework works. Please refer to [this notebook](/notebooks/Simple%20Classification%20Model%20%20TorchScript.ipynb) for further details.


