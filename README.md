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

