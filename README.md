### LPN ###
This is the code for paper "A Low-Latency Framework with Algorithm-Hardware  Co-Optimization for 3D Point Cloud". The structure of LPN model is 
in model/interpcnn2.py. You can test the model by using eval_voting.py. The config file is config/interpcnn2_test.py.

As for the environment and dataset, please refer to the following descriptions. 

3D Object Classification
============================

## Installation

### Requirements
* Hardware: GPU to hold 6000M. (Better with two gpus or higher-level gpu to satisfy the need of paralleled cuda_kernels.)
* Software: 
  Linux (tested on Ubuntu 18.04)
  PyTorch>=1.5.0, Python>=3, CUDA>=10.1, tensorboardX, h5py, pyYaml, scikit-learn


### Dataset
Download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) (415M). Then symlink the paths to it as follows (you can alternatively modify the path [here](https://github.com/CVMI-Lab/PAConv/blob/main/obj_cls/util/data_util.py#L10)):
``` 
mkdir -p data
ln -s /path to modelnet40/modelnet40_ply_hdf5_2048 data
```

## Usage

* Build the CUDA kernel: 

    When you run the program for the first time, please wait a few moments for compiling the [cuda_lib](./cuda_lib) **automatically**.
    Once the CUDA kernel is built, the program will skip this in the future running. 


* Train:

   * Multi-thread training ([nn.DataParallel](https://pytorch.org/docs/stable/nn.html#dataparallel)) :

     * `python main.py --config config/dgcnn_paconv_train.yaml` (Embed PAConv into [DGCNN](https://arxiv.org/abs/1801.07829))
    
     * `python main.py --config config/pointnet_paconv_train.yaml` (Embed PAConv into [PointNet](https://arxiv.org/abs/1612.00593))

   * We also provide a fast **multi-process training** ([nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html), **recommended**) with official [nn.SyncBatchNorm](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm). Please also remind to specify the GPU ID:
   
     * `CUDA_VISIBLE_DEVICES=x,x python main_ddp.py --config config/dgcnn_paconv_train.yaml` (Embed PAConv into [DGCNN](https://arxiv.org/abs/1801.07829))
     * `CUDA_VISIBLE_DEVICES=x,x python main_ddp.py --config config/pointnet_paconv_train.yaml` (Embed PAConv into [PointNet](https://arxiv.org/abs/1612.00593))


* Test:

  * Download our [pretrained model](https://drive.google.com/drive/folders/1eDBpIRt4iSCjEw2-Mk2G3gz7YwA6VfEB?usp=sharing) and put it under the [obj_cls](/obj_cls) folder.

  * Run the voting evaluation script to test our pretrained model, after this voting you will get an accuracy of 93.9% if all things go right:
  
    `python eval_voting.py --config config/dgcnn_paconv_test.yaml`
    
  * You can also directly test our pretrained model without voting to get an accuracy of 93.6%:
  
    `python main.py --config config/dgcnn_paconv_test.yaml`
    
  * For full test after training the model:
    * Specify the `eval` to `True` in your config file.
    
    * Make sure to use **[main.py](main.py)** (main_ddp.py may lead to wrong result due to the repeating problem of all_reduce function in multi-process training) :
    
      `python main.py --config config/your config file.yaml`
  
* Visualization: [tensorboardX](https://github.com/lanpa/tensorboardX) incorporated for better visualization.

   `tensorboard --logdir=checkpoints/exp_name`
   
 


