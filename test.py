import os
from typing import OrderedDict
import torch
import torch.nn as nn
from util.data_util import ModelNet40 as ModelNet40
from torch.utils.data import DataLoader
import numpy as np
from util.util import load_cfg_from_cfg_file

from pathlib import Path

import copy
from easydict import EasyDict as edict
from qqquantize.quantize import DEFAULT_QAT_MODULE_MAPPING, ModelConverter
import qqquantize.observers as qob
from qqquantize.observers import FakeQuantize, toggle, MovingAverageMinMaxObserver, MinMaxObserver
import qqquantize.qmodules as qm
from qqquantize.savehook import register_intermediate_hooks
from model.interpcnn2_utils import BatchMatMul
from model.interpcnn2 import PointMaxPooling, PointDownsample
import pickle
import lzma, time
from ptflops import get_model_complexity_info
if __name__ == '__main__':
    args = edict()
    args.config = '/home/luojiapeng/obj_cls/config/interpcnn2_test.yaml'
    args.out_dir = 'export_data/interpcnn2'
    args.ckpt = '/home/luojiapeng/obj_cls/checkpoints/interpcnn2_train_quant/best_model.t7'
    #args.device = torch.device("cuda:3")
    args.device = torch.device("cpu")
    args.out_file = Path(args.out_dir) / 'data.pkl.lzma'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cfg = load_cfg_from_cfg_file(args.config)

    """dataloader = DataLoader(ModelNet40(partition='test', num_points=cfg.num_points, pt_norm=False),
                    num_workers=4, batch_size=1, shuffle=False, drop_last=False)"""
    
    device = args.device
    if cfg.arch == 'dgcnn':
        from model.DGCNN_PAConv import PAConv
        model = PAConv(cfg).to(device)
    elif cfg.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(cfg).to(device)
    elif cfg.arch == 'interpcnn2':
        from model.interpcnn2 import InterpCNN2, PointMaxPooling, InterPConv, PointDownsample
        model = InterpCNN2(cfg).to(device)
    else:
        raise Exception("Not implemented")
    
    #toggle(model, quantize=False, observer=False, calc_qparam=False)
    model.eval()
    model_name='interpcnn2'
    flops, params = get_model_complexity_info(model, (3,1024),as_strings=True,print_per_layer_stat=True)
    print("%s |%s |%s" % (model_name,flops,params))
    
    info = pickle.load(lzma.open('/home/luojiapeng/obj_cls/data.pkl.lzma', 'rb'))
    data=torch.tensor(info['input_stub.act_quant']['values'])
    print(data.shape)
    data=data.to(device)
    print(data.shape)
    for i in range(0,5):
        data=torch.cat((data,data),0)
    print(data.shape)
    tic = time.time()
    for i in range(0,100):
        logits = model(data)
    
    #data = hookManager.output_data()

    ### because act and bias share the same quantizer, we use list to store data
    ### split them here
    
    print(f'finished ... ({time.time() - tic:.3f} secs)')
