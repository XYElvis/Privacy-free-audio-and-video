
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：628test 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：咋
@Date    ：2023/6/28 23:24 
"""
import torch
print(torch.cuda.is_available())
print(torch.__version__)
cuda_version = torch.version.cuda
print("CUDA 版本:", cuda_version)
device = torch.cuda.get_device_name()
print("名称:", device)

