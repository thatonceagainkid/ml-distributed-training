# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:10:15 2026

@author: clayt
"""
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print("=" * 50)
print("ENVIRONMENT SNAPSHOT")
print("=" * 50)

print(f"torch:        {torch.__version__}")
print(f"torchvision:  {torchvision.__version__}")
print(f"numpy:        {np.__version__}")
print(f"pandas:       {pd.__version__}")
print(f"sklearn:      {sklearn.__version__}")
print(f"matplotlib:   {matplotlib.__version__}")

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print("=" * 50)
