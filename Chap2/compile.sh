#!/bin/bash

nvcc sum.cu -o sum
nvcc checkDeviceInfo.cu -o checkDeviceInfo
nvcc sumMatrix1.cu -o sumMatrix1