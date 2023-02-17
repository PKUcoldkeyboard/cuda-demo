#!/bin/bash

nvcc sum.cu -o sum
nvcc checkDeviceInfo.cu -o checkDeviceInfo