# GPU Side Channel Attack

This repository implements a side-channel attack on Jetson Nano that allows the attacker to do website fingerprinting on the victims browser and track users behavior.

This is a proof-of-concept from the following paper - [Side Channel Attacks on GPUs](https://ieeexplore.ieee.org/abstract/document/8852671)

We use the edge device [Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/) to demonstrate the attack.

## Get Started

### Install CUDA
Use the following steps to [Install CUDA on Jetson Nano](https://jfrog.com/connect/post/installing-cuda-on-nvidia-jetson-nano/)

### Run the code
```bash
# Build the file
mkdir build
make

# Run the code
make run

# You should now see the performance metrics being returned every second
# These metrics can be used to infer behavior of the user
# Check the folder `analysis/` for more information
```

### Get list of available performance metrics
```
cd /usr/local/cuda/extras/CUPTI/samples/cupti_query
sudo ./cupti_query -device 0 -getmetrics
```
