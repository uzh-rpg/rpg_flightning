# Learning Quadrotor Control From Visual Features Using Differentiable Simulation

<p align="center">
  <a href="https://youtu.be/LdgvGCLB9do">
    <img src="https://img.youtube.com/vi/LdgvGCLB9do/maxresdefault.jpg" alt="youtube_video" width="800"/>
  </a>
</p>


This repository contains the flightning python package to learn quadrotor
control policies using differentiable simulation. If you use this code,
please cite the following publication (https://arxiv.org/abs/2410.15979):

```tex
@misc{heegIcra2025,
      title={Learning Quadrotor Control From Visual Features Using Differentiable Simulation}, 
      author={Johannes Heeg and Yunlong Song and Davide Scaramuzza},
      year={2025},
      booktitle={IEEE International Conference on Robotics and Automation, 2025}
      url={https://arxiv.org/abs/2410.15979}, 
}
```
## Abstract

This work demonstrates the potential of differentiable simulation for learning 
quadrotor control. To the best of our knowledge, this work shows the first 
application of differentiable-simulation-based learning to low-level quadrotor 
control.
We show that training in differentiable simulation significantly outperforms 
model-free RL in terms of both sample efficiency and training time, allowing a 
policy to learn to recover a quadrotor in seconds when providing vehicle state 
and in minutes when relying solely on visual features.
The key to our success is two-fold. First, the use of a simple surrogate model 
for gradient computation greatly accelerates training without sacrificing 
control performance. Second, combining state representation learning with policy
learning enhances convergence speed in tasks where only visual features are 
observable.
These findings highlight the potential of differentiable simulation for
real-world robotics and offer a compelling alternative to conventional
RL approaches.

## Installation

JAX ony supports GPU acceleration on Linux. If you are using Windows or MacOS,
you can still run the code, but it will be slower.

### Create Environment

```bash
mamba create -n flightning python=3.9
mamba activate flightning
```

### Install Flightning

To install the flightning package, run the following commands in the root
directory of this repository:

```bash
# install flightning requirements
pip install -r requirements.txt
# install flightning
pip install --use-pep517 -e .
```

## Introduction

The `examples` directory contains jupyter notebooks that demonstrate how to use 
Flightning to train a neural network to control a quadrotor.

```bash
jupyter-lab examples
```


## License

The code in this project is licensed under the GPL-3.0 license.

Portions of this project are based on code from 
[PureJaxRL](https://github.com/luchris429/purejaxrl) and 
[gymnax](https://github.com/RobertTLange/gymnax), licensed
under the Apache License 2.0.
See `THIRD_PARTY_LICENSES/APACHE_LICENSE.txt` for details.
