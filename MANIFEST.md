title: gr-theano
brief: blocks leveraging the theano library to run code in graphics cards
tags:
  - python
  - GPU
  - cuda
author:
  - Tim O'Shea <tim.oshea753@gmail.com>
copyright_owner:
  - Tim O'Shea <tim.oshea753@gmail.com>
dependencies:
  - gnuradio (>= 3.7.0)
repo: https://github.com/osh/gr-theano.git
stable_release: HEAD
---

GNU Radio Theano block library!

This OOT module contains a number of GNU Radio blocks which
leverage the theano library to accelerate signal processing
code on graphcis cards typically using Cuda as a backend.

# Current Blocks

- Signal Source (sinusoid complex64)
- Fading Model (single tap complex64)
- FIR Filter (float32)
- FFT (complex64)

# Information

For the latest verison please see
[gr-theano Github repo](https://github.com/osh/gr-theano)

For more information on the theano project please see
- [Theano Github repo](https://github.com/Theano/Theano)
- [Theano Documentation](http://deeplearning.net/software/theano/)
- [Theano tutorial](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb)
- [A video on Transparent GPU computing with Theano](https://archive.org/details/Scipy2010-JamesBergstra-TransparentGpuComputingWithTheano)

# Howto build

1. Install GNU Radio 3.7+
2. Install Theano
    pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
3. Install gr-threano just like any GNU Radio OOT Module
