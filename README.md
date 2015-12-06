LPMRF
=====

MATLAB and C++/MEX code that trains the Fixed-Length Poisson MRF model (LPMRF) described in the following paper:  

[Fixed-Length Poisson MRF: Adding Dependencies to the Multinomial](http://www.cs.utexas.edu/~dinouye/papers/inouye2015-fixed-length-poisson-mrfs-nips2015.pdf) ([pdf](http://www.cs.utexas.edu/~dinouye/papers/inouye2015-fixed-length-poisson-mrfs-nips2015.pdf), [poster](http://www.cs.utexas.edu/~dinouye/presentations/poster-nips2015-lpmrf.pdf)
D. Inouye, P. Ravikumar, I. Dhillon  
*Neural Information Processing Systems (NIPS)*, 28, 2015.

Install
=======

1. Open MATLAB and change directories into "mexcode".
2. Run "makemex()" to compile the mex files (unless the included linux 64-bit compiled mex files work with your system)

Demo
====

To run the demo, simply have "+mrfs" on your MATLAB path and run "mrfs.demo_lpmrf()".  The demo shows how to call the main functions.

Requirements
============

* MATLAB
* Compiler with C++11 support (used for random number generation in C++ file)

Developer
=========

David I. Inouye  
Website: http://cs.utexas.edu/~dinouye  
Email: dinouye@cs.utexas.edu
