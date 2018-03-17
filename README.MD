## Synopsis

Baseline correction algorithm for correcting (chemical) chromatograms. This is a C++ implementation of the BEADS algorithm, the original paper and Matlab implementation are from from X. Ning, I. Selesnick and L. Duval http://dx.doi.org/10.1016/j.chemolab.2014.09.014.
This implementation uses the EIGEN library for linear algebra. http://eigen.tuxfamily.org/index.php?title=Main_Page This implementation uses sparse matrices to reduce the computational complexity of BEADS.

## Code Example

A minimum working example is provided in Beadstest.cpp:  
//   y: Noisy observation  
//   d: Filter order (d = 1 or 2)  
//   fc: Filter cut-off frequency (cycles/sample) (0 < fc < 0.5)  
//   r: Asymmetry ratio  
//   lam0, lam1, lam2: Regularization parameters  
std::vector<std::vector<double>> AB = myBaseline.beads(y, d, fc, r, lam0, lam1, lam2);  

where AB.at(0) is the baseline corrected chromatogram  
AB.at(1) is the baseline.  

## Motivation

Hardly any C++ baseline correction implementation exists in the public domain. Hopefully this will bring some change to that.  

## Installation

1. Download EIGEN from http://eigen.tuxfamily.org/index.php?title=Main_Page    
2. Extract the zip file  
3. Copy the eigen-eigen-xxx/Eigen folder to your project working directory  
4. Download the files from this github page through git clone or http download  
5. Past project content in working directory 
6. Modify SparseQR.h of EIGEN line 314: OUTCOMMENT "ord(matCpy, m_perm_c);" (This is a bug in EIGEN, assert is triggered without this modification)
6. Compile should work now 

## API Reference

There is no API at the moment, this is just a bare-metal implementation. Don't expect great code performance from this.  
A chromatogram with 4000 data points takes roughly 2.5s to complete baseline correction on a intel i7-2630HQ.   

## Tests

Compile Beadstest.cpp using VS 2017 64 bit release. A small section outcommented code containing ofstream statements can write the corrected chromatogram to a text file.  

## Contributors

Created by T. Vijverberg - 01-01-2018 @Radboud University  
Contact me at tgmvijverberg at online dot nl  


## License

BEADS baseline correction is developed by X. Ning, I. Selesnick and L. Duval. This C++ implementation is based on their MATLAB code at https://nl.mathworks.com/matlabcentral/fileexchange/49974-beads--baseline-estimation-and-denoising-w--sparsity--chromatogram-signals-.  
Ownership of the original BEADS function resides with the authors mentioned, this implementation will be removed at their request.  