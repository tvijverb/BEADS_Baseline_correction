#ifndef BEADSBASELINE_H
#define BEADSBASELINE_H
#define _USE_MATH_DEFINES
#define EIGEN_NO_DEBUG
#define EIGEN_NO_STATIC_ASSERT

#include <vector>
#include <iostream>
#include "./Eigen/Dense"
#include "./Eigen/Sparse"
#include "./Eigen/Orderingmethods"


// [x, f, cost] = beads(y, d, fc, r, lam0, lam1, lam2)
//
// Baseline estimation and denoising using sparsity (BEADS)
//
// INPUT
//   y: Noisy observation
//   d: Filter order (d = 1 or 2)
//   fc: Filter cut-off frequency (cycles/sample) (0 < fc < 0.5)
//   r: Asymmetry ratio
//   lam0, lam1, lam2: Regularization parameters
//
// OUTPUT
//   x: Estimated sparse-derivative signal
//   f: Estimated baseline
//   cost: Cost function history

// Reference:
// Chromatogram baseline estimation and denoising using sparsity (BEADS)
// Xiaoran Ning, Ivan W. Selesnick, Laurent Duval
// Chemometrics and Intelligent Laboratory Systems (2014)
// doi: 10.1016/j.chemolab.2014.09.014
// Available online 30 September 2014
// Ported to C++ by T. Vijverberg, Radboud University Nijmegen 24 December 2017

using namespace Eigen;

class beadsBaseline
{
public:
    beadsBaseline();

public:
    std::vector<std::vector<double>> beads(std::vector<double> y,int d,double fc,double positivityBias,double lam0,double lam1,double lam2);
	std::vector<Eigen::SparseMatrix<double>> BAfilt(int d, double fc, int N);

private:
    VectorXd H(std::vector<Eigen::SparseMatrix<double>>, Eigen::VectorXd x); // anonimous functions of beads
	Eigen::VectorXd wfun(Eigen::VectorXd, double);
    double theta(double, double);

private:
    int numIterations = 30;
    int eps0 = 1e-6;            // cost smoothing parameter for x (small positive value)
    int eps1 = 1e-6;            // cost smoothing parameter for derivatives (small positive value)
    int d = 1;                  // degree of filter
    double fc = 0.1;            // cutoff frequency 0 < fc < 0.5
    int N;                      // signal length
    int positivityBias = 6;     // asymmetry parameter
    double amp = 0.8;           // amplifier
    double lam0 = 0.5*amp;      // lambda regularization parameters
    double lam1 = 5*amp;
    double lam2 = 4*amp;

};

#endif // BEADSBASELINE_H
