#ifndef _AIS_H // Include guard
#define _AIS_H

#include <iostream> // cout,cerr,etc.
#include <stdio.h>  // printf, etc.
#include <stdexcept> // Standard exceptions

//#include <omp.h> // Parallel
//#include <cstring>
//#include <string>

// Eigen
#include <Eigen/Dense> 
#include <Eigen/SparseCore>

void ais( const Eigen::VectorXd& thetaNode, 
        const Eigen::SparseMatrix<double>& thetaEdge, 
        const size_t L, 
        const size_t nSamples, 
        const Eigen::VectorXd& betaVec, 
        const size_t nGibbs, 
        const size_t nThreads, 
        const size_t verbosity, 
        Eigen::VectorXd& logW, 
        Eigen::SparseMatrix<double>& X );

#endif
