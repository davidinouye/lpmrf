#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "learnreg.h"
#include "mexutils.h"

using namespace Eigen;

/***** MATLAB mex function *****/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Verify and load arguments
    Eigen::SparseMatrix<double> X;
    VectorXd lamVec;
    double nodeBeta = 0;
    VectorXd thetaNode;
    SparseMatrix<double> thetaEdge;
    size_t nThreads = 0;
    try {
        if( nrhs < 2 ) { throw std::invalid_argument( "Too few arguments.  Need at least 2." ); }
        X = getSparseMatrix( prhs[0] );
        lamVec = getVector( prhs[1] );
        if(nrhs >= 3)
            nodeBeta = getScalar<double>( prhs[2] );
        if(nrhs >= 4)
            nThreads = getScalar<size_t>( prhs[3] );
        if(nrhs >= 5)
            thetaNode = getVector( prhs[4] );
        if(nrhs >= 6)
            thetaEdge = getSparseMatrix( prhs[5] );
    } catch ( const std::invalid_argument& e ) {
        std::cerr << "Error loading arguments: " << e.what() << std::endl;
        fake_answer(plhs);
        return;
    }

    // Run code
    GeneralizedMRF<Poisson> pmrf(2, nThreads);
    std::vector< VectorXd > thetaNodeArray;
    std::vector< SparseMatrix<double> > thetaEdgeArray;
    for(size_t li = 0; li < lamVec.size(); ++li) {
        // Run MRF
        pmrf.learnmrf(X, lamVec(li), nodeBeta, thetaNode, thetaEdge); 
        // Extract results
        thetaNodeArray.push_back( thetaNode );
        thetaEdgeArray.push_back( thetaEdge );
    }

    // Save final answer into MATLAB output */
    if( nlhs >= 1 ) {
        if(lamVec.size() == 1) {
            plhs[0] = eigenVec2matlabVec( thetaNodeArray[0] );
        } else {
            // Save cell array of outputs
            plhs[0] = mxCreateCellMatrix((mwSize)lamVec.size(),1);
            for(size_t li = 0; li < lamVec.size(); ++li)
                mxSetCell(plhs[0], li, eigenVec2matlabVec( thetaNodeArray[li] ) );
        }
    }
    if( nlhs >= 2 ) {
        if(lamVec.size() == 1) {
            plhs[1] = eigenSparse2matlabSparse( thetaEdgeArray[0] );
        } else {
            // Save cell array of outputs
            plhs[1] = mxCreateCellMatrix((mwSize)lamVec.size(),1);
            for(size_t li = 0; li < lamVec.size(); ++li)
                mxSetCell(plhs[1], li, eigenSparse2matlabSparse( thetaEdgeArray[li] ) );
        }
    }
}
