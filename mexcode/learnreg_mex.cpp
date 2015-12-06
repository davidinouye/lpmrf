#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "learnreg.h"
#include "mexutils.h"

using namespace Eigen;

/***** MATLAB mex function *****/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    //mexPrintf("DEBUG: Before loading mex input");
    /* Verify and load arguments */
    Eigen::SparseMatrix<double> X;
    Eigen::VectorXd Y;
    double lam, nodeBeta = 0;
    Eigen::VectorXd phi;
    try {
        if( nrhs < 3 ) { throw std::invalid_argument( "Too few arguments.  Need at least 3." ); }
        X = getSparseMatrix( prhs[0] );
        Y = getVector( prhs[1] );
        lam = getScalar<double>( prhs[2] );
        if(nrhs >= 4) {
            nodeBeta = getScalar<double>( prhs[3] );
        }
        if(nrhs >= 5) {
            phi = getVector( prhs[4] );
        }
    } catch ( const std::invalid_argument& e ) {
        std::cerr << "Error loading arguments: " << e.what() << std::endl;
        fake_answer(plhs);
        return;
    }
    mexPrintf("DEBUG: After loading mex input, nodeBeta = %g", nodeBeta);

    /* Simply call ais function */
    //mexPrintf("DEBUG: Before calling ais function");
    GeneralizedLinearModel<Poisson> poisson;
    poisson.learnreg( X, Y, lam, nodeBeta, phi );
    //mexPrintf("DEBUG: After calling ais function");

    /* Save final answer into MATLAB output */
    //mexPrintf("DEBUG: Before creating mex output arguments");
    if( nlhs >= 1 ) {
        plhs[0] = eigenVec2matlabVec( phi );
    }
    //mexPrintf("DEBUG: After creating mex output arguments");
}
