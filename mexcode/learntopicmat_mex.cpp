#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "learntopicmat.h"
#include "mexutils.h"

using namespace Eigen;

/***** MATLAB mex function *****/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Verify and load arguments */
    Eigen::SparseMatrix<double> Zt;
    Eigen::MatrixXd logPMat;
    Eigen::MatrixXd modLMat;
    std::vector< Eigen::VectorXd > thetaNodeArray;
    std::vector< Eigen::SparseMatrix<double> > thetaEdgeArray;
    std::vector< Eigen::VectorXd > logPArray;
    std::vector< Eigen::VectorXd > modLArray;
    try {
        if( nrhs < 4 ) { throw std::invalid_argument( "Incorrect number of arguments." ); }
        Zt = getSparseMatrix( prhs[0] );

        /* Extract thetaNode and thetaEdge */
        mwSize k = mxGetNumberOfElements( prhs[1] );
        for(mwIndex j = 0; j < k; ++j) {
            mxArray* model = mxGetCell(prhs[1], j);
            Eigen::VectorXd thetaNode = getVector(mxGetField( model, 0, "thetaNode" ));
            thetaNodeArray.push_back( thetaNode );

            Eigen::SparseMatrix<double> thetaEdge = getSparseMatrix(mxGetField( model, 0, "thetaEdge" ));
            thetaEdgeArray.push_back( thetaEdge );
        }

        logPMat = getMatrix( prhs[2] );
        modLMat = getMatrix( prhs[3] );
        for(size_t j = 0; j < thetaNodeArray.size(); ++j) {
            Eigen::VectorXd logPj = logPMat.row(j).transpose();
            Eigen::VectorXd modLj = modLMat.row(j).transpose();
            logPArray.push_back(logPj);
            modLArray.push_back(modLj);
        }

    } catch ( const std::invalid_argument& e ) {
        std::cerr << "Error loading arguments: " << e.what() << std::endl;
        fake_answer(plhs);
        return;
    }
    /* Load arguments into LPMRF models */
    typedef LPMRF< SparseMatrix<double> > LPMRFSparse;
    std::vector< LPMRFSparse > modelArray;
    for(size_t j = 0; j < thetaNodeArray.size(); ++j) {
        modelArray.push_back( LPMRFSparse(thetaNodeArray[j], thetaEdgeArray[j], logPArray[j], modLArray[j] ) );
    }

    /* Simply call C++ function */
    //std::cout << "Zt Before:" << std::endl << Zt << std::endl;
    learntopicmat( modelArray, 0, Zt );
    //std::cout << "Zt After:" << std::endl << Zt << std::endl;

    /* Save final answer into MATLAB output */
    if( nlhs >= 1 ) {
        plhs[0] = eigenSparse2matlabSparse( Zt );
    }
}
