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
    std::vector< Eigen::SparseMatrix<double> > XArray;
    Eigen::MatrixXd logPMat;
    Eigen::MatrixXd modLMat;
    std::vector< Eigen::VectorXd > thetaNodeArray;
    std::vector< Eigen::SparseMatrix<double> > thetaEdgeArray;
    std::vector< Eigen::VectorXd > logPArray;
    std::vector< Eigen::VectorXd > modLArray;
    try {
        if( nrhs < 4 ) { throw std::invalid_argument( "Incorrect number of arguments." ); }
        /* Load XArray */
        mwSize k = mxGetNumberOfElements( prhs[0] );
        for(mwIndex j = 0; j < k; ++j) {
            Eigen::SparseMatrix<double> X_j = getSparseMatrix( mxGetCell(prhs[0], j) );
            XArray.push_back( X_j );
        }

        /* Extract thetaNode and thetaEdge */
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

    /* Load ZtArray from XArray */
    std::vector< Eigen::SparseMatrix<double> > ZtArray;
    size_t k = XArray.size();
    size_t p = XArray[0].rows();
    size_t n = XArray[0].cols();
    for(size_t i = 0; i < n; ++i) {
        // Loop over inner iterator to extract non-zeros in columns from XArray
        std::vector< Eigen::Triplet<double> > triplets;
        for(size_t j = 0; j < k; ++j) {
            for(Eigen::SparseMatrix<double>::InnerIterator it(XArray[j], i); it; ++it) {
                triplets.push_back( Eigen::Triplet<double>( j, it.row(), it.value() ) );
            }
        }

        // Construct Zt from new triplet set
        Eigen::SparseMatrix<double> Zt(k, p);
        Zt.setFromTriplets( triplets.begin(), triplets.end() );
        ZtArray.push_back(Zt);
    }

    /* Load arguments into LPMRF models */
    typedef LPMRF< SparseMatrix<double> > LPMRFSparse;
    std::vector< LPMRFSparse > modelArray;
    for(size_t j = 0; j < thetaNodeArray.size(); ++j) {
        modelArray.push_back( LPMRFSparse(thetaNodeArray[j], thetaEdgeArray[j], logPArray[j], modLArray[j] ) );
    }

    /* Simply call C++ function */
    learnalltopicmats( modelArray, 0, ZtArray );

    /* Save final answer into MATLAB output */
    if( nlhs >= 1 ) {
        // Extract XArray contents from ZtArray 
        std::vector< std::vector< Eigen::Triplet<double> > > tripletArray;
        for(size_t j = 0; j < k; ++j) {
            tripletArray.push_back( std::vector< Eigen::Triplet<double> >() );
        }
        for(size_t i = 0; i < n; ++i) {
            for(size_t s = 0; s < ZtArray[i].outerSize(); ++s) {
                for(Eigen::SparseMatrix<double>::InnerIterator it(ZtArray[i], s); it; ++it) {
                    size_t j = it.row();
                    //std::cout << "(row=" << it.row() << ",col=" << it.col() << ") val= " << it.value() << std::endl;
                    //std::cout << "Adding (s=" << s << ",i=" << i << ") val= " << it.value() << " to topic " << j << std::endl;
                    Eigen::Triplet<double>( s, i, it.value() );
                    tripletArray[j].push_back( Eigen::Triplet<double>(s,i,it.value()) );
                }
            }
        }

        // Create MATLAB output cell array
        //std::cout << "Saving output..." << std::endl;
        plhs[0] = mxCreateCellMatrix((mwSize)k,1);
        for(size_t j = 0; j < k; ++j) {
            //std::cout << "Saving matrix " << j << std::endl;
            Eigen::SparseMatrix<double> X_j(p,n);
            X_j.setFromTriplets( tripletArray[j].begin(), tripletArray[j].end() );
            //std::cout << "Setting cell for matrix " << j << std::endl;
            mxSetCell(plhs[0], j, eigenSparse2matlabSparse( X_j ) );
        }
    }
}
