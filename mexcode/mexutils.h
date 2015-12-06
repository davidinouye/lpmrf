#ifndef _MEXUTILS_H // Include guard
#define _MEXUTILS_H
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "mex.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

void exit_with_help()
{
	mexPrintf(
	"Usage: [kpX] = kron_perm(X, block_size)\n"
	"       [kpX] = kron_perm(X, block_size, row_perm, col_perm)\n"
	"     X is an m-by-n sparse/dense double matrix\n"
	"     block_size is a length-2 array [block_m block_n] for the size of the last block\n"
	"     row_perm is an dense row permutation array\n"
	"     col_perm is an dense column permutation array\n"
	"     kpX is an M-by-N sparse double matrix\n"
	"        M = ceil(m/block_m)*ceil(n/block_n)\n"
	"        N = block_m*block_n\n"
	"Note: \n"
	"     kpX == kron_perm(X(row_perm, col_perm), block_size)\n"
	);
}

static void fake_answer(mxArray *plhs[]) { plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL); }

/***** MATLAB 2 EIGEN conversion *****/
double matlabScalar2scalar( const mxArray* mp ) {
    double scalar = mxGetScalar(mp);
    return scalar;
}

Eigen::VectorXd matlabVec2eigenVec( const mxArray* mp ) {
    // Load with a certain size
    const mwSize* size = mxGetDimensions( mp );
    mwSize length;
    if(size[0] > 1) { length = size[0]; } else { length = size[1]; }
    Eigen::VectorXd x(length);

    // Extract data
    double* data = mxGetPr( mp );
    for(size_t i = 0; i < length; i++) {
        x(i) = data[i];
    }

    return x;
}

mxArray* eigenVec2matlabVec( const Eigen::VectorXd& x ) {
    mxArray* mp = mxCreateDoubleMatrix( x.size(), 1, mxREAL );
    double* data = mxGetPr( mp );
    // Copy data from vector
    for(size_t i = 0; i < x.size(); ++i) {
        data[i] = x(i);
    }
    return mp;
}

Eigen::MatrixXd matlabMat2eigenMat( const mxArray* mp ) {
    // Load with a certain size
    const mwSize* size = mxGetDimensions( mp );
    Eigen::MatrixXd A(size[0],size[1]);

    // Extract data
    double* data = mxGetPr( mp );
    size_t idx = 0;
    for(int j = 0; j < size[1]; j++) {
        for(int i = 0; i < size[0]; i++) {
            A(i,j) = data[ idx++ ];
        }
    }

    return A;
}

Eigen::SparseMatrix<double> matlabSparse2eigenSparse( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    mwSize nnz = mxGetNzmax( mp );

    // Setup triplets
    std::vector< Eigen::Triplet< double > > triplets;
    triplets.reserve( nnz );

    // Extract nonzeros
    size_t *ir = mxGetIr( mp ); /* Row indexing */
    size_t *jc = mxGetJc( mp ); /* Column count */
    double *s = mxGetPr( mp ); /* Non-zero elements */
    for (int ci=0; ci < size[1]; ci++) { /* Loop through columns */
        for (int k = jc[ci]; k < jc[ci+1]; k++) { /* Loop through non-zeros in ith column */
            int ri = ir[k];
            double val = s[k];
            triplets.push_back( Eigen::Triplet<double>( ri, ci, val ) );
        }
    }

    // Create matrix
    Eigen::SparseMatrix<double> A(size[0],size[1]);
    A.setFromTriplets( triplets.begin(), triplets.end() );
    //cout << "Loaded sparse matrix:" << endl << A << endl;
    return A;
}

mxArray* eigenSparse2matlabSparse( Eigen::SparseMatrix<double> A ) {
    mxArray* mp = mxCreateSparse( (mwSize)A.rows(), (mwSize)A.cols(), (mwSize)A.nonZeros(), mxREAL );
    double* sr = mxGetPr( mp );
    mwSize* irs = mxGetIr( mp );
    mwSize* jcs = mxGetJc( mp );

    // Loop through non-zeros
    size_t k = 0;
    for (size_t j = 0; j < A.outerSize(); ++j) {
        mwSize i;
        jcs[j] = k;
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,j); it; ++it) {
            sr[k] = it.value();
            irs[k] = it.row();
            k++;
        }
    }
    jcs[ A.cols() ] = k;

    return mp;
}

/***** Argument load functions *****/
Eigen::VectorXd getVector( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( size[0] > 1 && size[1] > 1 ) { throw std::invalid_argument( "MATLAB argument is a matrix not a vector" ); }   
    if( mxIsSparse(mp) ) { throw std::invalid_argument( "MATLAB argument is sparse but should be dense" ); }   

    // Load vector
    Eigen::VectorXd x = matlabVec2eigenVec( mp ); 
    //cout << "Loaded vector:" << endl << x << endl;
    return x;
}

Eigen::MatrixXd getMatrix( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( mxIsSparse(mp) ) { throw std::invalid_argument( "MATLAB argument is sparse but should be dense." ); }   

    // Load matrix 
    Eigen::MatrixXd A = matlabMat2eigenMat( mp ); 
    //cout << "Loaded matrix:" << endl << A << endl;
    return A;
}

Eigen::SparseMatrix<double> getSparseMatrix( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( size[0] <= 1 || size[1] <= 1 ) { throw std::invalid_argument( "MATLAB argument is a vector rather than a matrix." ); }   
    if( !mxIsSparse(mp) ) { throw std::invalid_argument( "MATLAB argument is dense but should be sparse." ); }   

    // Load matrix 
    Eigen::SparseMatrix<double> A = matlabSparse2eigenSparse( mp ); 
    //cout << "Loaded matrix:" << endl << A << endl;
    return A;
}

template <typename T>
T getScalar( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( size[0] != 1 || size[1] != 1 ) { throw std::invalid_argument( "MATLAB argument is not a scalar." ); }   
    if( mxIsSparse(mp) ) { throw std::invalid_argument( "MATLAB argument is sparse but should be dense" ); }   

    // Load matrix 
    double c = matlabScalar2scalar( mp ); 
    //cout << "Loaded scalar: " << c << endl;
    return (T) c;
}

#endif
