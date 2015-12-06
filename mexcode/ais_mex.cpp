#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "mex.h"
#include "ais.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

using namespace Eigen;
using namespace std;

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

VectorXd matlabVec2eigenVec( const mxArray* mp ) {
    // Load with a certain size
    const mwSize* size = mxGetDimensions( mp );
    mwSize length;
    if(size[0] > 1) { length = size[0]; } else { length = size[1]; }
    VectorXd x(length);

    // Extract data
    double* data = mxGetPr( mp );
    for(size_t i = 0; i < length; i++) {
        x(i) = data[i];
    }

    return x;
}

mxArray* eigenVec2matlabVec( const VectorXd& x ) {
    mxArray* mp = mxCreateDoubleMatrix( x.size(), 1, mxREAL );
    double* data = mxGetPr( mp );
    // Copy data from vector
    for(size_t i = 0; i < x.size(); ++i) {
        data[i] = x(i);
    }
    return mp;
}

MatrixXd matlabMat2eigenMat( const mxArray* mp ) {
    // Load with a certain size
    const mwSize* size = mxGetDimensions( mp );
    MatrixXd A(size[0],size[1]);

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

SparseMatrix<double> matlabSparse2eigenSparse( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    mwSize nnz = mxGetNzmax( mp );

    // Setup triplets
    vector< Triplet< double > > triplets;
    triplets.reserve( nnz );

    // Extract nonzeros
    size_t *ir = mxGetIr( mp ); /* Row indexing */
    size_t *jc = mxGetJc( mp ); /* Column count */
    double *s = mxGetPr( mp ); /* Non-zero elements */
    for (int ci=0; ci < size[1]; ci++) { /* Loop through columns */
        for (int k = jc[ci]; k < jc[ci+1]; k++) { /* Loop through non-zeros in ith column */
            int ri = ir[k];
            double val = s[k];
            triplets.push_back( Triplet<double>( ri, ci, val ) );
        }
    }

    // Create matrix
    SparseMatrix<double> A(size[0],size[1]);
    A.setFromTriplets( triplets.begin(), triplets.end() );
    //cout << "Loaded sparse matrix:" << endl << A << endl;
    return A;
}

mxArray* eigenSparse2matlabSparse( SparseMatrix<double> A ) {
    mxArray* mp = mxCreateSparse( (mwSize)A.rows(), (mwSize)A.cols(), (mwSize)A.nonZeros(), mxREAL );
    double* sr = mxGetPr( mp );
    mwSize* irs = mxGetIr( mp );
    mwSize* jcs = mxGetJc( mp );

    // Loop through non-zeros
    size_t k = 0;
    for (size_t j = 0; j < A.outerSize(); ++j) {
        mwSize i;
        jcs[j] = k;
        for (SparseMatrix<double>::InnerIterator it(A,j); it; ++it) {
            sr[k] = it.value();
            irs[k] = it.row();
            k++;
        }
    }
    jcs[ A.cols() ] = k;

    return mp;
}

/***** Argument load functions *****/
VectorXd getVector( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( !((size[0] == 1 && size[1] > 1) || (size[0] > 1 && size[1] == 1)) ) { throw invalid_argument( "MATLAB argument is not a vector n x 1 or 1 x n" ); }   
    if( size[0] > 1 && size[1] > 1 ) { throw invalid_argument( "MATLAB argument is a matrix not a vector" ); }   
    if( mxIsSparse(mp) ) { throw invalid_argument( "MATLAB argument is sparse but should be dense" ); }   

    // Load vector
    VectorXd x = matlabVec2eigenVec( mp ); 
    //cout << "Loaded vector:" << endl << x << endl;
    return x;
}

MatrixXd getMatrix( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( size[0] <= 1 || size[1] <= 1 ) { throw invalid_argument( "MATLAB argument is a vector rather than a matrix." ); }   
    if( mxIsSparse(mp) ) { throw invalid_argument( "MATLAB argument is sparse but should be dense." ); }   

    // Load matrix 
    MatrixXd A = matlabMat2eigenMat( mp ); 
    //cout << "Loaded matrix:" << endl << A << endl;
    return A;
}

SparseMatrix< double > getSparseMatrix( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( size[0] <= 1 || size[1] <= 1 ) { throw invalid_argument( "MATLAB argument is a vector rather than a matrix." ); }   
    if( !mxIsSparse(mp) ) { throw invalid_argument( "MATLAB argument is dense but should be sparse." ); }   

    // Load matrix 
    SparseMatrix< double > A = matlabSparse2eigenSparse( mp ); 
    //cout << "Loaded matrix:" << endl << A << endl;
    return A;
}

template <typename T>
double getScalar( const mxArray* mp ) {
    const mwSize* size = mxGetDimensions( mp );
    // Error check
    if( size[0] != 1 || size[1] != 1 ) { throw invalid_argument( "MATLAB argument is not a scalar." ); }   
    if( mxIsSparse(mp) ) { throw invalid_argument( "MATLAB argument is sparse but should be dense" ); }   

    // Load matrix 
    double c = matlabScalar2scalar( mp ); 
    //cout << "Loaded scalar: " << c << endl;
    return (T) c;
}

/***** MATLAB mex function *****/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // ais_mex(thetaNode, thetaEdge, L, nSamples, betaVec, nGibbs, verbosity) 
    /*
    printf("\nThere are %d right-hand-side argument(s).\n", nrhs);
    for (int i=0; i<nrhs; i++)  {
        const mwSize* dim = mxGetDimensions(prhs[i]);
        printf("\tInput Arg %i is of type:\t%s size: %d x %d, Sparsity: %d\n", 
                i, mxGetClassName(prhs[i]), dim[0], dim[1], mxIsSparse(prhs[i]) );
    }
    */

    //mexPrintf("DEBUG: Before loading mex input");
    /* Verify and load arguments */
    VectorXd thetaNode, betaVec;
    SparseMatrix<double> thetaEdge;
    size_t L, nSamples, nGibbs, verbosity, nThreads;
    try {
        if( nrhs < 7 ) { throw invalid_argument( "Incorrect number of arguments." ); }
        thetaNode = getVector( prhs[0] );
        thetaEdge = getSparseMatrix( prhs[1] );
        L = getScalar<size_t>( prhs[2] );
        nSamples = getScalar<size_t>( prhs[3] );
        betaVec = getVector( prhs[4] );
        nGibbs = getScalar<size_t>( prhs[5] );
        verbosity = getScalar<size_t>( prhs[6] );
        if( nrhs >= 8) {
            nThreads = getScalar<size_t>( prhs[7] );
        } else {
            nThreads = 0;
        }
    } catch ( const std::invalid_argument& e ) {
        cerr << "Error loading arguments: " << e.what() << endl;
        fake_answer(plhs);
        return;
    }
    //mexPrintf("DEBUG: After loading mex input");

    /* Simply call ais function */
    //mexPrintf("DEBUG: Before calling ais function");
    VectorXd logW;
    SparseMatrix<double> X;
    ais( thetaNode, thetaEdge, L, nSamples, betaVec, nGibbs, nThreads, verbosity,
            logW, X );
    //mexPrintf("DEBUG: After calling ais function");

    /* Save final answer into MATLAB output */
    //mexPrintf("DEBUG: Before creating mex output arguments");
    if( nlhs >= 1 ) {
        // Sanity check test
        /*
        VectorXd temp(5);
        temp << 1,2,3,4,5;
        cout << "Vector temp:" << endl << temp << endl;
        plhs[0] = eigenVec2matlabVec( temp );
        */
        plhs[0] = eigenVec2matlabVec( logW );
    }
    if( nlhs >= 2 ) {
        // Sanity check test
        /*
        SparseMatrix<double> temp(4,3);
        std::vector< Triplet<double> > triplets;
        triplets.push_back( Triplet<double>(0,0,1) );
        triplets.push_back( Triplet<double>(0,1,1.5) );
        triplets.push_back( Triplet<double>(1,0,0.5) );
        triplets.push_back( Triplet<double>(1,1,2) );
        triplets.push_back( Triplet<double>(2,2,3) );
        temp.setFromTriplets( triplets.begin(), triplets.end() );
        cout << "Sparse matrix temp:" << endl << temp << endl;
        plhs[1] = eigenSparse2matlabSparse( temp );
        */
        plhs[1] = eigenSparse2matlabSparse( X );
    }
    //mexPrintf("DEBUG: After creating mex output arguments");
}

/*
mxArray* kron_perm_dense(const mxArray *mxX, size_t block_m, size_t block_n, size_t *row_perm, size_t *col_perm) {
	size_t rows = mxGetM(mxX), cols = mxGetN(mxX), size = rows*cols;
	double *X = mxGetPr(mxX);
	kmf_rk1_t rk1; 
	rk1.fill_attributes(rows, cols, block_m, block_n, row_perm, col_perm);
	size_t new_rows = rk1.Asize(), new_cols = rk1.Bsize(), new_size = new_rows*new_cols;
	printf("%lu x %lu -> %lu x %lu\n", rows, cols, new_rows, new_cols);
	mxArray *mxkpX = mxCreateDoubleMatrix(new_rows, new_cols, mxREAL);
	double *kpX = mxGetPr(mxkpX);

	for(size_t new_idx = 0; new_idx < new_size; new_idx++)
		kpX[new_idx] = 0;
	for(size_t c = 0, idx = 0; c < cols; c++) {
		for(size_t r = 0; r < rows; r++) {
			size_t i=0, j=0;
			rk1.get_indices(r,c,i,j);
			kpX[j*new_rows+i] = X[idx++];
		}
	}
	return mxkpX;
}
mxArray* kron_perm_sparse(const mxArray *X, size_t block_m, size_t block_n, size_t *row_perm, size_t *col_perm) {
    // ir pointer to row indexes, jc pointer to column starts
	mwIndex *ir = mxGetIr(X), *jc = mxGetJc(X);
	double *val = mxGetPr(X);
	size_t rows = mxGetM(X), cols = mxGetN(X), nnz = jc[cols]; 

	kmf_rk1_t rk1; 
	rk1.fill_attributes(rows, cols, block_m, block_n, row_perm, col_perm);
	size_t new_rows = rk1.Asize(), new_cols = rk1.Bsize();
	mxArray *kpX = mxCreateSparse(new_rows, new_cols, nnz, mxREAL);
	mwIndex *new_ir = mxGetIr(kpX), *new_jc = mxGetJc(kpX);
	double *new_val = mxGetPr(kpX);

    // Set all new column pointers to 0
	for(size_t j = 0; j <= new_cols; j++) 
		new_jc[j] = 0;
    // Find sizes of new columns
	for(size_t c = 0; c < cols; c++) {
		for(mwIndex idx = jc[c]; idx != jc[c+1]; idx++) {
			size_t r = ir[idx];
			size_t i, j;
			rk1.get_indices(r,c,i,j);
			new_jc[j+1]++;
		}
	}
    // Make the column pointers be indexes instead of counts
	for(size_t j = 1; j <= new_cols; j++)
		new_jc[j] += new_jc[j-1];
    // Now find update new values 
    // Essentially fill in the columns and update the for different j columns
    // Looping through in original order
	for(size_t c = 0; c < cols; c++) {
		for(mwIndex idx = jc[c]; idx != jc[c+1]; idx++) {
			size_t r = ir[idx];
			size_t i, j;
			rk1.get_indices(r,c,i,j);
			new_ir[new_jc[j]] = i;
			new_val[new_jc[j]] = val[idx];
			new_jc[j]++;
		}
	}
    // Reset column pointers
	for(size_t j = new_cols; j > 0; j--)
		new_jc[j] = new_jc[j-1];
	new_jc[0] = 0;
    // Sort column values based on row indices
	for(size_t j = 0; j < new_cols; j++)
		sort(zip_iter(&new_ir[new_jc[j]], &new_val[new_jc[j]]),
				zip_iter(&new_ir[new_jc[j+1]], &new_val[new_jc[j+1]]));
	return kpX;
}
*/
