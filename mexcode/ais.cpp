#ifndef _AIS_CPP // Include guard
#define _AIS_CPP
#include <iostream>
#include <string>
#include <iterator> // For iterating over vector
#include <stdarg.h>  // For va_start, etc.
#include <cmath> // Basic math operations
#include <random>
#include <omp.h>
#include "ais.h"

using namespace Eigen;
using namespace std;

// Simple output operator for vectors
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

// Base code from  http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf   
void message(const double level, const double verbosity, const std::string fmt, ...) {
    if( level > verbosity ) { return; } // Skip if higher than verbosity
    int size = ((int)fmt.size()) * 2 + 50;
    std::string str;
    va_list ap;
    while (1) {     // Maximum two passes on a POSIX system...
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char *)str.data(), size, fmt.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {  // Everything worked
            str.resize(n);
            break;
        }
        if (n > -1)  // Needed size returned
            size = n + 1;   // For null char
        else
            size *= 2;      // Guess at a larger size (OS specific)
    }

    cout << str << endl;
}


struct FPlusSampler {
    public:
    size_t p;
    size_t T;
    std::vector<double> nodeVec;
    std::vector<double> logVec;
    std::mt19937 generator;
    std::uniform_real_distribution<double> uniform;

    // Constructor
    FPlusSampler( size_t seed ): p(0), T(0), nodeVec(), generator(seed), uniform( 0.0, 1.0 ) {}

    void init( const VectorXd& initVec, const VectorXd& logInitVec ) {
        p = initVec.size();
        T = pow( 2, ceil(log2( p )) );
        nodeVec.resize( 2*T + 1 ); // NOTE: 0 doesn't hold anything
        logVec.resize(p);

        // Initialize values from vector
        for(size_t i = 0; i < initVec.size(); i++) {
            nodeVec[T+i] = initVec(i);
            logVec[i] = logInitVec(i);
        }

        // Update the rest of the vector
        for(size_t i = T-1; i >= 1; i--) {
            nodeVec[i] = nodeVec[ left(i) ] + nodeVec[ right(i) ];
        }
    }

    size_t sample( double u ) const {
        size_t i = 1;
        while (i < T) {
            if( u > nodeVec[ left(i) ] ) {
                u = u - nodeVec[ left(i) ]; 
                i = right(i);
            } else {
                i = left(i);
            }
        }
        size_t z = i - T + 1;
        return z;
    }

    size_t sample() {
        double u = totalSum()*uniform(generator);
        return sample( u );
    }


    private:
    void update( const size_t t, const double delta ) {
        size_t i = leaf( t );
        // Update tree sums in log( T ) time
        while ( i >= 1 ) {
            nodeVec[ i ] += delta;
            i = parent(i);
        }
    }

    public:
    void updateScale( const size_t t, const double scale, const double logScale ) {
        double val = nodeVec[ leaf( t ) ];
        double delta = (scale-1)*val;
        update( t, delta );
        logVec[ t-1 ] += logScale;
    }

    void updateScaleMat( const SparseMatrix<double>& scaleMat, 
            const SparseMatrix<double>& logScaleMat, 
            double logMult, const size_t r ) {
        /* Loop through both the scale and logScale sparse matrices */
        SparseMatrix<double>::InnerIterator itLog( logScaleMat, r-1 );
        for( SparseMatrix<double>::InnerIterator it( scaleMat, r-1 ); it; ++it) {
            updateScale( it.row()+1, it.value(), logMult * itLog.value() );
            ++itLog;
        }
    }

    double getLogLeaf( const size_t t ) const {
        return logVec[ t - 1 ];
    }

    // Accessor functions
    double totalSum( ) const { return nodeVec[1]; }
    size_t left( size_t i ) const { return i << 1; }
    size_t right( size_t i ) const { return (i << 1) + 1; }
    size_t parent( size_t i ) const { return i >> 1; }
    size_t leaf( size_t t ) const { return T + t - 1; }

};

VectorXd seq2vec( const VectorXi& xSeq, const size_t p ) {
    VectorXd x = VectorXd::Zero(p);
    for(size_t si = 0; si < xSeq.size(); ++si) {
        x(xSeq(si)-1)++;
    }
    return x;
}

inline void sampleMultSeq( FPlusSampler& fplus, const double L, VectorXi& xSeq ) {
    if( xSeq.size() != L ) { xSeq.resize(L); }
    for(size_t i = 0; i < L; i++) {
        xSeq(i) = fplus.sample();
    }
}

inline void gibbsStep( const double beta, 
        const VectorXd& thetaNode,
        const SparseMatrix<double>& thetaEdge,
        const SparseMatrix<double>& scaleMat,
        const SparseMatrix<double>& scaleNegMat,
        VectorXi& xSeq,
        FPlusSampler& fplus ) {
    // Loop through sequence and resample sequence updating fplus as necessary
    size_t rOld, rNew;
    double twoBeta = 2*beta;
    for(size_t si = 0; si < xSeq.size(); si++) {
        // Pop off top and update fplus
        rOld = xSeq(si);
        fplus.updateScaleMat( scaleNegMat, thetaEdge, -twoBeta, rOld );

        // Sample
        rNew = fplus.sample();

        // Push new back on and update fplus
        xSeq(si) = rNew;
        fplus.updateScaleMat( scaleMat, thetaEdge, twoBeta, rNew );
    }
}

inline void updateTriplets( const VectorXi& xSeq, const size_t i, vector< Triplet< double > >& Xtriplets ) {
    for(size_t si = 0; si < xSeq.size() ; si++) {
        Xtriplets.push_back( Triplet<double>( xSeq(si)-1, i, 1 ) );
    }
}

inline void resetFPlusSampler(const SparseMatrix<double>& scaleMat, const SparseMatrix<double>& thetaEdge, const VectorXi& xSeq, const double betaNext, VectorXd& initVec, VectorXd& logInitVec, FPlusSampler& fplus ) {
    size_t c;
    for(size_t si = 0; si < xSeq.size(); si++ ) {
        c = xSeq(si)-1;
        SparseMatrix<double>::InnerIterator logIt(thetaEdge, c);
        for (SparseMatrix<double>::InnerIterator it(scaleMat, c); it; ++it) {
            initVec(it.row()) *= it.value();
            logInitVec(it.row()) +=  2*betaNext*logIt.value();
            ++logIt;
        }
    }
    fplus.init( initVec, logInitVec );
}

inline void singleSample(const size_t i,
        const VectorXd& thetaNode,
        const SparseMatrix<double>& thetaEdge,
        const size_t L,
        const size_t nGibbs,
        const VectorXd& betaVec,
        const VectorXd& expThetaNode,
        const SparseMatrix<double>* scaleMatArray,
        const SparseMatrix<double>* scaleNegMatArray,
        const size_t vb,
        double& logW,
        VectorXi& xSeq
        ) {
    const size_t p = thetaNode.size();
    FPlusSampler fplus(i);
    logW = 0;
    // Initialize
    VectorXd initVec(p), logInitVec(p);
    //cout << "i   : " << i << endl;
    fplus.init( expThetaNode, thetaNode );
    // NOTE: Only upto size-1 rather than size
    for( size_t bi = 0; bi < betaVec.size()-1; bi++) {
        double beta = betaVec(bi);
        double betaNext = betaVec(bi+1);

        if( betaVec(bi) == 0 ) {
            // Sample from Multinomial
            sampleMultSeq( fplus, L, xSeq );
        } else {
            for(size_t gi = 0; gi < nGibbs; gi++) {
                gibbsStep( beta, thetaNode, thetaEdge, scaleMatArray[bi], scaleNegMatArray[bi], xSeq, fplus );
            }
            
            // Calculate logW update
            double gamma = 0.5*(betaNext/beta - 1);
            double sum = 0;
            for(size_t si = 0; si < xSeq.size(); si++) {
                sum += fplus.getLogLeaf( xSeq(si) ) - thetaNode( xSeq(si)-1 );
            }
            double update = gamma*sum;
            logW += update;
        }

        // Update fplus if needed
        if( bi < betaVec.size()-2 ) {
            initVec = expThetaNode;
            logInitVec = thetaNode;
            resetFPlusSampler( scaleMatArray[bi+1], thetaEdge, xSeq, betaNext, initVec, logInitVec, fplus );
        }
    }
}

void ais( const VectorXd& thetaNode, 
        const SparseMatrix<double>& thetaEdge, 
        const size_t L, 
        const size_t nSamples, 
        const VectorXd& betaVec, 
        const size_t nGibbs, 
        size_t nThreads, 
        const size_t verbosity, 
        VectorXd& logW, 
        SparseMatrix<double>& X ) {

    // Init
    const size_t vb = verbosity; // Alias verbosity
    const size_t p = thetaNode.size();
    const size_t nBeta = betaVec.size();

    // Precompute some values
    SparseMatrix<double> scaleMatArray[nBeta];
    SparseMatrix<double> scaleNegMatArray[nBeta];
    for(size_t bi = 0; bi < nBeta; ++bi) {
        double beta = betaVec(bi);
        scaleMatArray[bi] = thetaEdge;
        scaleNegMatArray[bi] = thetaEdge;
        for (size_t c = 0; c < thetaEdge.outerSize(); c++) {
            for (SparseMatrix<double>::InnerIterator it(scaleMatArray[bi],c); it; ++it) {
                it.valueRef() = exp( (2*beta) * it.value() );
            }
            for (SparseMatrix<double>::InnerIterator it(scaleNegMatArray[bi],c); it; ++it) {
                it.valueRef() = exp( (-2*beta) * it.value() );
            }
        }
    }
    VectorXd expThetaNode = thetaNode.array().exp();
    
    // Initialize loop variables
    logW = VectorXd::Zero(nSamples);
    std::vector< VectorXi > xSeqArray;
    xSeqArray.reserve(nSamples);
    for(size_t i = 0; i < nSamples; i++ ) {
        xSeqArray.push_back( VectorXi(L) );
    }
    if( nThreads == 0 ) {
        nThreads = (size_t)omp_get_num_procs();
    }
    omp_set_num_threads( nThreads );
    message( 2, vb, "Sampling %d AIS samples with %d threads...", nSamples, nThreads );

    // Main parallel loop
    int threadNums[nSamples];
    #pragma omp parallel for
    for(size_t i = 0; i < nSamples; i++ ) {
        singleSample( i, thetaNode, thetaEdge, L, nGibbs, betaVec, expThetaNode, scaleMatArray, scaleNegMatArray, vb, 
                logW(i), xSeqArray[i] );
        threadNums[i] = omp_get_thread_num();

    }

    /*
    for(size_t i = 0; i < nSamples; ++i) {
        std::cout << "i = " << i << ", thread num = " << threadNums[i] << std::endl; 
    }
    */
    
    // Setup triplets outside of parallel section
    vector< Triplet< double > > Xtriplets;
    Xtriplets.reserve( L*nSamples );
    for(size_t i = 0; i < nSamples; i++ ) {
        updateTriplets( xSeqArray[i], i, Xtriplets );
    }

    message( 2, vb, "Finished AIS sampling" );

    // Load samples
    message( 2, vb, "Loading samples into final matrix" );
    X.resize( p, nSamples );
    X.setFromTriplets( Xtriplets.begin(), Xtriplets.end() );
}

// Simple test function
int main(int argc, char **argv) {
    cout << "Hello World!" << endl;
    size_t p = 3;
    size_t L = 100;
    size_t nSamples = 1e4;
    size_t nGibbs = 1;
    size_t nThreads = 0;
    if(argc >= 2) { nThreads = atoi(argv[1]); }
    VectorXd thetaNode(p);
    thetaNode << -3,-2,-1;
    SparseMatrix<double> thetaEdge(p,p);
    std::vector< Triplet< double > > triplets;
    triplets.reserve(4);
    triplets.push_back( Triplet<double>(0,1,1) );
    triplets.push_back( Triplet<double>(1,0,1) );
    triplets.push_back( Triplet<double>(1,2,-1) );
    triplets.push_back( Triplet<double>(2,1,-1) );
    thetaEdge.setFromTriplets( triplets.begin(), triplets.end() );

    VectorXd betaVec(5);
    betaVec << 0, 0.25, 0.5, 0.75, 1;
    VectorXd logW;
    SparseMatrix<double> X;
    
    cout << "<<< Running ais >>>" << endl;
    ais( thetaNode, thetaEdge, L, nSamples, betaVec, nGibbs, nThreads, 100, logW, X );
    cout << "<<< Finished running ais >>>" << endl;

    return 0;

    VectorXd initVec(3);
    initVec << 3,2,1;
    VectorXd logInitVec(3);
    logInitVec << -3,-2,-1;

    // Test initalizer
    /*
    {
        FPlusSampler fplus( initVec ); 
        double vv[] = { 0, 6, 5, 1, 3, 2, 1, 0, 0 };
        std::vector<double> expected( begin(vv), end(vv) );
        cout << "<<< Initialization test >>>" << endl;
        cout << "Actual:   " << fplus.nodeVec << endl;
        cout << "Expected: " << expected << endl;
    }

    // Test update
    for( double delta = -1; delta <= 1; delta++) {
        for( size_t t = 1; t < 3; t++ ) {
            cout << "<<< Update test t=" << t << " delta=" << delta << " >>>" << endl;

            FPlusSampler fplus( initVec ); 
            fplus.updateScale( t, delta );
            cout << "Actual:   " << fplus.nodeVec << endl;

            if(t == 1) {
                double vv[] = {0, 6+delta, 5+delta, 1, 3+delta,2,1,0, 0};
                std::vector<double> expected( begin(vv), end(vv) );
                cout << "Expected: " << expected << endl;
            } else if(t==2) {
                double vv[] = {0, 6+delta, 5+delta, 1, 3,2+delta,1,0,0};
                std::vector<double> expected( begin(vv), end(vv) );
                cout << "Expected: " << expected << endl;
            } else {
                double vv[] = {0, 6+delta, 5, 1+delta, 3,2,1+delta,0, 0};
                std::vector<double> expected( begin(vv), end(vv) );
                cout << "Expected: " << expected << endl;
            }
        }
    }

    // Test update
    for( double delta = -1; delta <= 1; delta++) {
        for( size_t t = 1; t < 3; t++ ) {
            cout << "<<< Different update test t=" << t << " delta=" << delta << " >>>" << endl;

            FPlusSampler fplus( initVec ); 
            fplus.update( t, delta );

            VectorXd updated = initVec;
            updated[t-1] += delta;

            FPlusSampler fplusExpected( updated ); 
            cout << "Actual:   " << fplus.nodeVec << endl;
            cout << "Expected: " << fplusExpected.nodeVec << endl;
        }
    }

    // Test log update
    for( double delta = -1; delta <= 1; delta++) {
        for( size_t t = 1; t < 3; t++ ) {
            cout << "<<< Log update test t=" << t << " delta=" << delta << " >>>" << endl;

            FPlusSampler fplus( logInitVec.array().exp() ); 
            fplus.updateScale( t, exp(delta) );

            VectorXd updated;
            updated = logInitVec;
            updated[t-1] += delta;

            FPlusSampler fplusExpected( updated.array().exp() ); 
            cout << "Actual:   " << fplus.nodeVec << endl;
            cout << "Expected: " << fplusExpected.nodeVec << endl;
        }
    }

    // Test sampler
    {
        FPlusSampler fplus( initVec );
        for( double u = 0; u <= 6; u += 0.5 ) {
            size_t z = fplus.sample( u );
            size_t zExpected = 0;
            if( u <= 3) {
                zExpected = 1;
            } else if( u <= 5 ) {
                zExpected = 2;
            } else {
                zExpected = 3;
            }
            cout << "<<< Testing sampler u=" << u << " >>>" << endl;
            cout << "Actual z:   " << z << endl;
            cout << "Expected z: " << zExpected << endl;
        }
    }
    */
}

#endif
