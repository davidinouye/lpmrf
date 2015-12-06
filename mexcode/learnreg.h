#ifndef _LEARNREG_H // Include guard
#define _LEARNREG_H

#include <iostream> // cout,cerr,etc.
#include <stdio.h>  // printf, etc.
#include <stdexcept> // Standard exceptions
#include <omp.h>

// Eigen
#include <Eigen/Dense> 
#include <Eigen/SparseCore>
#include "utils.h"
typedef Eigen::VectorXd VecType;
typedef Eigen::MatrixXd MatType;

// Simple output operator for vectors
/*
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}
*/

/***** Definition of exponential families *****/
#ifndef _EXPFAM // Include guard
#define _EXPFAM
enum ExpFam { Poisson };
#endif

// Sufficient statistics T(x)
template<ExpFam Family, typename T1, typename T2> void T( const T1& x, T2& y ) { y = x; }

// Log partition function A(theta)
template<ExpFam Family, typename T1, typename T2> void A( const T1& x, T2& y ) { y = x.derived().array().exp(); }
template<ExpFam Family> void A( const double& x, double& y ) { y = std::exp(x); }

// Derivative of log partition function dA(theta)/dtheta
template<ExpFam Family, typename T1, typename T2> void dA( const T1& x, T2& y ) { A<Family>(x,y); };
template<ExpFam Family, typename T1, typename T2, typename T3> void dA( const T1& x, const T2& A, T3& y ) { y = A; };

// Second derivative of log partition function d^2A(theta)/dtheta^2
template<ExpFam Family, typename T1, typename T2> void d2A( const T1& x, T2& y ) { A<Family>(x,y); };
template<ExpFam Family, typename T1, typename T2, typename T3> void d2A( const T1& x, const T2& A, T3& y ) { y = A; };
template<ExpFam Family, typename T1, typename T2, typename T3, typename T4> void d2A( const T1& x, const T2& A, const T3& dA, T4& y ) { y = A; };

/***** Logical interface for shifted (remove s column) and padded (add all ones column) X using original X *****/
template<typename XType>
struct VirtualMatrix {
    const XType& origX; // Reference to the original matrix class
    bool isTransposed;
    size_t p; // Total number of nodes
    size_t n; // Total number of nodes
    size_t s; // The current column from 0..p-1?

    VirtualMatrix(const XType& origX, size_t s) : origX(origX), s(s), isTransposed(false) {
        p = origX.cols(); // Extract number of variables
        n = origX.rows(); // Extract number of instances/rows
    }

    VirtualMatrix(const XType& origX, size_t s, bool isTransposed ) : origX(origX), s(s), isTransposed(isTransposed) {
        p = origX.cols(); // Extract number of variables
        n = origX.rows(); // Extract number of instances/rows
    }

    VirtualMatrix<XType> transpose() const {
        return VirtualMatrix<XType>(origX, s, !isTransposed);
    }

    VecType operator*(const VecType& y ) const {
        // Calculate product in piecewise fashion
        size_t nFront = s;
        size_t nBack = p-s-1;
        if( isTransposed ) {
            assert(y.cols() == 1 && y.rows() == n && "y is not the correct size");
            VecType result = VecType::Zero(p,1);
            result(0) = y.sum();
            if(nFront > 0) {
                result.segment(1,nFront) = origX.leftCols(nFront).transpose()*y;
            }
            if(nBack > 0) {
                result.segment(nFront+1,nBack) = origX.rightCols(nBack).transpose()*y;
            }
            return result;
        } else {
            assert(y.cols() == 1 && y.rows() == p && "y is not the correct size");
            if(nFront > 0 && nBack > 0) {
                VecType v(origX.leftCols(nFront)*y.block(1,0,nFront,1) + origX.rightCols(nBack)*y.bottomRows(nBack));
                return y(0) + v.array();
            } else if (nFront > 0) { // Implicitly nBack = 0
                VecType v(origX.leftCols(nFront)*y.block(1,0,nFront,1));
                return y(0) + v.array();
            } else { // Implicitly nFront = 0
                VecType v(origX.rightCols(nBack)*y.bottomRows(nBack));
                return y(0) + v.array();
            }
        }
    }

    template<typename T, typename T2>
    VecType hessianDiag( const T& d2A, const T2& colIdx, double scale ) const {
        VecType Y = VecType::Zero(colIdx.size(),1);
        double temp;
        size_t t;
        for (size_t i = 0; i < colIdx.size(); ++i) { // Only loop over selected colIdx
            if( colIdx[i] == 0 ) {
                temp = d2A.sum(); // Just ones times diagVec
            } else {
                t = adjustForS( colIdx[i] );
                temp = origX.col(t).cwiseProduct(d2A).dot( origX.col(t) );
            }
            Y(i) = scale*temp;
        }
        return Y;
    }

    template<typename T>
    void updateR(const double mu, const size_t t, T& r) const {
        if( t == 0 ) {
            r.array() += mu; // All ones column
        } else {
            r += mu*origX.col( adjustForS(t) );
        }
    }

    template<typename T, typename T2>
    double dotR( const T& r, const T2& d2A, const size_t t, const double scale ) const {
        if(t == 0) {
            return scale*r.dot(d2A); // All ones
        } else {
            return scale*(r.transpose()*origX.col(adjustForS(t)).cwiseProduct(d2A)).sum();
        }
    }

    /** Note this function is not used anymore **/
    // Function to compute the hessian
    // To get indices based on vector: MatrixXi indices = (A.array() < 3).cast<int>();
    template<typename T, typename T2, typename T3>
    void XdiagX( const T& diagVec, const T2& colIdx, T3& Y, const double scale ) const {
        if( isTransposed ) {
            std::cerr << "XdiagX is not implemented for transpose matrices yet" << std::endl;
            std::exit(1);
        }
        // Error check
        assert(diagVec.size() == n && "diagVec is not a vector");
        
        // Need to adjust colIdx based on s
        Y = T3::Zero( colIdx.size(), colIdx.size() );
        // Assuming sparse matrix
        for (size_t i=0; i < colIdx.size(); ++i) { // Only loop over selected colIdx
            for (size_t i2=0; i2 < colIdx.size(); ++i2) { // Only loop over selected colIdx
                double temp = 0;
                size_t t = adjustForS( colIdx[i] ); // Change actual column depending on s
                size_t t2 = adjustForS( colIdx[i2] ); // Change actual column depending on s
                if( colIdx[i] == 0 && colIdx[i2] == 0 ) {
                    temp = diagVec.sum(); // Just ones times diagVec
                } else if( colIdx[i] == 0 || colIdx[i2] == 0) {
                    // Choose which column to sum over
                    size_t colT;
                    if( colIdx[i] == 0 ) { colT = t2; }
                    else { colT = t; }
                    // Sum over this column scaling by diagVec
                    for( typename XType::InnerIterator it(origX, colT); it; ++it) {
                        temp += diagVec[it.row()] * it.value();
                    }
                } else {
                    temp = origX.col(t).cwiseProduct(diagVec).dot(origX.col(t2));
                }
                Y(i2, i) = scale*temp;
            }
        }
    }

    template<typename T>
    T adjustForS( const T t ) const {
        // Adjust the column index t to account for the fact that we are using logical indexing
        if( t <= s ) {
            return t-1; // Subtract 1 because of all ones column
        } else {
            return t; // Otherwise same because of all ones column
        }
    }
};

template<typename XType>
std::ostream& operator<< (std::ostream& out, const VirtualMatrix<XType>& vX ) {
    for(size_t r = 0; r < vX.n; ++r) {
        for(size_t c = 0; c < vX.p; ++c) {
            if(c == 0) {
                out << 1;
            } else if( c <= vX.s ) {
                out << " " << vX.origX.coeff(r, c-1 );
            } else {
                out << " " << vX.origX.coeff(r, c );
            }
        }
        if(r < vX.n-1) out << std::endl;
    }
    return out;
}

template<ExpFam Family>
struct GeneralizedLinearModel {
    // Evaluate objective value
    private:
    double verbosity;

    double evalobj( const VecType& beta, const VecType& Acur, const VecType& xTilde, const double lam ) const {
        double n = Acur.size();
        double lassoTerm = lam * beta.tail(beta.size()-1).array().abs().sum();
        double linTerm = beta.dot(xTilde);
        double Aterm = Acur.sum(); 
        return (1/n)*( -linTerm + Aterm ) + lassoTerm;
    }

    public:
    GeneralizedLinearModel( ): verbosity(3) {}
    GeneralizedLinearModel( double verbosity ): verbosity(verbosity) {}

    // Main learning function
    template< typename XType, typename YType, typename BetaType >
    void learnreg(
            const VirtualMatrix<XType>& vX, // Note this is a virtual X rather than an eigen type 
            const YType& Y, 
            const double lam,
            BetaType& beta ) const {
        // Declare variables
        VecType eta, xTilde, grad, Acur, dAcur, d2Acur;
        BetaType betaNew; // To store proposed new beta
        MatType hessianFree;
        std::vector<size_t> freeSet;
        size_t n = vX.n, p = vX.p;
        double obj, obj0, objNew;

        // Initialize variables
        if(beta.size() != p) { // If beta is uninitialized
            beta = BetaType::Zero(p,1);
            // Initialize node parameter near mean
            beta(0) = std::log( Y.sum()/Y.size() );
            eta = VecType::Constant(n,1,beta(0));
        } else {
            eta = vX*beta;
        }
        A<Family>(eta, Acur);
        xTilde = vX.transpose()*Y;
        obj = obj0 = evalobj( beta, Acur, xTilde, lam );

        // Outer loop
        size_t maxOuterIter = 500;
        for(size_t outerIter = 0; outerIter < maxOuterIter; ++outerIter) {
            // Calculate gradient
            dA<Family>(eta, Acur, dAcur);
            grad = (1/(double)n)*(-xTilde + vX.transpose()*dAcur);

            // Compute free set 
            freeSet.clear();
            for(size_t i = 0; i < grad.size(); ++i)
                if( i == 0 || std::abs(grad(i)) >= lam || beta(i) != 0 )
                    freeSet.push_back(i); 

            // Calculate Hessian on free set
            d2A<Family>(eta, Acur, dAcur, d2Acur);
            VecType hessianDiag = vX.hessianDiag( d2Acur, freeSet, 1/(double)n);

            // Inner loop to approximate Newton direction
            VecType dFree = VecType::Zero(freeSet.size());
            VecType r = VecType::Zero(n,1); // Maintain X*d product which is initially 0
            size_t t;
            double a,b,c,mu,z;
            size_t maxInnerIter = floor(1 + ((double)outerIter+1)/3.0);
            for(size_t innerIter = 0; innerIter < maxInnerIter; ++innerIter) {
                for(size_t i = 0; i < freeSet.size(); ++i) {
                    // Solve single variable problem
                    t = freeSet[i];
                    a = hessianDiag(i);
                    b = grad(t) + vX.dotR(r, d2Acur, t, 1/(double)n);
                    c = beta(t) + dFree(i);
                    if( t == 0 ) {
                        mu = -b/a; // Without regularization
                    } else {
                        z = c - b/a;
                        mu = -c + copysign( fmax( std::abs(z) - lam/a, 0), z );
                    }
                    if(mu != 0) {
                        dFree(i) += mu;
                        vX.updateR(mu, t, r);
                    }
                }
                //message(3,verbosity, "    innerIter = %d", innerIter);
            }

            // Inner loop to calculate step size
            size_t maxStepIter = 50;
            double stepSize = 1, stepParam1 = 0.5, stepParam2 = 1e-10, stepConstant;
            for(size_t stepIter = 0; stepIter < maxStepIter; ++stepIter ) {
                // Compute new beta
                betaNew = beta;
                for( size_t i = 0; i < freeSet.size(); ++i ) 
                    betaNew[ freeSet[i] ] += stepSize*dFree(i);

                // Compute some constant stepsize quantities
                if( stepIter == 0 ) {
                    double gradTimesD = 0;
                    for( size_t i = 0; i < freeSet.size(); ++i ) 
                        gradTimesD += grad(freeSet[i])*dFree(i);
                    double sumBeta = beta.array().abs().sum();
                    double sumBeta0 = betaNew.array().abs().sum();
                    stepConstant = stepParam2*(gradTimesD + sumBeta0 - sumBeta);  
                }

                // Compute objective
                eta = vX*betaNew; 
                A<Family>(eta, Acur);
                objNew = evalobj( betaNew, Acur, xTilde, lam );
                
                // Check Armijo step condition
                if( objNew <= obj + stepSize*stepConstant ) {
                    break;
                } else {
                    stepSize *= stepParam1;
                }
            }

            // Update parameters
            double relDiffBeta = (betaNew-beta).norm()/beta.norm();
            beta = betaNew;
            double relDiff = (obj-objNew)/obj;
            obj = objNew;
            //message(0, verbosity, "  outerIter = %d, obj = %g, relDiffObj = %g, relDiffBeta = %g, stepSize = %g", outerIter, objNew, relDiff, relDiffBeta, stepSize );
            if(relDiffBeta < 1e-5) {
                break;
            }
        }
    }

    // Alias when calling with simple XType
    template< typename XType, typename YType, typename BetaType >
    void learnreg(
            const XType& X,
            const YType& Y, 
            const double lam,
            const double nodeBeta,
            BetaType& beta ) const {
        // Create padded X
        XType Xpadded(X.rows(), X.cols()+1);
        Xpadded.rightCols(X.cols()) = X;
        // Make virtual X from padded (i.e. remove column 0 which was already padded)
        VirtualMatrix<XType> vX(Xpadded, 0);
        // Setup Y based on given nodeBeta
        YType Ymod = Y.array() + nodeBeta;
        // Run regression with this new virtualized program
        learnreg(vX, Ymod, lam, beta);
    }
};

template<ExpFam Family>
struct GeneralizedMRF {
    private:
        double verbosity;
        size_t nThreads;

    public:
    GeneralizedMRF( ): verbosity(2) {}
    GeneralizedMRF( double verbosity ): verbosity(verbosity) {}
    GeneralizedMRF( double verbosity, size_t nThreads ): verbosity(verbosity), nThreads(nThreads) {
        if(nThreads != 0) {
            omp_set_num_threads(nThreads);
        }
    }
    
    /*
    typedef Eigen::SparseMatrix<double> XType;
    typedef Eigen::VectorXd ThetaNodeType;
    typedef Eigen::SparseMatrix<double> ThetaEdgeType;
    */

    // Alias for nodeBeta = 0
    /*
    template< typename XType, typename ThetaNodeType, typename ThetaEdgeType >
    void learnmrf(
            const XType& X, // Note this is a virtual X rather than an eigen type 
            const double lam,
            ThetaNodeType& thetaNode,
            ThetaEdgeType& thetaEdge ) {
        learnmrf(X,lam,0,thetaNode,thetaEdge);
    }
    */

    template< typename XType, typename ThetaNodeType, typename ThetaEdgeType >
    void learnmrf(
            const XType& X,
            const double lam,
            const double nodeBeta,
            ThetaNodeType& thetaNode,
            ThetaEdgeType& thetaEdge ) {
        // Initialize variables
        size_t n = X.rows();
        size_t p = X.cols();
        if(thetaNode.size() != p){
            // Initialize node variables to close to correct if no edges instead of 0
            VecType ones = VecType::Constant(n,1,1);
            VecType sumX = (X.transpose()*ones).array() + nodeBeta*n;
            thetaNode = (sumX/X.rows()).array().log();
        }
        if(thetaEdge.cols() != p || thetaEdge.cols() != p) {
            thetaEdge.resize(p,p);
            thetaEdge.setZero();
        }

        // Learn p regressions in parallel and combine
        std::vector< Eigen::Triplet<double> > tripletListArray[p];
        GeneralizedLinearModel<Poisson> poisson(verbosity-1); // Reduce verbosity
        size_t threadId[p];

        // Parallel for loop (dynamic because work is not evenly distributed)
        #pragma omp parallel for schedule(dynamic)
        for(size_t s = 0; s < p; ++s) {
            // Setup arguments
            VirtualMatrix<XType> vX( X, s );
            VecType phi = VecType::Zero(p,1);
            phi(0) = thetaNode(s);
            for(typename ThetaEdgeType::InnerIterator it(thetaEdge, s); it; ++it) {
                if(it.row() < s) {
                    phi(it.row()+1) = it.value();
                } else if(it.row() > s) {
                    phi(it.row()) = it.value();
                }
            }

            // Run regression
            Eigen::VectorXd y = X.col(s);
            y.array() += nodeBeta;
            poisson.learnreg( vX, y, lam, phi );
            
            // Save results in output vectors
            for(size_t i = 0; i < phi.size(); ++i) {
                if(i == 0) {
                    thetaNode(s) = phi(i);
                } else if( phi(i) != 0 ) {
                    if( i <= s) {
                        tripletListArray[s].push_back( Eigen::Triplet<double>(s, i-1, phi(i) ));
                    } else {
                        tripletListArray[s].push_back( Eigen::Triplet<double>(s, i, phi(i) ));
                    }
                }
            }
            threadId[s] = omp_get_thread_num();
            //std::cout << threadId[s];
        }
        //std::cout << std::endl;

        // Display output checking for thread_id
        //for(size_t s = 0; s < p; ++s) message(2,verbosity, "s = %d, thread_id = %d", s, threadId[s]);

        // Concatenate all tripletLists
        size_t nnz = 0;
        for(size_t s = 0; s < p; ++s) nnz += tripletListArray[s].size();

        std::vector< Eigen::Triplet<double> > triplets;
        triplets.reserve(nnz);
        for(size_t s = 0; s < p; ++s)
            triplets.insert(triplets.end(), tripletListArray[s].begin(), tripletListArray[s].end());
        assert( triplets.size() == nnz && "Triplets size is not equal to the concatenation of all triplet list sizes");

        // Create thetaEdge from this triplet list
        thetaEdge.setFromTriplets( triplets.begin(), triplets.end() );
    }
};

#endif
