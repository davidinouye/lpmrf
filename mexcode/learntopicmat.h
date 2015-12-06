#ifndef _LEARNTOPICMAT_H // Include guard
#define _LEARNTOPICMAT_H

#include <iostream> // cout,cerr,etc.
#include <stdio.h>  // printf, etc.
#include <stdexcept> // Standard exceptions
#include <omp.h>
#include <cmath>

// Eigen
#include <Eigen/Dense> 
#include <Eigen/SparseCore>
#include "utils.h"
typedef Eigen::VectorXd VecType;
typedef Eigen::MatrixXd MatType;

template<typename ThetaEdgeType>
struct LPMRF {
    private:
    const VecType& _logP;
    const VecType& _modL;

    public:
    const VecType& thetaNode;
    const ThetaEdgeType& thetaEdge;

    LPMRF(const VecType& thetaNode, const ThetaEdgeType& thetaEdge, const VecType& logP, const VecType& modL) 
        : thetaNode(thetaNode), thetaEdge(thetaEdge), _logP(logP), _modL(modL)  {}

    // Needed because of reference fields
    LPMRF operator=(const LPMRF& other) {
        return LPMRF(other.thetaNode, other.thetaEdge, other._logP, other._modL);
    }

    double logP(const size_t L) const {
        return _logP(L);
    }

    double modL(const size_t L) const {
        return _modL(L);
    }
};

template<typename ZType, template<typename> class ProbModel, typename ThetaEdgeType, typename PriorType>
class DualCoordinateStep {
    private:
    const std::vector< ProbModel<ThetaEdgeType> >& modelArray;
    const PriorType& prior;
    const std::vector<size_t>& filtIdx;
    // Maintenance variables
    VecType quadTerm; // Quad term (i.e. z_j'*thetaEdge{j}*z_j)
    MatType crossTerm; // Cross product term for step (i.e. z_j'*thetaEdge{j}*e_s = z_j'*thetaEdge{j}(:,s)
    //std::vector< MatType > thetaEdgeFiltArray;

    public:
    DualCoordinateStep(const std::vector< ProbModel<ThetaEdgeType> >& modelArray, const PriorType& prior, const ZType& ZtFilt, const std::vector<size_t>& filtIdx, const size_t Lmax) : modelArray(modelArray), prior(prior), filtIdx(filtIdx) {
        initMaintenanceVars(ZtFilt);
    }

    void initMaintenanceVars(const ZType& ZtFilt) {
        // Initialize maintenance variables
        size_t k = modelArray.size();
        size_t pFilt = filtIdx.size();
        
        // Compute cross term thetaEdge[j](filt)*ZtFilt
        // Hard to decide but updating is probably less frequent and probably needs to load a lot in memory since ell and m might be far apart
        crossTerm = MatType::Zero(k, pFilt);
        for(size_t j = 0; j < k; ++j) {
            for(size_t sFilt = 0; sFilt < pFilt; ++sFilt) {
                size_t sFilt2 = 0;
                for(typename ThetaEdgeType::InnerIterator it(modelArray[j].thetaEdge, filtIdx[sFilt]); it; ++it) {
                    while(sFilt2 < pFilt && it.row() > filtIdx[sFilt2]) ++sFilt2;
                    if(sFilt2 >= pFilt) break;
                    if(it.row() == filtIdx[sFilt2])
                        crossTerm(j,sFilt) += ZtFilt(j,sFilt2)*it.value();

                }
            }
        }

        // Compute quadratic term
        quadTerm = VecType::Zero(k, 1);
        for(size_t j = 0; j < k; ++j) {
            quadTerm(j) = ZtFilt.row(j).dot( crossTerm.row(j) );
        }
    }

    // Update for a particular step
    double update(const size_t rFilt, const size_t ell, const size_t m, ZType& ZtFilt) {
        // Check different sizes (a)
        double diff, diff0, bestDiff = 1e100;
        int bestStep;
        for(int step = -ZtFilt(ell,rFilt); step <= ZtFilt(m,rFilt); ++step) {
            diff = stepDiff( step, rFilt, ell, m, ZtFilt );
            if(step == 0) diff0 = diff;
            // Track minimum
            if(diff < bestDiff) {
                bestDiff = diff;
                bestStep = step;
            }
        }

        // If best step is non-zero, then update Zt and update maintenance variables
        if(bestStep != 0) {
            double bestStepD = bestStep;
            //std::cout << "      Moving " << bestStep << " to coord " << ell+1 << " from coord " << m+1 << std::endl;
            MatType ZtFiltBefore = ZtFilt;
            // Update ZtFilt
            ZtFilt(ell,rFilt) += bestStep;
            ZtFilt(m,rFilt) -= bestStep;

            VecType quadTermBefore = quadTerm; // Quad term (i.e. z_j'*thetaEdge{j}*z_j)
            MatType crossTermBefore = crossTerm; // Cross product term for step (i.e. z_j'*thetaEdge{j}*e_s = z_j'*thetaEdge{j}(:,s)
            // Update maintenance variables
            quadTerm(ell) += 2*bestStepD*crossTerm(ell, rFilt);
            quadTerm(m) -= 2*bestStepD*crossTerm(m, rFilt);

            // Update crossTerm
            size_t r = filtIdx[rFilt];
            size_t pFilt = filtIdx.size();
            size_t sFilt = 0;
            for(typename ThetaEdgeType::InnerIterator it(modelArray[ell].thetaEdge, r); it; ++it) {
                while(sFilt < pFilt && it.row() > filtIdx[sFilt]) ++sFilt;
                if(sFilt >= pFilt) break;
                if(it.row() == filtIdx[sFilt])
                    crossTerm(ell,sFilt) += bestStepD*it.value();
            }
            sFilt = 0;
            for(typename ThetaEdgeType::InnerIterator it(modelArray[m].thetaEdge, r); it; ++it) {
                while(sFilt < pFilt && it.row() > filtIdx[sFilt]) ++sFilt;
                if(sFilt >= pFilt) break;
                if(it.row() == filtIdx[sFilt])
                    crossTerm(m,sFilt) -= bestStepD*it.value();
            }
        }         

        double bestStepDiff = bestDiff - diff0;
        if( bestStep == 0 ) {
            bestStepDiff = 0;
        }
        return bestStepDiff;
    }

    protected:
    // Compute step difference from moving a words from ell to m of word r
    double stepDiff(const int a, const size_t rFilt, 
            const size_t ell, const size_t m, const ZType& ZtFilt) const {
        // Setup some variables
        VecType ZsumNew = ZtFilt.rowwise().sum(); 
        ZsumNew(ell) += a;
        ZsumNew(m) -= a;
        double modLell =    modelArray[ell].modL( ZsumNew(ell) );
        double modLm =      modelArray[m].modL( ZsumNew(m) );
        double logPell =    modelArray[ell].logP( ZsumNew(ell) );
        double logPm =      modelArray[m].logP( ZsumNew(m) );
        size_t r = filtIdx[rFilt];

        // Compute each term
        double quad = -( modLell*(quadTerm(ell) + 2*a*crossTerm(ell, rFilt)) 
                  + modLm*(quadTerm(m) - 2*a*crossTerm(m, rFilt)) );
        double lin = -a*( modelArray[ell].thetaNode(r) - modelArray[m].thetaNode(r) );
        double bm = lgamma( ZtFilt(ell,rFilt)+a+1 ) + lgamma( ZtFilt(m,rFilt)-a+1 );
        double logP = logPell + logPm;
        double negPrior = 0; //TODO implement prior specification
        double stepDiff = lin + quad + bm + logP + negPrior;
        /*
        std::cout.precision(5);
        std::cout << "step = " << a;
        std::cout << " lin = " << std::fixed << lin;
        std::cout << " quad = " << std::fixed << quad;
        std::cout << " bm = " << std::fixed << bm;
        std::cout << " logP = " << std::fixed << logP;
        std::cout << " negPrior = " << std::fixed << negPrior;
        std::cout << " stepDiff = " << std::fixed << stepDiff;
        std::cout << std::endl;
        */
        return stepDiff;
    }
};

template<typename ZType, template<typename> class ProbModel, typename ThetaEdgeType, typename PriorType>
double learntopicmat( const std::vector< ProbModel< ThetaEdgeType > >& modelArray, const PriorType& prior, ZType& Zt ) {
    size_t k = modelArray.size();
    assert( k == Zt.rows() && "Zt is not the right size");

    // Convert Zt to MatType Zt only on non-zero rows
    size_t Lmax = -1;
    std::vector<size_t> filtIdx;
    for(size_t c = 0; c < Zt.outerSize(); ++c) {
        double sum = 0;
        for (typename ZType::InnerIterator it(Zt,c); it; ++it) {
            sum += it.value();
        }
        if(sum != 0) filtIdx.push_back(c);
        if(ceil(sum) > Lmax) Lmax = ceil(sum);
    }
    MatType ZtFilt( k, filtIdx.size() );
    for(size_t i = 0; i < filtIdx.size(); ++i) 
        ZtFilt.col(i) = Zt.col( filtIdx[i] );

    // Setup dual coordinate step object
    DualCoordinateStep<MatType, ProbModel, ThetaEdgeType, PriorType> 
        dualStep(modelArray, prior, ZtFilt, filtIdx, Lmax);

    // Bunch of loops
    size_t outerMaxIter = 10, innerMaxIter = 2;
    double cumStepDiff = 0;
    for(size_t outerIter = 0; outerIter < outerMaxIter; ++outerIter) {
        bool outerNoMovement = true;
        // Loop over all nonzero columns of Zt
        for(size_t sFilt = 0; sFilt < filtIdx.size(); ++sFilt) {
            //varprint(filtIdx[sFilt])
            for(size_t innerIter = 0; innerIter < innerMaxIter; ++innerIter) {
                bool noMovement = true;
                for(size_t ell = 0; ell < k; ++ell) {
                    for(size_t m = ell+1; m < k; ++m) {
                        //std::cout << outerIter << " " << innerIter << " " << ell << " " << m << std::endl;
                        if(ZtFilt(ell, sFilt) == 0 && ZtFilt(m, sFilt) == 0) continue;
                        double stepDiff = dualStep.update(sFilt, ell, m, ZtFilt);
                        if(stepDiff > 0 && stepDiff < 1e-100) {
                            std::cout << stepDiff << std::endl;
                        }
                        cumStepDiff += stepDiff;
                        bool movement = (stepDiff != 0);
                        noMovement = noMovement && !movement ;
                    }
                }
                //std::cout << "    Completed inner iter " << innerIter << std::endl;
                outerNoMovement = outerNoMovement && noMovement;
                if(noMovement) break;
            }
        }
        //std::cout << "  Completed outer iter " << outerIter << std::endl;
        if(outerNoMovement) break;
    }

    //std::cout << "Done with iterations, saving output" << std::endl;
    // Extract results and update Zt
    std::vector< Eigen::Triplet<double> > triplets;
    for(size_t rFilt = 0; rFilt < ZtFilt.cols(); ++rFilt)
        for(size_t j = 0; j < ZtFilt.rows(); ++j)
            if(ZtFilt(j,rFilt) != 0) {
                triplets.push_back( Eigen::Triplet<double>(j, filtIdx[rFilt], ZtFilt(j, rFilt)) );
            }
    Zt.setFromTriplets( triplets.begin(), triplets.end() );
    //std::cout << "From returning function" << cumStepDiff << std::endl;
    return cumStepDiff;
}

// For learning all the topic matrices by simply looping through them
template<typename ZType, template<typename> class ProbModel, typename ThetaEdgeType, typename PriorType>
void learnalltopicmats(const std::vector< ProbModel<ThetaEdgeType> >& modelArray, const PriorType& prior, std::vector<ZType>& ZtArray) {
    size_t n = ZtArray.size();
    int threadNums[n];
    // Run parallel loop
    #pragma omp parallel for
    for(size_t i = 0; i < n; ++i) {
        double cumStepDiff = learntopicmat( modelArray, prior, ZtArray[i] );
        threadNums[i] = omp_get_thread_num(); 
        if(cumStepDiff < 0) {
            //std::cout.precision(10);
            //std::cout << "  i = " << i+1 << "thread_id = " << omp_get_thread_num() << ", cumStepDiff = " << std::fixed << cumStepDiff << std::endl;
        }
        if(cumStepDiff > 0) {
            //std::cout << "Error? i = " << i << ", cumStepDiff = " << cumStepDiff << std::endl;
            //exit(1);
        }
    }

    /*
    for(size_t i = 0; i < n; ++i) {
        std::cout << "Num thread = " << threadNums[i] << std::endl;
    }
    */
}

#endif
