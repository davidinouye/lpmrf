classdef PoissonReg < mrfs.learners.Learner
    %POISSONREG Estimate thetaNode and thetaEdge using nodewise Poisson
    %  regressions or equivalently optimizing the pseudo-log likelihood.
    
    properties
        lambda;
        beta;
    end
    
    methods
        function [o] = PoissonReg( lambda, beta )
            if(nargin < 1 || isempty(lambda)); lambda = 1; end
            if(nargin < 2 || isempty(beta)); beta = 1e-4; end
            o.lambda = lambda;
            o.beta = beta;
        end
        
        function [params, stats] = learn(o, Xt, metadata)
            % Initialize
            Lmean = full(mean(sum(Xt,2)));
            if(isfield(metadata,'nThreads'))
                nThreads = metadata.nThreads;
            else
                nThreads = 0;
            end

            o.message(2, sprintf('-- Starting %d regressions in parallel (nThreads = %d; 0 means as many cores available)', size(Xt,2), nThreads) );
            tStart = tic;
            [thetaNode, thetaEdge] = mrfs.learners.learnmrf_mex( sparse(Xt), o.lambda, o.beta, nThreads);
            mexTime = toc(tStart);
            o.message(2, sprintf('-- Finished %d regressions', size(Xt,2)) );
            o.message(2, 'Timing: (n,p,time)');
            o.message(2, sprintf('%d,%d,%g', size(Xt,1), size(Xt,2), mexTime));
            
            % Estimator should be half (or equivalently just the lower triangular part)
            thetaEdge = thetaEdge/2;
            
            params = o.getParams( thetaNode, thetaEdge, Lmean );
            
            stats = [];
            stats.mexTime = mexTime;
        end
        
        function params = getParams(o, thetaNode, thetaEdge, Lmean)
            % Set parameters
            params = [];
            params.thetaNode = thetaNode;
            params.thetaEdge = thetaEdge;
            params.Lmean = Lmean;
        end
        
    end
    
    methods (Access = protected)
        function str = getArgString( o )
            str = sprintf('lam=%g, beta=%g', o.lambda, o.beta);
        end
    end
    
end

