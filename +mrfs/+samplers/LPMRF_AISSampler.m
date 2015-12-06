classdef LPMRF_AISSampler < mrfs.samplers.LPMRF_MCSampler & mrfs.LogPartEstimator

    properties
        betaVec;
        nGibbs;
    end
    
    methods
        function o = LPMRF_AISSampler(model, beta, nGibbs)
            if(nargin < 1); model = []; end
            if(nargin < 2); beta = 100; end
            if(nargin < 3); nGibbs = 1; end
            
            o@mrfs.samplers.LPMRF_MCSampler(model);
            
            % Setup beta depending
            if(length(beta) == 1)
                o.betaVec = linspace(0,1,beta);
            elseif(length(beta) > 1)
                o.betaVec = beta;
            else
                error('Beta should either be a scalar denoting the number of steps or a vector of beta values.');
            end
           
            % Setup other variables
            o.nGibbs = nGibbs;
        end
    end
    
    methods (Access = protected)
        function [XtSample, logW] = sampleL_protected(o, nSamples, Lsample, thetaNodeSample, thetaEdgeSample)
            % Split into edge nodes and independent nodes
            depNodes = (full(sum(thetaEdgeSample~=0,2)) > 0);
            nDepNodes = sum(depNodes);
            if(nDepNodes == 0) % Completely independent than just sample multinomial
                multProb = exp(thetaNodeSample)/sum(exp(thetaNodeSample));
                XtSample = mnrnd(Lsample, multProb', nSamples);
                logW = zeros(nSamples, 1);
            elseif(nDepNodes == o.model.p)
                % All nodes are dependent so just sample directly
                [logW, XtSample] = o.ais(thetaNodeSample, thetaEdgeSample, Lsample, nSamples);
            else
                % Sample first from [dependentNodes, nInd] and then from independent nodes using L = LInd
                thetaNodeDep = [thetaNodeSample(depNodes); log(sum(exp(thetaNodeSample(~depNodes))))];
                thetaEdgeDep = thetaEdgeSample(depNodes, depNodes);
                thetaEdgeDep(end+1,end+1) = 0; % Extend by 1 row and 1 column
                [logW, XtDep] = o.ais(thetaNodeDep, thetaEdgeDep, Lsample, nSamples);
                
                % Sample independent words based on previous sample
                LindVec = XtDep(:,end); % Number of independent words for each sample
                indProb = exp(thetaNodeSample(~depNodes))/sum(exp(thetaNodeSample(~depNodes)));
                XtInd = mnrnd(LindVec, indProb);
                
                % Combine sampling results for final output
                %  (Note: logW is the same as the depLogW because ind part is the same)
                XtSample(:,[find(depNodes); find(~depNodes)]) = [XtDep(:,1:(end-1)), XtInd];
            end
        end

        function [logW, Xt] = ais(o, thetaNodeSample, thetaEdgeSample, Lsample, nSamples)
            % NOTE: thetaNodeSample and thetaEdgeSample are not necessarily 
            %  o.model.thetaNode or o.model.thetaEdge because of simplifications
            [logW, X] = mrfs.samplers.ais_mex( thetaNodeSample, thetaEdgeSample, Lsample, nSamples, o.betaVec, o.nGibbs, o.verbosity);
            Xt = X';
        end
    end
    
end

