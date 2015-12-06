classdef LPMRF_MCSampler < mrfs.samplers.LPMRF_Sampler & mrfs.LogPartEstimator

    methods
        function o = LPMRF_MCSampler(model)
            if(nargin < 1); model = []; end
            o@mrfs.samplers.LPMRF_Sampler(model);
            % If it is an MCSampler (NOT a subclass) 
            %  then sample 10 times more than other samplers
            if(strcmp(class(o),'mrfs.samplers.LPMRF_MCSampler'))
                o.nSamplesDefault = o.nSamplesDefault*10; 
            end
        end
    end
    
    methods (Access = protected)
        function [XtSample, logW] = sampleL_protected(o, nSamples, Lsample, thetaNodeSample, thetaEdgeSample)
            multProb = exp(thetaNodeSample)/sum(exp(thetaNodeSample));
            XtSample = mnrnd(Lsample, multProb, nSamples);
            if(isa(o.model, 'mrfs.models.MRF'))
                logW = sum((XtSample*thetaEdgeSample).*XtSample, 2);
            else
                logW = zeros(nSamples, 1); % Fully independent model
            end
        end
    end
    
end

