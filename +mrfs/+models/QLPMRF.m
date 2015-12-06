classdef QLPMRF < mrfs.models.LPMRF
    
    methods
        function o = QLPMRF(nSamples, logPartEstimator)
            if(nargin < 1); nSamples = []; end
            if(nargin < 2); logPartEstimator = []; end
            
            % Simple constant function for modLFunc
            modLFunc = @(Lvec) ones(size(Lvec));
            
            o@mrfs.models.LPMRF(modLFunc, nSamples, logPartEstimator);
        end
    end
    
end

