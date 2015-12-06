classdef (HandleCompatible) LogPartEstimator
    % Interface for estimating log partition function.  
    %  Some samplers can implement this but there may be 
    %  other ways of estimating the log partition function.
    
    methods (Abstract)
        [logP, stats] = estimateLogPart( o, metadata );
        setModel( o, model );
    end
    
end

