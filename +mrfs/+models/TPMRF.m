classdef TPMRF < mrfs.models.PMRF
    properties
        R; % Primary cutoff
    end
    
    methods
        function o = TPMRF(R)
            o@mrfs.models.PMRF();
            if(nargin < 1)
                R = Inf; % No cutoff (i.e. assume the cutoff is arbitarily large)
            end
            o.R = full(R);
        end
        
        function validate(o)
            % NOTE: The main difference from PMRF is that this does NOT check for
            %  positive edge weights because TPMRF can have positive edge weights.
            
            % Check diagonal is all zeros
            o.validateDiagonal();
            
            % Check for symmetric and AND of edges if requested
            validate@mrfs.models.MRF(o);
        end
        
        
        function [thetaNodeSample, thetaEdgeSample] = getModLParams( o, Lsample )
            % Hack so that it works with LPMRF code
            thetaNodeSample = o.thetaNode;
            thetaEdgeSample = o.thetaEdge;
        end
    end
    
    methods (Access = protected)
        function [logPropXt, stats] = logProp(o, Xt, metadata)
            % Compute sublinear sufficient statistics
            if(any(Xt(:) > o.R))
                error('TPMRF:OutOfDomain', 'All count values for this TPMRF model must be <= %d.  Please consider using a different model or a larger R parameter', o.R);
            end
            [logPropXt, stats] = logProp@mrfs.models.PMRF(o, Xt, metadata);
        end
    end
end

