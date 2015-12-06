classdef SPMRF < mrfs.models.TPMRF
    properties
        R0; % First cutoff from paper (initial bending of suff. statistics)
    end
    
    methods 
        function o = SPMRF(R0, R)
            if(R0 > R)
                error('SPMRF Error: R0 should be smaller than R but R0 = %g, and R = %g', R0, R);
            end
            o@mrfs.models.TPMRF(R);
            o.R0 = full(R0);
        end
    end

    methods (Access = protected)
        function [logPropXt, stats] = logProp(o, Xt, metadata)
            % Compute sublinear sufficient statistics
            %Xt(Xt <= o.R0) = Xt(Xt <= o.R0); % These sufficient statistics stay the same
            quadRegime = (Xt > o.R0 & Xt <= o.R);
            Xt(quadRegime) = (-1/(2*(o.R-o.R0)))*Xt(quadRegime).^2 + (o.R/(o.R - o.R0))*Xt(quadRegime) - o.R0^2/(2*(o.R-o.R0)); % These sufficient statistics stay the same
            Xt(Xt > o.R) = (o.R + o.R0)/2;
            
            % Compute log partition as before but with sublinear sufficient statistics
            [logPropXt, stats] = logProp@mrfs.models.PMRF(o, Xt, metadata);
            stats.sublinearXt = Xt;
        end
    end
end

