classdef Inverse < mrfs.models.modl.MetaModLFunc
    properties (Access = protected)
        c; % Scaling constant for Lmean
    end
    
    methods
        function o = Inverse( c )
            if(nargin < 1); c = 2; end
            o.c = c/2;
        end
        
        % Move from metaModlFunc -> modLFunc where modLFunc is a function handle
        function modLFunc = createModLFunc( o, params )
            meanScale = 1.3*o.c
            alpha = meanScale*params.Lmean;
            modLFunc = @(Lvec) mrfs.models.modl.Inverse.modLFunc( Lvec, alpha );
        end
        
        function str = name( o )
            str = sprintf( 'Inverse(c=%.1f)', o.c );
        end
    end

    % Actual modL functional form
    methods (Static)
        function modLVec = modLFunc( Lvec, alpha )
            modLVec = alpha./(Lvec+alpha/10);
        end
    end
end

