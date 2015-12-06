classdef NegativeLogLogistic < mrfs.models.modl.MetaModLFunc
    properties (Access = protected)
        c; % Scaling constant for Lmean
    end
    
    methods
        function o = NegativeLogLogistic( c )
            if(nargin < 1); c = 2; end
            o.c = c;
        end
        
        % Move from metaModlFunc -> modLFunc where modLFunc is a function handle
        function modLFunc = createModLFunc( o, params )
            alpha = o.c*params.Lmean;
            beta = 2;
            modLFunc = @(Lvec) mrfs.models.modl.NegativeLogLogistic.modLFunc(Lvec, alpha, beta );
        end
        
        function str = name( o )
            str = sprintf( 'NLL(c=%.1f)', o.c );
        end
    end

    % Actual modL functional form
    methods (Static)
        function modLVec = modLFunc( Lvec, alpha, beta )
            modLVec = 1 - 1./( 1 + (Lvec./alpha).^(-beta) );
        end
    end
end

