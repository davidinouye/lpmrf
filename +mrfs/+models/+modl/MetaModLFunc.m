classdef MetaModLFunc < mrfs.VerboseHandler
    methods (Static, Abstract)
        metaFunc = modLFunc( params, varargin )
    end
    
    methods (Abstract)
        modLFunc = createModLFunc( params );
    end
end

