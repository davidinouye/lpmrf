classdef VerboseHandler < handle
    %VERBOSEHANDLER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        verbosity = 1; % Default to issuing warnings
    end
    
    methods
        % Display message depending on verbosity
        function message(o, level, msg)
            if(o.verbosity >= level)
                if(level == 1)
                    warning(msg);
                else
                    fprintf('%s%s\n', repmat('  ',1,level-1), msg);
                end
            end
        end
        
        function str = name( o )
            str = sprintf('%s(%s)', strrep(class(o),'mrfs.',''), o.getArgString());
        end
    end
    
    methods (Access = protected)
        function str = getArgString( o )
            str = '?'; % Should replace by something in subclasses
        end
    end
    
end

