classdef IndependentModel < mrfs.models.ProbabilityModel
    %INDEPENDENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        thetaNode; % Independent term
    end
    
    methods
        function set.thetaNode( o, input )
            o.thetaNode = input;
            o.clearCache();
        end
    end
    
end

