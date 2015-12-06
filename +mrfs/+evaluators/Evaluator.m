classdef Evaluator
    %EVALUATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Abstract, SetAccess = protected)
        WORST_VAL; % Worst value for evaluator
    end
    
    methods (Abstract)
        [val, stats] = evaluate( o, Xt, model, metadata );
        TF = isBetter( o, newVal, curVal );
        %[val, stats] = aggregateResults( o, valVec, statsArray, statsXtArray ); % Aggregate multiple values (e.g. for mixture models)
    end
    
end

