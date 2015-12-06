classdef Learner < mrfs.VerboseHandler
    
    properties
    end
    
    methods (Abstract)
        [params, stats] = learn(obj, Xt, metadata);
    end
end

