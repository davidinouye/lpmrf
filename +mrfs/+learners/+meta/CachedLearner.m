classdef CachedLearner < mrfs.learners.Learner
    
    properties
        origLearner;
        params;
        stats;
    end
    
    methods
        % Simply save and return cached param and stats values
        function o = CachedLearner( origLearner, params, stats)
            o.origLearner = origLearner;
            o.params = params;
            o.stats = stats;
        end
        
        function [params, stats] = learn(o, Xt, metadata)
            params = o.params;
            stats = o.stats;
        end
    end
    
end

