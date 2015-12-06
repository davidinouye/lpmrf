classdef TopicModel < mrfs.models.Mixture
    
    methods
        
        function o = TopicModel()
            o@mrfs.models.Mixture();
            o.mixtureLearner = mrfs.learners.meta.LPMRFTopicModelLearner();
        end
    end
    
end
