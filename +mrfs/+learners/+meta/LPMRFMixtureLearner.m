classdef LPMRFMixtureLearner < mrfs.learners.meta.MixtureLearner
    
    methods
        
        function o = LPMRFMixtureLearner( k, modelClass, learnerClass, learnerParamArray, maxIter )
            if(nargin < 1); k = []; end
            if(nargin < 2); modelClass = []; end
            if(nargin < 3); learnerClass = []; end
            if(nargin < 4); learnerParamArray = []; end % Set to empty if none given
            if(nargin < 5); maxIter = []; end
            o@mrfs.learners.meta.MixtureLearner( k, modelClass, learnerClass, learnerParamArray, maxIter );
        end
        
        % Initialize with Multinomial mixture rather than random
        function [XtArray, indicatorBoolMat, stats] = initXtArray( o, Xt )
            % Setup multinomial mixture and learner
            multMixLearner = mrfs.learners.meta.MixtureLearner( o.k, @mrfs.models.Multinomial, @mrfs.learners.MultLearner, {1e-4}, 100 );
            multMixLearner.verbosity = o.verbosity;
            
            % Train Multinomial mixture
            fprintf('Using MultinomiaMixtureLearnerTune to initialize arrays');
            params = multMixLearner.learn( Xt );
            
            % Get XtArrays from trained model
            XtArray = params.XtArray;
            indicatorBoolMat = params.indicatorBoolMat;
            
            % Debugging code
            %modelArray = params.modelArray;
            %logL = 0;
            %totalNumWords = 0;
            %for j = 1:length(modelArray)
            %    logL = logL + modelArray{j}.logLikelihood( XtArray{j} );
            %    totalNumWords = totalNumWords + sum(XtArray{j}(:));
            %end
            %perplexity = exp( -logL/totalNumWords );
            %o.message(3,sprintf('Perplexity of multinomial mixture before training LPMRF = %g\n',perplexity) );
                        
            % Save clusterVec
            stats = [];
        end
        
    end
    
end

