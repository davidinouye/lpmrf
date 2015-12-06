classdef Mixture < mrfs.models.ProbabilityModel
    properties
        modelArray;
        trainXtArray;
        trainIndicatorBoolMat;
    end
    
    properties (SetAccess = protected)
        mixtureLearner = mrfs.learners.meta.MixtureLearner();
    end
    
    properties (SetAccess = private)
        k;
    end
    
    methods
        function k = get.k( o )
            k = length(o.modelArray);
        end
        
        % Trains the model
        function stats = train(o, Xt, learner, metadata)
            if(nargin < 4); metadata = []; end
            train@mrfs.models.ProbabilityModel( o, Xt, learner, metadata );
            
            %if(~isa(learner, 'mrfs.learners.meta.MixtureLearner') && ~isa(learner, 'mrfs.learners.meta.MultMixtureLearnerTune')); 
            %    error('MixtureModel:NotMixtureLearner', 'Learner is not a mixture model learner rather it is a %s learner.', class(learner) ); 
            %end
            
            [params, stats] = learner.learn( Xt, metadata);
            
            o.modelArray = params.modelArray;
            o.trainXtArray = params.XtArray;
            o.trainIndicatorBoolMat = params.indicatorBoolMat;
            
            o.validate();
        end
        
        % Tests the model
        function [val, stats] = test(o, Xt, evaluator, metadata)
            if(nargin < 3); metadata = []; end
            [ XtArray, indicatorBoolMat, ~ ] = o.mixtureLearner.fitXtArray( Xt, o.modelArray, metadata );
            
            % Evaluate each topic matrix in turn
            valVec = zeros(o.k,1);
            statsArray = cell(o.k,1);
            for j = 1:o.k
                [valVec(j), statsArray{j}] = evaluator.evaluate( XtArray{j}, o.modelArray{j}, metadata );
            end
            
            % Aggregate values based on evaluator
            [val, stats] = evaluator.aggregateResults( valVec, statsArray, indicatorBoolMat );
        end
        
        % Log-Likelihood
        function [logL, stats] = logLikelihood( o, Xt, metadata )
            if(nargin < 3); metadata = []; end
            
            % Get logL based on MLE/MAP estimates
            error('Incorrect implementation for now');
            [~, logLvec, statsXtArray] = o.mixtureLearner.fitXtArray( Xt, o.modelArray, metadata );
            logL = sum(logLvec);
            
            stats = [];
            stats.logL = logL;
            stats.logLvec = logLvec;
            stats.totalNumWords = sum(Xt(:));
            stats.perplexity = exp( -logL/stats.totalNumWords );
            
            if(isfield(statsXtArray, 'clusterVec'))
                stats.clusterVec = statsXtArray.clusterVec;
            end
        end
        
        % Validate the parameters of the model
        function validate(o)
            assert( o.k == length(o.modelArray), 'Somehow k and length(modelArray) are different.');
            % Just check to make sure modelArray is populated with models
            for mi = 1:length(o.modelArray)
                if( ~isa( o.modelArray{1}, 'mrfs.models.ProbabilityModel' ) )
                    error('MixtureModel:NonModelInArray', 'One of the models in modelArray is not a mrfs.models.ProbabilityModel');
                end
            end
        end
        
    end
    
    methods (Access = protected)
        function str = getArgString( o )
            if(~isempty(o.modelArray))
                modelStr = o.modelArray{1}.name;
                for j = 2:length(o.modelArray)
                    modelStr = sprintf('%s,%s',modelStr, o.modelArray{j}.name);
                end
                str = sprintf('models=%s', modelStr);
            else
                str = 'base=?';
            end
        end
    end
    
end
