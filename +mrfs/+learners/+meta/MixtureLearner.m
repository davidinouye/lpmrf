classdef MixtureLearner < mrfs.learners.Learner
    
    properties (SetAccess = protected)
        k;
        modelClass;
        learnerClass;
        learnerParamArray;
        maxIter;
    end
    
    methods
        function o = MixtureLearner( k, modelClass, learnerClass, learnerParamArray, maxIter )
            if(nargin < 1 || isempty(k) ); k = 3; end
            if(nargin < 2 || isempty(modelClass) ); modelClass = @mrfs.models.Multinomial; end
            if(nargin < 3 || isempty(learnerClass) ); learnerClass = @mrfs.learners.MultLearner; end
            if(nargin < 4 || isempty(learnerParamArray)); learnerParamArray = {}; end % Set to empty if none given
            if(nargin < 5 || isempty(maxIter)); maxIter = 10; end
            o.k = k;
            o.modelClass = modelClass;
            o.learnerClass = learnerClass;
            o.learnerParamArray = learnerParamArray;
            o.maxIter = maxIter;
        end
        
        function [params, stats] = learn(o, Xt, metadata)
            if(nargin < 3); metadata = []; end
            % Randomly initialize XtArray and assignments
            [XtArray, indicatorBoolMat] = o.initXtArray( Xt );
            
            % Initialize and train models
            modelArray = cell(o.k, 1);
            for j = 1:o.k
                learner = o.learnerClass( o.learnerParamArray{:} );
                learner.verbosity = o.verbosity;
                modelArray{j} = o.modelClass();
                modelArray{j}.verbosity = o.verbosity;
                modelArray{j}.train( XtArray{j}, learner, metadata );
            end
            
            % Debugging code
            % Check perplexity
            %logL = 0;
            %totalNumWords = 0;
            %for j = 1:length(modelArray)
            %    logL = logL + modelArray{j}.logLikelihood( XtArray{j}, metadata );
            %    totalNumWords = totalNumWords + sum(XtArray{j}(:));
            %end
            %perplexity = exp( -logL/totalNumWords );
            %o.message(3, sprintf('Perplexity after initial fitting = %g\n', perplexity));
            %assert(totalNumWords == sum(Xt(:)));
            
            % Alternate between clustering and model fitting
            o.message(2, sprintf('\n<< %s: Starting alternating iterations >>\n', class(o)) );
            for iter = 1:(o.maxIter-1) % NOTE: Already did 1 iteration as initialization
                % Make new assignments based on current model
                metadata.initIndicatorBoolMat = indicatorBoolMat;
                metadata.initXtArray = XtArray;
                [XtArray, indicatorBoolMat, statsXtArray] = o.fitXtArray( Xt, modelArray, metadata );
                
                % Stopping condition
                if( o.hasConverged( statsXtArray ) )
                    break;
                end
                
                % Learn new models based on reassignments
                for j = 1:o.k
                    learner = o.learnerClass( o.learnerParamArray{:} );
                    learner.verbosity = o.verbosity;
                    modelArray{j}.train( XtArray{j}, learner, metadata );
                end
            end
            o.message(2, sprintf('\n<< %s: Finished alternating iterations >>\n', class(o)) );
            if(iter == o.maxIter); o.message(1, sprintf('MixtureLearner: MaxIter of %d reached', o.maxIter)); end
            
            % Set modelArray of params
            params = [];
            params.modelArray = modelArray;
            params.XtArray = XtArray;
            params.indicatorBoolMat = indicatorBoolMat;
            stats = [];
        end
        
        function [XtArray, indicatorBoolMat, stats] = initXtArray( o, Xt )
            % K-means initialization (10 different initializations)
            nReps = 10;
            o.message(2, sprintf('<< Initializing Multinomial mixture with k-means (%d repetitions) >>', nReps));
            bestSumSquared = Inf;
            for i = 1:nReps
                rng(i);
                curClusterVec = mrfs.utils.litekmeans(Xt', o.k)';
                sumSquared = 0;
                for j = 1:o.k
                    Xj = Xt(curClusterVec == j,:);
                    meanJ = mean(Xj);
                    sqDist = dot(meanJ,meanJ) - 2*full(Xj*meanJ') + sum(Xj.*Xj,2); % Squared distance
                    sumSquared = sumSquared + sum(sqDist);
                end
                if(sumSquared < bestSumSquared)
                    bestSumSquared = sumSquared;
                    clusterVec = curClusterVec;
                end
                o.message(2,sprintf('  k-means rng(%d): sumSquared = %g', i, sumSquared));
            end
            
            % Extract
            XtArray = cell(o.k, 1);
            for j = 1:o.k
                XtArray{j} = Xt(clusterVec == j, :);
            end
            o.message(2, sprintf('<< Finished initializing Multinomial mixture with k-means >>\n'));
            
            % Convert to indicator bool mat
            indicatorBoolMat = o.clusterVec2BoolMat( clusterVec, o.k );
            
            stats = [];
            stats.a = 1;
        end
        
        % Split into k different matrices using MAP estimates of cluster index
        function [XtArray, indicatorBoolMat, stats] = fitXtArray( o, Xt, modelArray, metadata )
            if(nargin < 3); metadata = []; end
            
            % Compute best component for each instance
            kTemp = length(modelArray);
            [n, ~] = size(Xt);
            logLMat = zeros(n, kTemp);
            for j = 1:kTemp
                [~, stats] = modelArray{j}.logLikelihood( Xt, metadata );
                logLMat(:,j) = stats.logLvec;
            end
            
            % Compute MAP estimate of cluster vector
            [logLvec, clusterVec] = max(logLMat, [], 2);
            
            % Check for empty clusters
            try
                reassigned = [];
                for j = 1:kTemp
                    if(sum(clusterVec == j) == 0)
                        % Assign the 10 points that have the best likelihood to this cluster even though not max
                        [~, bestIdxJ] = sort(logLMat(:,j),'descend');

                        % Remove already assigned points
                        [~, IA, ~] = intersect(bestIdxJ, reassigned);
                        bestIdxJ(IA) = [];

                        % Select the top 10 points and reassign them to cluster j
                        if(length(bestIdxJ) >= 10)
                            newJ = bestIdxJ(1:10);
                        else
                            newJ = bestIdxJ(:)';
                        end
                        clusterVec(newJ) = j;
                        reassigned = [reassigned, newJ];
                    end
                end
            catch
                o.message(1,'Problem assigning points to empty cluster');
            end
            
            % Extract split Xt based on clusterVec
            XtArray = cell(kTemp, 1);
            for j = 1:kTemp
                XtArray{j} = Xt(clusterVec == j, :);
            end
            
            % Create indicator bool mat
            indicatorBoolMat = mrfs.learners.meta.MixtureLearner.clusterVec2BoolMat( clusterVec, kTemp );
            
            % Save stats
            stats = [];
            if( isfield(metadata, 'initIndicatorBoolMat') )
                stats.nDiff = round(nnz(metadata.initIndicatorBoolMat ~= indicatorBoolMat)/2);
            else
                stats.nDiff = Inf;
            end
            stats.logLvec = logLvec;
        end
        
    end
    
    methods (Access = protected)
        
        function TF = hasConverged( o, statsXtArray )
            o.message(2, sprintf('\n<< AlternatingIteration: nChanged = %d (threshold = 0)>>\n', statsXtArray.nDiff));
            if( statsXtArray.nDiff == 0 )
                TF = true;
            else
                TF = false;
            end
        end
        
        function str = getArgString( o )
            try 
                paramStr = sprintf('%g,', o.learnerParamArray{:});
                if( ~isempty(paramStr) )
                    paramStr(end) = []; % Remove last comma
                end
            catch
                paramStr = '?';
            end
            str = sprintf( 'base=%s:%s(%s)', ...
                func2str(o.modelClass), ...
                func2str(o.learnerClass), ...
                paramStr ...
                );
        end
        
    end
    
    methods (Static)
        
        function indicatorBoolMat = clusterVec2BoolMat( clusterVec, kTemp )
            nTemp = length(clusterVec);
            indicatorBoolMat = sparse((1:nTemp)', clusterVec, true(nTemp,1), nTemp, kTemp);
        end
        
    end
    
end

