classdef LogLikeEvaluator < mrfs.evaluators.Evaluator
    
    properties (SetAccess = protected)
        WORST_VAL = -Inf;
    end
    
    methods
        function [val, stats] = evaluate(o, Xt, model, metadata )
            if(nargin < 4); metadata = []; end
            [val, stats] = model.logLikelihood( Xt, metadata );
        end
        
        function TF = isBetter(o, newVal, curVal)
            if(newVal > curVal)
                TF = true;
            else
                TF = false;
            end
        end
        
        function [val, stats] = aggregateResults( o, valVec, statsArray, indicatorBoolMat )
            % Init 
            [n,k] = size(indicatorBoolMat);
            assert( k == length(valVec), 'LogLikeEvaluator:aggregateResults:  k given from indicatorBoolMat and k from valVec are different');
            
            % Aggregate stats
            val = 0;
            totalNumWords = 0;
            logLvec = zeros(n,1); % Don't actually know how large it will be
            logPArray = cell(k,1);
            for j = 1:k
                val = val + valVec(j); % Simply aggregate logLikelihood values
                totalNumWords = totalNumWords + statsArray{j}.totalNumWords;
                compBool = indicatorBoolMat(:,j); % Boolean for logLvec
                logLvec( compBool ) = logLvec( compBool ) + statsArray{j}.logLvec;
                logPArray{j} = statsArray{j}.logPvec;
            end

            % Save stats into output
            stats = [];
            stats.logL = val;
            stats.logLvec = logLvec;
            stats.logPArray = logPArray;
            stats.logPvec = [];
            stats.totalNumWords = totalNumWords;
            stats.perplexity = exp( -stats.logL/stats.totalNumWords );
            
            % Save full basic parameters just in case
            stats.indicatorBoolMat = indicatorBoolMat;
            stats.statsArray = statsArray; 
            stats.valVec = valVec;
        end
    end
end

