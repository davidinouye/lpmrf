classdef LPMRFTopicModelLearner < mrfs.learners.meta.LPMRFMixtureLearner
    
    properties (SetAccess = protected)
        %priorLogLFunc = @( j, probVec ) 0; % Dirichlet with alpha = 1
        convergeThreshold = 1e-6;
        convergeAltThreshold = 1e-6;
    end
    
    methods
        function o = LPMRFTopicModelLearner( k, modelClass, learnerClass, learnerParamArray, maxIter )
            if(nargin < 1); k = []; end
            if(nargin < 2); modelClass = []; end
            if(nargin < 3); learnerClass = []; end
            if(nargin < 4); learnerParamArray = []; end % Set to empty if none given
            if(nargin < 5); maxIter = []; end
            o@mrfs.learners.meta.LPMRFMixtureLearner( k, modelClass, learnerClass, learnerParamArray, maxIter );
        end

        % Initialize with Multinomial topic model rather than random
        function [XtArray, indicatorBoolMat, stats] = initXtArray( o, Xt )
            % Setup multinomial topic model learner
            
            temp = o.learnerClass();
            if(isa(temp, 'mrfs.learners.MultLearner'))
                % Use simple Multinomial mixture if learner is MultLearner
                multTMLearner = mrfs.learners.meta.MixtureLearner( ...
                    o.k, @mrfs.models.Multinomial, @mrfs.learners.MultLearner, {1e-4}, 100 );
                o.message(2,sprintf('\n<< Using Multinomial Mixture Learner to initialize Multinomial Topic Model >>\n'));
                strModel = 'Multinomial';
            else
                % If learner is LPMRF learner, then use MultTopicModelLearner
                multTMLearner = mrfs.learners.meta.LPMRFTopicModelLearner( ...
                    o.k, @mrfs.models.LPMRF, @mrfs.learners.MultLearner, {1e-4}, 100 );
                o.message(2,sprintf('\n<< Using Multinomial Topic Model Learner to initialize LPMRF Topic Model >>\n'));
                strModel = 'LPMRF';
            end
            multTMLearner.verbosity = o.verbosity;
            
            % Train initial model
            params = multTMLearner.learn( Xt );
            o.message(2,sprintf('\n<< Finished initializing %s Topic Model >>\n', strModel));
            
            % Get XtArrays from trained model
            XtArray = params.XtArray;
            indicatorBoolMat = params.indicatorBoolMat;

            % Save clusterVec
            stats = [];
        end
        
        %% Split into k different matrices using MAP estimates
        function [XtArray, indicatorBoolMat, stats] = fitXtArray( o, Xt, modelArray, metadata )
            if(nargin < 3); metadata = []; end
            
            % Initialize
            if( isfield( metadata, 'initXtArray') )
                initXtArray = metadata.initXtArray;
                initIndicatorBoolMat = metadata.initIndicatorBoolMat;
            else
                % Smarter LPMRF mixture initialization
                [initXtArray, initIndicatorBoolMat] = fitXtArray@mrfs.learners.meta.MixtureLearner( o, Xt, modelArray, metadata );
            end
            
            % Precompute and prepare values for mex function
            k = length(modelArray);
            Lmax = max(full(sum( Xt, 2 )));
            Lvec = 0:Lmax;

            logPMat = zeros(k, Lmax+1);
            modLMat = zeros(k, Lmax+1);
            modelStructArray = cell(size(modelArray));
            tStart = tic;
            for j = 1:length(modelArray)
                % NOTE: Lmax may be smaller this time so only extract needed values
                temp = modelArray{j}.getLogPvec( Lvec, metadata )';
                logPMat(j,2:end) = temp(1:(size(logPMat,2)-1));
                modLMat(j,:) = modelArray{j}.modLFunc( Lvec );
                warning('off','all');
                modelStructArray{j} = struct(modelArray{j});
                warning('on','all');
            end
            timeForLogP = toc(tStart);
            [logLInitial, perpInitial] = o.topicModelLikelihood( initXtArray, modelArray );
            
            % Run mex file
            XArray = o.Xt2XArray(initXtArray, initIndicatorBoolMat);
            o.message(2, sprintf('-- Fitting %d topic matrices for each instance (learnalltopicmats_mex)', size(Xt,1) ));
            tStart = tic;
            XArray = mrfs.learners.meta.learnalltopicmats_mex(XArray, modelStructArray, logPMat, modLMat);
            timeForFit = toc(tStart);
            [XtArray, indicatorBoolMat] = o.X2XtArray( XArray );
            
            % Compute relDiff before and after
            [logLFinal, perpFinal] = o.topicModelLikelihood( XtArray, modelArray );
            o.message(3, sprintf('Perplexity initial: %g, Perplexity final: %g', perpInitial, perpFinal));
            relDiff = (logLFinal-logLInitial)/abs(logLInitial);

            % Output a few stats
            o.message(2,'-- Finished fitting topic matrices (learnalltopicmats_mex)');
            o.message(2,'Timing: (n,p,k,relDiff,time)');
            o.message(2, sprintf('%d,%d,%d,%g,%g', size(Xt,1), size(Xt,2), length(modelArray), relDiff, timeForFit));
            
            if(relDiff > 0)
                o.message(3,sprintf('RelDiff = %g, Likelihood is increasing!', relDiff));
            elseif(relDiff == 0)
                o.message(3,sprintf('Hmm...Likelihood is not changing!'));
            else
                error('RelDiff = %g, Likelihood of training set is decreasing!', relDiff);
            end
            
            % Save stats
            stats = [];
            stats.relDiff = relDiff;
            stats.timeForFit = timeForFit;
            stats.timeForLogP = timeForLogP;
            if( nargout >= 3 )
                % Compute logLvec
                %stats.logLvec = logLvec;
            end
        end
        
    end
        
    
    methods (Access = protected)

        % Called by super objects
        function TF = hasConverged( o, statsXtArray )
            o.message(2, sprintf('\n<< AlternatingIteration: relDiff = %g (converge threshold = %g) >>\n', statsXtArray.relDiff, o.convergeAltThreshold ));
            if( statsXtArray.relDiff <= o.convergeAltThreshold )
                TF = true;
            else
                TF = false;
            end
        end
        
        function [logL, perplexity] = topicModelLikelihood(o, XtArray, modelArray)
            logL = 0;
            totalNumWords = 0;
            for j = 1:length(modelArray)
                logL = logL + modelArray{j}.logLikelihood( XtArray{j} );
                totalNumWords = totalNumWords + sum(XtArray{j}(:));
            end
            perplexity = exp( -logL/totalNumWords );
        end
        
        function XArray = Xt2XArray(o, XtArray, indicatorBoolMat )
            [n,k] = size(indicatorBoolMat);
            p = size(XtArray{1},2);
            XArray = cell(k,1);
            for j=1:k
                XArray{j} = sparse(p,n);
                XArray{j}(:,indicatorBoolMat(:,j)) = XtArray{j}';
                o.message(3,sprintf('Xt2X  topic %d=%d docs', j, full(sum(indicatorBoolMat(:,j)))) );
            end
        end
        
        function [XtArray, indicatorBoolMat] = X2XtArray(o, XArray)
            % Setup indicatorBoolMat
            k = length(XArray);
            n = size(XArray{1},2);
            XtArray = cell(k,1);
            indicatorBoolMat = false(n, k);
            for j = 1:k
                indicatorBoolMat(:,j) = (full(sum(XArray{j}, 1)) > 0);
                o.message(3,sprintf('X2Xt  topic %d=%d docs', j, full(sum(indicatorBoolMat(:,j)))) );
                tempXt = XArray{j}';
                tempXt = tempXt(indicatorBoolMat(:,j), :); % Only select nonzero rows
                XtArray{j} = mrfs.utils.setmatrixtype( tempXt );
            end
        end
    end
   
end

