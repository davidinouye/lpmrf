classdef LPMRF < mrfs.models.PMRF
    properties
        Lmean;
        modLFunc; % Function of Lvec that computes a weighting depending on L (e.g. 1/L or 1/L^2)
    end
    
    properties (SetAccess = protected)
        logPartEstimator;
        nSamplesDefault;
    end
    
    properties (Access = protected)
        cacheLogPvec = []; % In order to save time if logPvec already computed
    end
    
    methods
        
        function o = LPMRF( nSamplesDefault, logPartEstimator )
            if(nargin < 1 || isempty(nSamplesDefault)); nSamplesDefault = 1000; end % Want accurate estimates
            if(nargin < 2 || isempty(logPartEstimator)); logPartEstimator = mrfs.samplers.LPMRF_AISSampler(); end
            o.nSamplesDefault = nSamplesDefault; % Set default number of samples (can be overidden in logLikelihood call)
            o.logPartEstimator = logPartEstimator;
        end
        
        function set.modLFunc( o, input )
            o.modLFunc = input;
            o.clearCache();
        end

        function clearCache( o )
            o.message(3, 'LogPvec cache cleared.');
            o.cacheLogPvec = [];
        end
        
        function setLogPCache( o, logPvec )
            o.cacheLogPvec = logPvec;
        end
        
        function [logL, stats] = logLikelihood(o, Xt, metadata)
            if(nargin < 3); metadata = []; end
            
            %% Compute logPvec
            Lvec = full(sum(Xt, 2));
            nonZero = (Lvec > 0);
            logLvec = zeros(size(Xt,1),1);
            
            if(sum(nonZero) == 0) % Trivial case
                logPvec = [];
            else
                % Get logPvec if needed
                logPvec = o.getLogPvec( Lvec, metadata );

                % Handle the case of zero length (note: 0 vectors lead to a 0 logL so we can ignore)
                XtNonZero = Xt(nonZero, :);
                logPnonZero = logPvec(Lvec(nonZero));

                % Compute likelihood for each instance
                logLvec(nonZero) = o.logProportion( XtNonZero ) - logPnonZero;
            end
            
            logL = sum(logLvec);

            % Save stats for use later
            stats = [];
            stats.logL = logL;
            stats.logLvec = logLvec;
            stats.logPvec = logPvec; % Save partition vector
            stats.totalNumWords = sum(Xt(:));
            stats.perplexity = exp( -logL/stats.totalNumWords );
        end
        
        function stats = train(o, Xt, learner, metadata)
            % Initialize p and words (optionally display message)
            if(nargin < 4); metadata = struct(); end
            train@mrfs.models.ProbabilityModel(o, Xt, learner, metadata);
            
            % Train model
            [params, stats] = learner.learn(Xt, metadata);
            
            % Update parameters
            o.thetaNode = params.thetaNode;
            o.thetaEdge = params.thetaEdge;
            o.Lmean = params.Lmean;
            o.modLFunc = params.modLFunc;
            
            % Validate
            o.validate();
        end
        
        function validate(o)
            % Check that Lmean is positive
            assert(o.Lmean > 0, 'LPMRF Lmean parameter is negative.');

            % Check that modLFunc is a function handle
            assert( isa(o.modLFunc,'function_handle') == 1, 'LPMRF:ModLFunc', 'ModLFunc is not a function handle');
            
            % Check to make sure modL can handle vectors and is simple numeric output
            domain = (1:10)';
            modL = o.modLFunc(domain);
            assert( isequal( size(modL), size(domain) ), 'LPMRF:ModLFunc', 'Size of ModLFunc output is not the same as the input.');
            assert( isnumeric(modL), 'LPMRF:ModLFunc', 'ModLFunc does not return a numeric array.');
            
            % Shift thetaNode so that log-sum-exp is 0 since shift can be arbitrary
            o.thetaNode = o.thetaNode - log(sum(exp(o.thetaNode)));
                       
            % Check diagonal is all zeros
            o.validateDiagonal();

            % Check for symmetric and AND of edges if requested
            validate@mrfs.models.MRF(o);
        end
        
        % Critically used in sampling the log partition function
        function [thetaNodeModL, thetaEdgeModL] = getModLParams( o, L )
            thetaNodeModL = o.thetaNode;
            thetaEdgeModL = o.modLFuncClean( L )*o.thetaEdge;
        end
        
        function logP = getLogP( o, L, metadata )
            if(nargin < 3); metadata = []; end
            if( L == 0 ); logP = 0; return; end % Handle trivial case
            logPvec = o.getLogPvec( (1:L)', metadata );
            logP = logPvec(L);
        end
        
        
        function logPvec = getLogPvec(o, Lvec, metadata)
            if(nargin < 2); Lvec = 1; end
            if(nargin < 3); metadata = []; end
            
            % If thetaEdge = 0, then compute Multinomial log partition in closed form
            if(nnz(o.thetaEdge) == 0)
                Lmax= max(Lvec);
                fullLvec = (1:Lmax)';
                o.cacheLogPvec = log(sum(exp(o.thetaNode)))*fullLvec - gammaln(fullLvec+1);
            end
            
            % Return/update cached version if available
            if(~isempty(o.cacheLogPvec) && max(Lvec) <= length(o.cacheLogPvec))
                logPvec = o.cacheLogPvec;
                return;
            end
            
            % Set defaults
            if(isfield(metadata, 'nSamples'))
                nSamples = metadata.nSamples; % Override model default
            else
                nSamples = o.nSamplesDefault; % Otherwise use model default
            end
            
            if(isfield(metadata, 'nTest'))
                nTest = metadata.nTest;
            else
                nTest = 5;
            end
            
            % Test several points
            Lmax = max(Lvec);
            Ltest = ceil( o.Lmean * linspace(0.5,3,nTest) );
            
            % Setup 
            o.message(2, sprintf('-- Estimating log partition function with %d samples (nSamples = %d, nTest = %d)', nSamples*nTest, nSamples, nTest));
            t1 = tic;
            o.logPartEstimator.verbosity = o.verbosity-1;
            o.logPartEstimator.setModel( o ); % Set the current model to this one
            
            % Calculate the partition value for each L
            logPvecTest = zeros(length(Ltest), 1);
            for ii = 1:length(Ltest)
                L = Ltest(ii);
                estimatorMetadata = struct('L', L, 'nSamples', nSamples); % Set L for estimator
                t2 = tic;
                [meanLogP, stats] = o.logPartEstimator.estimateLogPart( estimatorMetadata );
                if(stats.stdDev < 1)
                    logPvecTest(ii) = meanLogP + 2*stats.stdDev; % Upper bound
                else
                    logPvecTest(ii) = 1e3; % Very large value since bad approximation
                end
                o.message(3, sprintf('Sampled L=%d in %g s', L, toc(t2)) );
            end
            timeLogP = toc(t1);
            
            % Find upperbound (i.e. take max of the two points)
            %   NOTE: If one has bad +/-, then this will select very large constant
            aisLogPvecTest = logPvecTest + gammaln(Ltest+1)'; % Mod by gammaln
            constantVec = zeros(size(Ltest));
            for i = 1:length(Ltest)
                constantVec(i) = aisLogPvecTest(i)/(o.modLFunc(Ltest(i))*Ltest(i)^2);
            end
            maxC = max(constantVec);
            
            % Compute final logPvec
            fullLvec = (1:Lmax)';
            maxQuadForm = o.modLFunc(fullLvec).*(fullLvec.^2).*maxC;
            logPvec = maxQuadForm + fullLvec.*log(sum(exp(o.thetaNode))) - gammaln(fullLvec+1);
            % To plot the approximation
            %plot(fullLvec(1:160),maxQuadForm(1:160),'r--','LineWidth',2); hold on; plot(Ltest, aisLogPvecTest,'bo','LineWidth',1);hold off; ylabel('Log Partition Function (ignoring constants)'); xlabel('L'); legend({'Smooth Max Approximation','AIS Estimates'},'Location','SouthEast');
            
            % Save to cache for use later
            o.cacheLogPvec = logPvec; % Save into cache for usage later
            o.message(2, '-- Finished estimating log partition function and saved in cache');
            o.message(2, 'Timing: (p,nnz,nSamples,nTest,timeLogP)');
            o.message(2, sprintf('%d,%d,%d,%d,%g',...
                size(o.thetaEdge,1), nnz(o.thetaEdge), nSamples, nTest, timeLogP));

        end
    end
    
    methods (Access = protected)
        
        % Critically used by logProportion (superclass) to compute everything but the log partition constant
        function [logPropXt, stats] = logProp(o, Xt, metadata)
            Lvec = full(sum(Xt,2));
            Lvec(Lvec == 0) = 1; % Remove Inf/NaN problem for divide by 0
            logPropXt = Xt*o.thetaNode + o.modLFuncClean(Lvec).*sum((Xt*o.thetaEdge).*Xt,2) + metadata.XtBaseMeasure;
            stats = [];
        end
        
        % Handle problems like 1/L = 1/0 = Inf, which should just be 1 since L = 0
        function modLvec = modLFuncClean( o, Lvec )
            modLvec = o.modLFunc( Lvec );
            modLvec( isinf(modLvec) ) = 1;
            modLvec( isnan(modLvec) ) = 1;
        end
    end
end

