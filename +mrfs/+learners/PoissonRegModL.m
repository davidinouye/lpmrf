classdef PoissonRegModL < mrfs.learners.PoissonReg
    %POISSONREGMODL Also returns a weighting function for L that is dependent
    %  on the other parameters of the learned model (i.e. Lmean).
    %  The default weighting function is mrfs.models.modl.NegativeLogLogistic.
    %
    %  Experimental:
    %  If a modLParam is not supplied, then this learner will attempt
    %  to find the best parameter for mrfs.models.modl.NegativeLogLogistic.
    
    properties
        metaModLFunc;
    end
    
    methods
        function o = PoissonRegModL( lambda, beta, modLParam )
            if(nargin < 1); lambda = []; end
            if(nargin < 2); beta = []; end
            if(nargin < 3 || isempty(modLParam)); modLParam = 1; end
            
            o@mrfs.learners.PoissonReg( lambda, beta );
            if(isa(modLParam, 'mrfs.models.modl.MetaModLFunc'))
                o.metaModLFunc = modLParam;
            elseif(~isempty(modLParam))
                % Assume negative log logistic with c = modLParam
                o.metaModLFunc = mrfs.models.modl.NegativeLogLogistic( modLParam );
            else
                o.metaModLFunc = [];
            end
        end
        
        function [params, stats] = learn(o, Xt, metadata)
            %% Run regressions
            tStart = tic;
            [params, stats] = learn@mrfs.learners.PoissonReg( o, Xt, metadata );
            regTime = toc(tStart);
            
            %% Determine modLFunc
            % Create metaModLFunc if not already instantiated
            tStart = tic;
            if( isempty(o.metaModLFunc) )
                [o.metaModLFunc, stats] = ...
                    o.learnMetaModLFunc( Xt, params, stats, metadata );
            end
            modLTime = toc(tStart);
            
            % Compute modLFunc using metaModLFunc and params
            params.modLFunc = o.metaModLFunc.createModLFunc( params );
            
            %% Save some stats
            stats.regTime = regTime;
            stats.modLTime = modLTime;
        end
        
        function [bestMetaModLFunc, stats] = learnMetaModLFunc( o, Xt, params, stats, metadata )
            if(nargin < 3); stats = []; end
            if(nargin < 4); metadata = []; end
            
            % Setup/train model to evaluate likelihood
            params.modLFunc = @(Lvec) ones(size(Lvec)); % Trival, default modLFunc just to initialize
            cachedLearner = mrfs.learners.meta.CachedLearner( o, params, stats );
            model = mrfs.models.LPMRF();
            model.verbosity = 0;
            model.train( Xt, cachedLearner, metadata );

            % Setup possible constants
            cVec = linspace( 1, 2, 5 );
            %cVec = [2.289,2.368,2.447];
            best.logL = -Inf;
            for ci = 1:length(cVec)
                % Set modLFunc of model
                curMetaFunc = mrfs.models.modl.NegativeLogLogistic( cVec(ci) );
                model.modLFunc = curMetaFunc.createModLFunc( params );
                
                % Evaluate current modLFunc
                tStart = tic;
                if( ~isfield(metadata,'nSamples'))
                    metadata.nSamples = round(size(Xt,2)/10);
                    metadata.nTest = 50;
                end
                logLcur = model.logLikelihood( Xt, metadata );
                perpCur = exp(-logLcur/sum(Xt(:)));
                o.message(2, sprintf('  ModL constant %g using %d samples at %d locations, perp = %.2f, logL = %g in %g s',...
                    cVec(ci), metadata.nSamples, metadata.nTest, perpCur, logLcur, toc(tStart) ));

                % Keep best modLFunc
                if( logLcur > best.logL)
                    best.logL = logLcur;
                    bestMetaModLFunc = curMetaFunc;
                end
            end
            
            % Save a few other stats in the stats struct
            stats.cVec = cVec;
        end
    end
    
    methods (Access = protected)
        function str = getArgString( o )
            if(isa(o.metaModLFunc, 'mrfs.models.modl.MetaModLFunc'))
                str = sprintf('lam=%g, beta=%g, func=%s', o.lambda, o.beta, o.metaModLFunc.name() );
            else
                str = sprintf('lam=%g, beta=%g, func=?', o.lambda, o.beta );
            end
        end
    end
    
end

