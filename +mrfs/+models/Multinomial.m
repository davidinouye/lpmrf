classdef Multinomial < mrfs.models.IndependentModel
    
    %% Core methods
    properties (SetAccess = protected)
        Lmean;
        multProb;
    end
    
    
    methods
        % Trains the model
        function stats = train(o, Xt, learner, metadata)
            if(nargin < 4); metadata = struct(); end
            train@mrfs.models.ProbabilityModel( o, Xt, learner, metadata );
            
            % Train model
            [params, stats] = learner.learn(Xt, metadata);
            
            % Update parameters and normalize
            C = max(params.thetaNode);
            logSumExp = log(sum(exp( params.thetaNode - C ))) + C;
            o.thetaNode = params.thetaNode - logSumExp;
            o.multProb = exp( o.thetaNode )/sum(exp( o.thetaNode ));
            o.Lmean = params.Lmean;
            
            o.validate();
        end
        
        % Log proportion of instances (i.e. without normalizing constants)
        function [logPropXt, stats] = logProportion(o, Xt, metadata)
            if(nargin < 3); metadata = []; end
            if(~isfield(metadata, 'XtBaseMeasure'))
                metadata.XtBaseMeasure = mrfs.utils.poissonbasemeasure( Xt );
            end
            logPropXt = Xt*o.thetaNode + metadata.XtBaseMeasure;
            stats = [];
        end
        
        % Log-Likelihood
        function [logL, stats] = logLikelihood(o, Xt, metadata)
            Lvec = full(sum( Xt, 2 ));
            
            % Directly compute logPvec
            Ltest = (1:max(Lvec))';
            logPvec = Ltest*log(sum(exp( o.thetaNode ))) - gammaln(Ltest + 1);
            
            % Handle the case of zero length (note: 0 vectors lead to a 0 logL so we can ignore)
            nonZero = (Lvec > 0);
            XtNonZero = Xt(nonZero, :);
            logPnonZero = logPvec(Lvec(nonZero));

            % Compute likelihood for each instance
            
            logLvec = zeros(size(Xt,1),1);
            logLvec(nonZero) = o.logProportion( XtNonZero ) - logPnonZero;
            logL = sum(logLvec);
            
            % Save stats for use later
            stats = [];
            stats.logL = logL;
            stats.logLvec = logLvec;
            stats.logPvec = logPvec; % Save partition vector
            stats.totalNumWords = sum(Xt(:));
            stats.perplexity = exp( -logL/stats.totalNumWords );
        end
        
        % Validate the parameters of the model
        function validate(o)
            errorMsg = sprintf('Sum of probability is %g off from summing to 1',  abs(sum(o.multProb)-1) );
            assert(abs(sum(o.multProb)-1) < o.p*eps, errorMsg);
            assert(~any(isnan(o.thetaNode)), 'Sum of thetaNode are NaN' );
            assert(~any(isinf(o.thetaNode)), 'Sum of thetaNode are Inf or -Inf' );
        end
    end

end

