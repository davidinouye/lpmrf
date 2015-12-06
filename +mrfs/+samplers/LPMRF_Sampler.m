classdef LPMRF_Sampler < mrfs.samplers.Multinomial_Sampler

    methods (Abstract, Access = protected)
        sampleL_protected( o, nSamples, Lsample, thetaNodeSample, thetaEdgeSample );
    end
    
    methods
        function o = LPMRF_Sampler(model)
            if(nargin < 1); model = []; end
            o@mrfs.samplers.Multinomial_Sampler(model);
        end
        
        function setModel(o, model)
            %if(~isempty(model) && ~isa(model, 'mrfs.models.LPMRF'))
            %    error('LPMRF_Sampler:OnlyLPMRF', 'This can only handle LPMRF objects (or subclasses).');
            %end
            setModel@mrfs.samplers.Sampler(o, model);
        end
        
        % Convenience method for simple L sampling
        function [XtSample, logW] = sampleL( o, nSamples, Lsample, saveSamples )
            if(nargin < 4); saveSamples = true; end
            
            % Get appropriate thetaNode and thetaEdge from model
            [thetaNodeSample, thetaEdgeSample] = o.model.getModLParams( Lsample );
            
            % Sample (subclasses need to implement this method)
            [XtSample, logW] = o.sampleL_protected( nSamples, Lsample, thetaNodeSample, thetaEdgeSample );
            
            % Save samples if requested
            if(saveSamples)
                o.XtSample = XtSample;
                o.logW = logW;
                o.LsampleVec = Lsample*ones(nSamples,1);
            end
        end
        
        function [logP, stats] = estimateLogPart( o, metadata )
            if(nargin < 2); metadata = []; end
            
            % Clear samples if different L
            if(isfield(metadata,'L'))
                if(any(o.LsampleVec ~= metadata.L))
                    o.XtSample = [];
                    o.logW = [];
                end
            end
            
            % Sample if needed
            if(isempty(o.XtSample))
                if(isfield(metadata, 'nSamples'))
                    o.sample(metadata.nSamples, metadata);
                else
                    o.sample([], metadata); % Use default number of samples for this sampler
                end
            end

            % Compute the log partition function
            [thetaNodeSample, ~] = o.model.getModLParams( o.LsampleVec(1));
            logZ0 = o.LsampleVec*log(sum(exp( thetaNodeSample ))) - gammaln(o.LsampleVec+1);
            logZ1 = o.logW + logZ0;
            C = max(logZ1); % To avoid overflow
            logP = log( mean( exp(logZ1 - C) ) ) + C;

            % Output some stats like an upper bound on the standard deviation of logP
            stats = [];
            varLogZ1 = var(logZ1);
            stats.stdDev = sqrt(varLogZ1); % Upper bound on standard deviation (i.e. same as N = 1) but taking N into account seems difficult
            
            % Output message
            if(all(o.LsampleVec == o.LsampleVec(1)))
                Lstr = sprintf('%d', o.LsampleVec(1));
            else
                Lstr = 'all';
            end
            o.message(2, sprintf('L=%s, LogP=%g +/- %g\n', Lstr, logP, stats.stdDev )); 
        end
        
    end
   
end

