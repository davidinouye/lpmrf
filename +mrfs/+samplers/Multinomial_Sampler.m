classdef Multinomial_Sampler < mrfs.samplers.Sampler
    %MULTINOMIAL_SAMPLER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (GetAccess = public, SetAccess = protected)
        LsampleVec;
    end
    
    methods
        function o = Multinomial_Sampler(model)
            if(nargin < 1); model = []; end
            o@mrfs.samplers.Sampler(model);
        end
        
        function setModel(o, model)
            if(~isempty(model) && ~( isa(model, 'mrfs.models.Multinomial') || isa(model, 'mrfs.models.LPMRF') ))
                error('Multinomial_Sampler:OnlyMultOrLPMRF', 'This can only handle Multinomial or LPMRF objects (or subclasses).');
            end
            setModel@mrfs.samplers.Sampler(o, model);
        end
        
        % Sample from the joint distribution of Poisson*LPMRF
        function [XtSample, logW] = sample( o, nSamples, metadata )
            if(nargin < 2 || isempty(nSamples)); nSamples = o.nSamplesDefault; end
            if(nargin < 3); metadata = []; end
            
            o.LsampleVec = o.getLsampleVec( nSamples, o.model, metadata );
            if(isempty(nSamples)); nSamples = length(o.LsampleVec); end % Slight hack for density sampler
            
            % Group LsampleVec into groups of like L
            maxLsample = max(o.LsampleVec);
            LDomain = 0:maxLsample;
            if(numel(LDomain) == 1)
                Lcounts = numel(o.LsampleVec); % Just when L = 0
            else
                Lcounts = hist(o.LsampleVec, LDomain)';
                assert(sum(Lcounts) == nSamples, 'The number of grouped counts is not the same as the total number of samples.');
            end
            
            % Sample from each needed L with the appropriate number of samples
            XtSample = zeros(nSamples, o.model.p);
            logW = zeros(nSamples, 1);
            curI = 1;
            groupStarts = [1; cumsum(Lcounts)+1];
            for Lsample = LDomain
                nSamplesL = Lcounts(Lsample+1); % Add one since 1 indexed
                
                % Sample with specific L
                idx = curI:(curI-1+nSamplesL);
                curI = curI+nSamplesL;
                idx2 = groupStarts(Lsample+1):(groupStarts(Lsample+2)-1);
                assert(isequaln(idx,idx2), 'Indexing is different.');
                
                if(Lsample > 0 && nSamplesL > 0)
                    [XtSample(idx,:), logW(idx')] = o.sampleL(nSamplesL, Lsample, false);
                end
                
                % NOTE: Rescaling of weights is already implicitly accomplished 
                %  by sampling from the mean Poisson.
            end
            
            % Cache sample output for later use
            o.XtSample = XtSample;
            o.logW = logW;
        end
        
        % Convenience method for simple L sampling
        function [XtSample, logW] = sampleL( o, nSamples, Lsample, saveSamples )
            if(nargin < 4); saveSamples = true; end
            
            % Sample from multinomial directly
            if(isa( o.model, 'mrfs.models.Multinomial' ))
                multProb = model.multProb;
            elseif(isa( o.model, 'mrfs.models.IndependentModel' ))
                multProb = exp( model.thetaNode )/sum(exp( model.thetaNode ));
            end
            XtSample = mnrnd( Lsample, multProb, nSamples );
            logW = zeros( nSamples, 1 );
            
            % Save samples if requested
            if(saveSamples)
                o.XtSample = XtSample;
                o.logW = logW;
                o.LsampleVec = Lsample*ones(nSamples,1);
            end
        end
    end
    
    
    methods ( Access = protected )
        function LsampleVec = getLsampleVec(o, nSamples, model, sampleMetadata)
            if(isfield(sampleMetadata, 'L'))
                % Sample with all the same L
                if(isscalar(sampleMetadata.L))
                    LsampleVec = sampleMetadata.L*ones(nSamples, 1);
                else
                    error('LPMRF_Sampler:LNotScalar', 'L given must be a scalar value.  Use Lvec instead if you want specific samples with different L.');
                end
            elseif(isfield(sampleMetadata, 'Lvec'))
                LsampleVec = sampleMetadata.Lvec;
            else
                % Consistent sampling for same Lmean and nSamples
                rngS = rng(); rng(1);
                LsampleVec = poissrnd(model.Lmean, nSamples, 1);
                rng(rngS);
            end
            LsampleVec = sort(LsampleVec); % Sort sample vec
        end
    end
    
end

