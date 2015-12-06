classdef LPMRF_DensitySampler < mrfs.samplers.LPMRF_Sampler & mrfs.LogPartEstimator

    properties
        propX;
    end
    
    methods
        function o = LPMRF_DensitySampler(model)
            if(nargin < 1); model = []; end
            o@mrfs.samplers.LPMRF_Sampler(model);
            o.nSamplesDefault = [];
        end
    end
    
    methods (Access = protected)
        function LsampleVec = getLsampleVec(o, nSamples, model, sampleMetadata)
            % For density samplers, the number of samples is very specific
            if(isfield(sampleMetadata, 'L'))
                % Sample with all the same L
                if(isscalar(sampleMetadata.L))
                    nSamples = nchoosek(sampleMetadata.L + model.p - 1, model.p - 1);
                    LsampleVec = sampleMetadata.L * ones(nSamples, 1);
                else
                    error('LPMRF_DensitySampler:LNotScalar', 'L given must be a scalar value.');
                end
            else
                error('LPMRF_DensitySampler:LNotSpecified', 'L must be given for LPMRF_DensitySampler since it samples every point in the domain.');
            end
        end
        
        function [XtSample, logW] = sampleL_protected(o, nSamples, Lsample, thetaNodeSample, thetaEdgeSample)
            p = length(thetaNodeSample);
            % Sample exactly the true density with the exact weights
            if(Lsample == 2 && p > 100)
                [I,J,~] = find(tril(ones(p)));
                Icomb = zeros(2*length(I),1);
                Jcomb = zeros(2*length(I),1);
                Scomb = zeros(2*length(I),1);
                cur = 1;
                for ii = 1:length(I)
                    Icomb(cur) = ii;
                    Jcomb(cur) = I(ii);
                    Scomb(cur) = 1;
                    cur = cur+1;
                    
                    Icomb(cur) = ii;
                    Jcomb(cur) = J(ii);
                    Scomb(cur) = 1;
                    cur = cur+1;
                end
                XtSample = sparse(Icomb, Jcomb, Scomb);
            elseif(Lsample == 3 && p > 100)
                Icomb = zeros(nchoosek(Lsample+p-1,p-1),1);
                Jcomb = zeros(nchoosek(Lsample+p-1,p-1),1);
                Scomb = zeros(nchoosek(Lsample+p-1,p-1),1);
                cur = 1;
                ii = 1;
                for i1 = 1:p, for i2 = 1:i1, for i3 = 1:i2
                    Icomb(cur) = ii;
                    Jcomb(cur) = i1;
                    Scomb(cur) = 1;
                    cur = cur+1;
                    
                    Icomb(cur) = ii;
                    Jcomb(cur) = i2;
                    Scomb(cur) = 1;
                    cur = cur+1;
                    
                    Icomb(cur) = ii;
                    Jcomb(cur) = i3;
                    Scomb(cur) = 1;
                    cur = cur+1;
                    ii = ii + 1;
                end, end, end
            else
                XtSample = mrfs.utils.allVL1(p, Lsample); % Get all solutions
            end

            % Log partition of uniform normalization factor 
            logPuniform = gammaln((Lsample+p-1)+1) - gammaln((p-1)+1) - gammaln((Lsample)+1);
            % Log partition of Multinomial (LPMRF with thetaEdge = 0)
            logPmult = Lsample*log(sum(exp( thetaNodeSample ))) - gammaln(Lsample+1);
            
            % Want log(Plpmrf/Pmult) = log(Plpmrf/Puniform*Puniform/Pmult)
            % = log(Plpmrf/Puniform) + logPuniform - logPmult
            % = logPropLpmrf + logPuniform - logPmult
            logW = o.model.logProportion(XtSample) + (logPuniform - logPmult);
        end
    end
    
end

