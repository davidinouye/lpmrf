classdef Sampler < mrfs.VerboseHandler & mrfs.ExpectationEstimator

    properties
        model;
        XtSample; % N x P matrix of samples
        logW; % Log weights for each sample
        nSamplesDefault = 1000;
    end
    
    methods (Abstract)
        sample( o, nSamples, sampleMetadata );
    end
    
    methods
        
        function o = Sampler(model)
            o.setModel( model ); % May include error checking in subclasses
        end
        
        function setModel(o, model)
            % Set model and clear previous samples and logW
            o.model = model;
            o.XtSample = [];
            o.logW = [];
        end

        function empProb = density(o)
            % Sample if needed
            if(isempty(o.XtSample))
                nSamples = 1000;
                L = 10;
                o.message(1, sprintf('No previous samples.  Sampling %d samples at L = %d to calculate density.', nSamples, L));
                o.sample( nSamples, struct('L', L) );
            end
            
            % Setup distribution
            [n,p] = size(o.XtSample);
            Lmax = full(max(sum(o.XtSample,2)));
            nEntries = (Lmax+1)^p;
            if(nEntries > 1e6)
                error('Number of entries (%d) is too large for density function.  Please only use density function for testing purposes.');
            end
            empProb = zeros( repmat( Lmax+1, 1, p ) );
            subs = mat2cell( o.XtSample+1, n, ones(p, 1) );
            inds = sub2ind(size(empProb), subs{:});
            W = exp(o.logW - max(o.logW));
            
            %for i = 1:n
            %    empProb(inds(i)) = empProb(inds(i)) + W(i);
            %end
            %empProb = empProb/sum(empProb(:));
            
            % Faster way of doing the above loop
            empProb = accumarray(inds, W./sum(W), [], [], [], true);
            %empProb = empProb/sum(empProb);
            empProb(nEntries) = 0; % Make sparse output the correct size;
            empProb = reshape(full(empProb), repmat( Lmax+1, 1, p ) ); % Reshape to correct size
        end
    end
    
    methods (Access = protected)
        function [expectation, stats] = estimateExpectation_protected( o, expectFunc )
            % Sample if no samples or non-empty model (i.e. new model)
            if(isempty(o.XtSample))
                o.sample( [] );
            end

            % Normalize weights
            C = max(o.logW);
            logP = log(sum(exp(o.logW - C))) + C;
            W = exp(o.logW-logP);
            W = W/sum(W);

            % Simply iterate over samples 
            XSample = o.XtSample';
            firstSize = size(expectFunc(XSample(:,1)));
            expectation = zeros(firstSize);
            for i = 1:size(XSample, 2)
                expectation = expectation + W(i)*expectFunc(XSample(:,i));
            end

            stats = [];
        end
    end
end

