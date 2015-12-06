classdef ProbabilityModel < mrfs.VerboseHandler
    %PROBABILISTICMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        p = 0; % Dimensionality of the data
        labels = {}; % Feature labels (e.g. dictionary of words when modeling text)
    end
    
    %% Core methods
    methods
        
        % Trains the model
        function stats = train(o, Xt, learner, metadata)
            [n,o.p] = size(Xt);
            %o.message(2, sprintf('Training model %s using %s learner with training dataset n=%d and p=%d',...
            %        class(o), class(learner), n, o.p));
            if(isfield(metadata, 'labels'))
                o.labels = metadata.labels;
            elseif(isfield(metadata, 'words'))
                o.labels = metadata.words;
            end
        end
        
        function clearCache( o )
            % Empty since no cache
        end
        
        % Tests the model
        function [val, stats] = test(o, Xt, evaluator, metadata)
            [val, stats] = evaluator.evaluate( Xt, o, metadata );
        end
        
        % Log proportion of instances (i.e. without normalizing constants)
        function [logPropXt, stats] = logProportion(o, Xt, metadata)
            stats = [];
            error('Unimplemented method');
        end
        
        % Log-Likelihood
        function [logL, stats] = logLikelihood(o, Xt, metadata)
            stats = [];
            error('Unimplemented method');
        end
        
        % Simple accessor functions to get other stats
        function [perplexity, stats] = perplexity(o, Xt, metadata)
            if(nargin < 4); metadata = []; end
            [~, stats] = o.logLikelihood( Xt, metadata);
            perplexity = stats.perplexity;
        end
        
        % Validate the parameters of the model
        function validate(o)
            error('Unimplemented method');
        end
        
    end
    
    %% Utility methods
    methods (Access = protected)
        function str = getArgString( o )
            str = ''; % Usually not many arguments for models
        end
    end
    
    
end