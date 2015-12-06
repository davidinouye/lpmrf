classdef MultLearner < mrfs.learners.Learner
    %MULTINOMIAL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        beta;
    end
    
    methods
        function o = MultLearner( beta )
            if(nargin < 1); beta = 1e-4; end
            o.beta = beta;
        end
        
        function [params, stats] = learn(o, Xt, metadata)
            % Super simple Multinomial fitting
            [~,p] = size(Xt);
            Lmean = full(mean(sum(Xt,2)));
            %prob = full(mean(Xt + o.beta)');
            prob = full(mean(Xt)') + o.beta;
            prob = prob/sum(prob);
            thetaNode = log(prob);
            thetaEdge = sparse(p,p);
            
            % Set final parameters
            params.thetaNode = thetaNode;
            params.thetaEdge = thetaEdge;
            params.Lmean = Lmean;
            params.modLFunc = @(Lvec) ones(size(Lvec)); % Just hack so that it can train LPMRF model
            stats = [];
            stats.prob = prob;
        end
    end
    
    methods (Access = protected)
        function str = getArgString( o )
            str = sprintf('%g', o.beta);
        end
    end
end

