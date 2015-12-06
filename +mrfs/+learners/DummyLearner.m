classdef DummyLearner < mrfs.learners.Learner
    %MULTINOMIAL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        type;
        p;
        ops;
    end
    
    methods
        function o = DummyLearner(type, p, ops)
            o.type = type;
            o.p = p;
            o.ops = ops;
        end
        
        function [params, stats] = learn(o, Xt, metadata)
            switch o.type
                case 'single-dep'
                    dep = o.ops;
                    thetaNode = log(ones(o.p,1)/o.p); % Equal probability
                    thetaEdge = sparse(o.p, o.p);
                    thetaEdge(2,1) = dep/2;
                    thetaEdge(1,2) = dep/2;
                case 'sprandn'
                    thetaNode = randn(o.p, 1);
                    thetaEdge = sprandn(o.p, o.p, o.ops);
                otherwise
                    thetaNode = randn(o.p, 1);
                    thetaEdge = randn(o.p, o.p);
            end
            Lmean = o.p;
            
            % Set final parameters
            params.thetaNode = thetaNode;
            params.thetaEdge = thetaEdge;
            params.Lmean = Lmean;
            params.modLFunc = @(Lvec) ones(size(Lvec));
            stats = [];
        end
    end
    
    methods (Access = protected)
        function str = getArgString(o)
            str = sprintf('%s,p=%d,ops=%s', o.type, o.p, o.ops);
        end
    end
end

