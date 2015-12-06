classdef PMRF < mrfs.models.MRF
    methods
        function stats = train(o, Xt, learner, metadata)
            % Initialize p and words (optionally display message)
            if(nargin < 4); metadata = struct(); end
            train@mrfs.models.ProbabilityModel(o, Xt, learner, metadata);
            
            % Train model
            [params, stats] = learner.learn(Xt, metadata);
            
            % Update parameters
            o.thetaNode = params.thetaNode;
            o.thetaEdge = params.thetaEdge;
            
            o.validate();
        end
        
        function validate(o)
            % Reduce to only negative edge weights
            posBool = (o.thetaEdge > 0);
            if(nnz(posBool) > 0)
                o.message(2,'PMRF: Found positive edge weights for PMRF model but the PMRF model cannot have positive weights (See TPMRF or SPMRF instead)');
                o.message(2,'Converting positive weights to 0.');
                o.thetaEdge(posBool) = 0;
            end
            
            % Check that diagonal is 0
            o.validateDiagonal();
            
            % Check for symmetric and AND of edges if requested
            validate@mrfs.models.MRF(o);
        end
        
        function [logPropXt, stats] = logProportion(o, Xt, metadata)
            if(nargin < 3); metadata = []; end
            if(~isfield(metadata, 'XtBaseMeasure'))
                metadata.XtBaseMeasure = mrfs.utils.poissonbasemeasure( Xt );
            end
            [logPropXt, stats] = o.logProp(Xt, metadata);
        end
       
    end
    
    methods (Access = protected)
        
        % Internal function for simplicity after XtBaseMeasure is calculated
        function [logPropXt, stats] = logProp(o, Xt, metadata)
            logPropXt = Xt*o.thetaNode + sum((Xt*o.thetaEdge).*Xt,2) + metadata.XtBaseMeasure;
            stats.XtBaseMeasure = metadata.XtBaseMeasure;
        end
        
        function validateDiagonal(o)
            % Check that diagonal is 0
            diagEdge = diag(o.thetaEdge);
            if(any(diagEdge ~= 0))
                o.message(2, 'Some diagonal entries thetaEdge were non-zero.');
                o.message(2, 'Converting diagonal entries to 0.');
                o.thetaEdge(logical(speye(size(o.thetaEdge)))) = 0;
            end
        end
    end
    
end

