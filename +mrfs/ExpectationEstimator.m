classdef (HandleCompatible) ExpectationEstimator
    
    properties (Abstract)
        model;
    end
    
    methods (Abstract, Access = protected)
        % Only need to handle a single ep
        [expectation, stats] = estimateExpectation_protected( o, expectFunc );
    end
    
    methods (Abstract)
        setModel( o, model );
    end
    
    methods
        function [expectation, stats] = estimateExpectation( o, expectFunc )
            % Handle some common cases
            if(ischar(expectFunc))
                switch lower(expectFunc)
                    case 'mean'
                        expectFuncArray = {@(x) x};
                    case 'var'
                        expectFuncArray = {...
                            @(x) x, ...
                            @(mean) @(x) (x-mean).^2 ...
                            };
                    case 'cov'
                        expectFuncArray = {...
                            @(x) x, ...
                            @(mean) @(x) (x-mean)*(x-mean)'...
                            };
                    case 'corr'
                        expectFuncArray = {
                            @(x) x, ...
                            @(mean) @(x) (x-mean).^2, ...
                            @(mean) @(var) @(x) ((x-mean)./sqrt(var))*((x-mean)./sqrt(var))'
                            };
                    otherwise
                        error('ExpectationEstimator:UnknownExpectFuncString', 'Unknown expectFunc string constant.');
                end
            elseif(isa(expectFunc, 'function_handle'))
                expectFuncArray = {expectFunc}; % Make into a cell array of function handle
            elseif(~iscell(expectFunc))
                error('ExpectationEstimator:NotCellArray', 'expectFunc is not a function handle or a cell array of function handles');
            end
            
            % Loop through all function handles in sequence
            nFunc = length(expectFuncArray);
            resultArray = struct('expectation', cell(nFunc,1), 'stats', []);
            curFunc = expectFuncArray{1};
            for fi = 1:nFunc
                % Check to make sure each is a function handle
                if(~isa(curFunc, 'function_handle'));
                    error('ExpectationEstimator:NotFunctionHandle', 'One of the elements in the cell array is not a function handle.');
                end
                
                % Compute current expectation
                [resultArray(fi).expectation, resultArray(fi).stats] = ...
                    o.estimateExpectation_protected( curFunc );
                
                % Update new function handle based on result of previous functions
                if(fi < nFunc)
                    curFunc = expectFuncArray{fi+1};
                    for fi2 = 1:fi
                        curFunc = curFunc(resultArray(fi2).expectation);
                    end
                end
            end
            
            expectation = resultArray(end).expectation;
            stats = resultArray(end).stats;
            if(nFunc > 1)
                stats.intermediateResults = resultArray(1:(end-1));
            end
        end
    end
end

