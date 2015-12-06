classdef MRF < mrfs.models.IndependentModel
    properties
        thetaEdge; % Interaction term
        isSymmetric = true;
        useAnd = false;
    end
    
    methods
        function set.thetaEdge( o, input )
            o.thetaEdge = input;
            o.clearCache();
        end
        
        function validate(o)
            % Check for symmetry
            if(o.isSymmetric)
                % Use AND of edges if specified
                if(o.useAnd)
                    o.thetaEdge = mrfs.models.MRF.andedges(o.thetaEdge);
                end
                % Make symmetric
                o.thetaEdge = (o.thetaEdge + o.thetaEdge')/2;
            end
        end
        
        % Display the model in simple 
        function stats = summary(o, metadata)
            % Backwards compatibilty with dispsummary
            thetaNodeArray = {o.thetaNode};
            thetaEdgesArray = {o.thetaEdge};
            words = o.labels;
            topN = min(length(words), 10);
            
            % Code from dispsummary
            k = length(thetaNodeArray);
            summary = cell(k*(topN+1)+1, 6);
            summary(1, :) = {'NodeWgt', 'TopNodes', 'EdgeWgt', 'Top+Edges', 'EdgeWgt', 'Top-Edges'};
            for j = 1:k
                offset = (j-1)*(topN+1)+1;
                summary(offset+1, :) = {sprintf('Topic %d',j), '', '', '', '', ''};
                [sortedNode, sortedNodeI] = sort(thetaNodeArray{j}, 'descend');

                edges = (thetaEdgesArray{j} + thetaEdgesArray{j}')/2;
                edges = edges - triu(edges); % Remove duplicate edges
                [sortedValues, sortedValuesI]= sort(edges(:));
                minValues = sortedValues;
                maxValues = flipud(sortedValues);
                [min1, min2] = ind2sub(size(edges), sortedValuesI);
                [max1, max2] = ind2sub(size(edges), flipud(sortedValuesI));

                offset = offset + 1;
                for topI = 1:topN
                    % Weight, node, weight, +edge, weight -edge
                    summary{offset+topI, 1} = sortedNode(topI);
                    summary{offset+topI, 2} = words{sortedNodeI(topI)};
                    summary{offset+topI, 3} = maxValues(topI);
                    summary{offset+topI, 4} = sprintf('%s+%s', words{max1(topI)}, words{max2(topI)});
                    summary{offset+topI, 5} = minValues(topI);
                    summary{offset+topI, 6} = sprintf('%s-%s', words{min1(topI)}, words{min2(topI)});
                end
            end
            %disp(summary);
            sumTex = summary(:,[2,4,6]);
            [nRows, nCols] = size(sumTex);
            %fprintf('\\begin{tabular}{%s}\n', repmat('c',1,nCols));
            for r = 1:nRows
                for c = 1:nCols
                    if(c==nCols)
                        fprintf('%s,',sumTex{r,c});
                    else
                        fprintf('%s,',sumTex{r,c});
                    end
                end
                if(r ~= nRows)
                    %fprintf(' \\\\\n');
                else
                    %fprintf('\n');
                end
                fprintf('\n');
            end
            %fprintf('\\end{tabular}\n');
        end
    end
    
    methods (Access = protected)
        function str = getArgString( o )
            str = sprintf('nnz=%d,nnzwords=%d', nnz(o.thetaEdge), sum(full(sum(o.thetaEdge))>0)); % Usually not many arguments for models
        end
    end
    
    methods (Static)
        function thetaEdge = andedges(thetaEdge)
            andMask = (thetaEdge ~= 0) & (thetaEdge ~= 0)';
            temp = double(andMask);
            temp(andMask) = thetaEdge(andMask);
            thetaEdge = temp;
        end
    end
end

