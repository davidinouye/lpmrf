function label = litekmeans(X, k)
% Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
% Modified by David Inouye to fix compatability with new MATLAB versions
n = size(X,2);
last = 0;
label = ceil(k*rand(1,n));  % random initialization
while any(label ~= last)
    [u,~,label] = unique(label);   % remove empty clusters
    label = label(:)'; % For compatability with old and new UNIQUE function
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
end
[~,~,label] = unique(label);
label = label(:)'; % For compatability with old and new UNIQUE function
