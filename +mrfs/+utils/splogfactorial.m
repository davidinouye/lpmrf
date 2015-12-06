function XtGammaLn = splogfactorial( Xt )
%SPLOGFACTORIAL Faster sparse version of log factorial that only iterates over non-zero entries
%  because log(1!) = log(0!) = 0 so the solution is also sparse
Xt = max(Xt-1,0); % Remove 1s as well since they will evaluate to 0
[I,J,S] = find(Xt);
gammaLnS = gammaln(S+2);
XtGammaLn = sparse(I,J,gammaLnS, size(Xt,1), size(Xt,2)); 

end

