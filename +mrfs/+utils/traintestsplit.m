function [XtTrain, XtTest, finalPerm, revFinalPerm, idxTrain, idxTest] = traintestsplit( Xt, testPerc, rndSeed )
if(nargin < 2); testPerc = 0.1; end
if(nargin < 3); rndSeed = []; end
if(~isempty(rndSeed))
    rng(rndSeed);
end

% Determine split index
[n, ~] = size(Xt);
permutation = randperm( n );
splitIdx = round( testPerc*n );

% Final permutation and reverse to reorder at last step
finalPerm = [permutation((splitIdx+1):end), permutation(1:splitIdx)];
revFinalPerm = zeros(size(finalPerm));
revFinalPerm(finalPerm) = 1:length(finalPerm);

% Create ZtTune and reset Zt
idxTest = permutation(1:splitIdx);
XtTest = Xt(idxTest,:);
idxTrain = permutation((splitIdx+1):end);
XtTrain = Xt(idxTrain,:);

end

