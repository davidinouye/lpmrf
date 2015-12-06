function [model, learner, trainStats, evalTrainStats, evalTestStats] ...
        = trainandtest( model, learner, XtTrain, XtTest, trainMetadata, testMetadata, verbosity )
    % NOTE: nTest = number of test L for estimating the log partition function
    %       nSamples = number of AIS samples at a particular L
    %       Thus, the total number of samples is nSamples*nTest
    if(nargin < 5); trainMetadata = struct( 'nSamples', 100, 'nTest', 50 ); end
    if(nargin < 6); testMetadata = struct( 'nSamples', 100, 'nTest', 50 ); end
    if(nargin < 7); verbosity = 2; end
    
    % Setup
    model.verbosity = verbosity;
    learner.verbosity = verbosity;
    tTotal = tic;

    % Train model
    fprintf('\n\n');
    fprintf( '<<<<< TrainStart: %s:%s >>>>>\n', model.name(), learner.name());
    tStart = tic;
    trainStats = model.train( XtTrain, learner, trainMetadata );
    fprintf( '<<<<< Train  End: %s:%s in %g s >>>>>\n', model.name(), learner.name(), toc(tStart) );

    % Test model
    evaluator = mrfs.evaluators.LogLikeEvaluator();
    fprintf( '<<<<< Test Start: %s:%s >>>>>\n', model.name(), learner.name());
    tStart = tic;
    [trainLogL, evalTrainStats] = model.test( XtTrain, evaluator, testMetadata );
    [testLogL, evalTestStats] = model.test( XtTest, evaluator, testMetadata );
    fprintf( '<<<<< Test   End: %s:%s in %g s >>>>>\n', model.name(), learner.name(), toc(tStart) );

    % Output results
    timeTotal = toc(tTotal);
    fprintf( '<<<<< Completed: %s:%s in a total of %g s >>>>>\n', model.name(), learner.name(), timeTotal);
    fprintf( 'Training Set Perplexity = %g, Training Set Log Likelihood = %g\n', evalTrainStats.perplexity, trainLogL);
    fprintf( 'Test Set Perplexity = %g, Test Set Log Likelihood = %g\n', evalTestStats.perplexity, testLogL);
    fprintf('\n\n');
end