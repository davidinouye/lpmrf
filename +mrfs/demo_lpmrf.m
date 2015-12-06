%% Load demo data
load('./+mrfs/demo_lpmrf_classic3_data.mat');

%% Split into train and test splits
testPerc = 0.1; rndSeed = 1;
[XtTrain, XtTest] = mrfs.utils.traintestsplit( Xt, testPerc, rndSeed);

%% Setup parameters (values attempted in NIPS2015 paper)
beta = 1e-4;
lamVec = logspace( log10(1), log10(0.001), 20 );
cVec = linspace( 1, 2, 5 );
k = 3;
multMaxIter = 100;
lpmrfMaxIter = 10;

%% Run Multinomial
multModel = mrfs.models.Multinomial();
multLearner = mrfs.learners.MultLearner(beta);
mrfs.utils.trainandtest( multModel, multLearner, XtTrain, XtTest );

%% Run Multinomial Mixture
multMixModel = mrfs.models.Mixture();
multMixLearner = mrfs.learners.meta.MixtureLearner( k, @mrfs.models.Multinomial, @mrfs.learners.MultLearner, {beta}, multMaxIter );
mrfs.utils.trainandtest( multMixModel, multMixLearner, XtTrain, XtTest );

%% Run Multinomial Topic Model
multTopicModel = mrfs.models.TopicModel();
% NOTE: (slight hack) Uses LPMRFTopicModelLearner but just sets thetaEdge = 0
%  (which is a Multinomial) because using MultLearner as base learner, which 
%  sets thetaEdge = 0.
multTopicModelLearner = mrfs.learners.meta.LPMRFTopicModelLearner( ...
                k, @mrfs.models.LPMRF, @mrfs.learners.MultLearner, ...
                {beta}, multMaxIter );
mrfs.utils.trainandtest( multTopicModel, multTopicModelLearner, XtTrain, XtTest );

%% Run LPMRF
lpmrfModel = mrfs.models.LPMRF();
lpmrfLearner = mrfs.learners.PoissonRegModL(lamVec(17), beta, cVec(5)); % Optimal parameter settings
mrfs.utils.trainandtest( lpmrfModel, lpmrfLearner, XtTrain, XtTest );

%% Run LPMRF Mixture
lpmrfMixModel = mrfs.models.Mixture();
lpmrfMixLearner = mrfs.learners.meta.LPMRFMixtureLearner(k, @mrfs.models.LPMRF, ...
    @mrfs.learners.PoissonRegModL, {lamVec(15), beta, cVec(2)}, lpmrfMaxIter ); % Optimal parameter settings
mrfs.utils.trainandtest( lpmrfMixModel, lpmrfMixLearner, XtTrain, XtTest );

%% Run LPMRF Topic Model
lpmrfTopicModel = mrfs.models.TopicModel();
lpmrfTopicModelLearner = mrfs.learners.meta.LPMRFTopicModelLearner( ...
                k, @mrfs.models.LPMRF, @mrfs.learners.PoissonRegModL, ...
                {lamVec(14),beta,cVec(3)}, lpmrfMaxIter ); % Optimal settings
mrfs.utils.trainandtest( lpmrfTopicModel, lpmrfTopicModelLearner, XtTrain, XtTest );