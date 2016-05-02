function [clusters, association, featuretuning] = UPDATE(...
            target, stimulus, clusters, featuretuning, association,...
            classactivation, clusteroutput, distances, winner, learningrate)
%--------------------------------------------------------------------------
% This script updates the cluster centers, association weights, and 
% feature tunings of a SUSTAIN network, given the results of a prior call
% to FORWARDPASS.m.
% 
% Future versions of this code should support lists of inputs -- the code 
% has not yet been tested on its function beyond a single training pattern.
% 
% -------------------------------------
% --INPUT ARGUMENTS         DESCRIPTION
%   target                  target (teacher) values, in range [0 1]
%   stimulus                network input used in FORWARDPASS.m
%   clusters                stored cluster centers
%   featuretuning           feature-wise tuning strengths
%   association             cluster -> category weights
%   classactivation         category activations from FORWARDPASS.m
%   clusteroutput           cluster activations from FORWARDPASS.m
%   distances               3D matrix of differences from FORWARDPASS.m
%   winner                  index of the winning cluster from FORWARDPASS.m
%   learningrate            learning rate parameter
%--------------------------------------------------------------------------

% get info about winning cluster
winnercenter    = clusters(winner,:);
winnerout       = clusteroutput(winner);
winnerdiffs     = distances(:,:,winner);

% compute change in position of cluster
deltapos =  (stimulus - winnercenter);

% compute change in association, using 'humble teachers'
classactivation(classactivation>1) = 1;
classactivation(classactivation<0) = 0;
deltaassoc = (target - classactivation) * winnerout;

% compute change in tuning of receptive field
deltatuning = exp(-featuretuning.*winnerdiffs) .* (1-featuretuning.*winnerdiffs);

% execute updates
clusters(winner,:)     = clusters(winner,:) +  learningrate * deltapos;
association(winner,:)  = association(winner,:) + learningrate * deltaassoc;
featuretuning          = featuretuning + learningrate * deltatuning;

end