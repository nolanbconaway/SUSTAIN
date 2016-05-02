function [classactivation, clusteroutput, distances, winners] = FORWARDPASS(...
            stimuli, clusters, featuretuning, association, attentionfocus, clustercomp)
%--------------------------------------------------------------------------
% This script runs a forward pass in SUSTAIN and returns information
% about network performance.
% 
% -------------------------------------
% --INPUT ARGUMENTS      DESCRIPTION
%   stimuli              items to be passed through the model
%   clusters             coordinates of each stored cluster
%   featuretuning        feature-wise tuning strengths
%   association          cluster -> category weights
%   attentionfocus       attention focus parameter
%   clustercomp          cluster competition parameter

% -------------------------------------
% --OUTPUT ARGUMENTS     DESCRIPTION
%   classactivation      category output activations
%   clusteroutput        cluster activations [after competition]
%   distances            3D matrix of stimuli -> cluster differences
%   winners              index of winning cluster for each stimulus
%--------------------------------------------------------------------------

% define constants
numstimuli       = size(stimuli,1);
numcategories    = size(association,2);

% multiply feature tunings against attentional focus param
tuning2focus = featuretuning.^attentionfocus;
sumtuning2focus = sum(tuning2focus,2);

% compute feature-wise distance, tile values for future multiplication
distances = 0.5 * abs(bsxfun(@minus, stimuli, permute(clusters, [3 2 1])));
tuned_distances = exp( -bsxfun(@times ,featuretuning, distances));

% compute cluster activations
clusteractivation = sum( bsxfun( @times, tuning2focus, tuned_distances), 2);
clusteractivation = permute(clusteractivation,[1,3,2]) / sumtuning2focus;

% compute cluster output as normalized activation
clusteroutput = clusteractivation.^clustercomp;
clusteroutput = bsxfun(@rdivide,clusteroutput,sum(clusteroutput,2)) .* clusteractivation;

% determine winning cluster and class activation
%	[vectorize this one day]
winners          = zeros(numstimuli,1);
classactivation  = zeros(numstimuli,numcategories);
for S = 1:numstimuli
	
	% determine winning clusters
	options = find(clusteractivation(S,:) == max(clusteractivation(S,:)));
	winner = options(randperm(length(options),1));
	
	% compute class activation
	winners(S) = winner;
	classactivation(S,:) = clusteroutput(S,winner) * association(winner,:);
end

end