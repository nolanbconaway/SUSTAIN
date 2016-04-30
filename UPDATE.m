function [clusters, association, featuretuning] = UPDATE(...
			target, stimulus, clusters, featuretuning, association,...
			classactivation, clusteroutput, distances,...
			learningrate)


% get info about winning cluster
[~,winner] = max(clusteroutput,[],2);
center = clusters(winner,:);
hout = clusteroutput(winner);
D = distances(:,:,winner);


% get change in position of cluster
deltapos = learningrate * (stimulus - center);

% get change in association
classactivation(classactivation>1) = 1;
classactivation(classactivation<0) = 0;
deltaassoc = learningrate * (target - classactivation) * hout;

% get change in tuning of receptive field
deltatuning = learningrate * exp(-featuretuning.*D).*(1-featuretuning.*D);

% execute updates
clusters(winner,:) = clusters(winner,:) + deltapos;
association(winner,:) = association(winner,:) + deltaassoc;
featuretuning = featuretuning + deltatuning;


end