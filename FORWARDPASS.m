function [	probabilities, ...
			classactivation, ...
			clusteroutput,...
			distances...
			] = FORWARDPASS(...
			...
			stimuli, ...
			clusters, ...
			featuretuning, ...
			association,...
			attentionfocus,...
			clustercomp,...
			decisionconsis)
		

numstimuli		= size(stimuli,1);
numfeatures		= size(stimuli,2);
numclusters		= size(clusters,1);
numcategories	= size(association,2);

% multiply feature tunings against attentional focus param
featurefocus = featuretuning.^attentionfocus;


% compute feature-wise distance and attention-weighted activation
distances = zeros(numstimuli,numfeatures,numclusters);
clusteractivation = zeros(numstimuli,numclusters);
for S = 1:numstimuli
	for C = 1:numclusters
		featurediffs = 0.5 * abs(stimuli(S,:) - clusters(C,:));
		distances(S,:,C) = featurediffs;
		clusteractivation(S,C) =  featurefocus * exp(-1.*featuretuning.*featurediffs)' ...
			/ sum(featurefocus,2);
	end
end


% cluster output is normalized activation
clusteroutput = clusteractivation.^clustercomp;
clusteroutput = clusteroutput ./ ...
	repmat(sum(clusteroutput,2),[1 numclusters]) .* clusteroutput;


% get winning cluster for each stimulus
[~,winners] = max(clusteroutput,[],2);


% compute class activation
classactivation = zeros(numstimuli,numcategories);
for S = 1:numstimuli
	winningoutput = clusteroutput(S,winners(S));
	winningassoc = association(winners(S),:);
	classactivation(S,:) = winningoutput * winningassoc;
end

% response probabilities are normalized activations
probabilities = exp(decisionconsis*classactivation);
probabilities = probabilities ./ repmat(sum(probabilities,2),[1,numcategories]);


end