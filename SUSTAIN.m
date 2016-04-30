function result = SUSTAIN(model)
	

v2struct(model)

targets = dummyvar(assignments);

% define constants
numexemplars	= size(exemplars,1);
numfeatures		= size(exemplars,2);
numcategories	= numel(unique(assignments));

% set up storage
training = zeros(numblocks*numexemplars,numinits);

for modelnum = 1:numinits
	
	% get random presentation order
	presentationorder = shuffletrials(numexemplars,numblocks);
	
	% initalize feature tunings
	featuretuning = ones(1, numfeatures) / numfeatures;
	
	% set up initial cluster and association
	clusters	= exemplars(presentationorder(1),:);
	association	= zeros(1,numcategories);
	association(assignments(presentationorder(1))) = learningrate;
	
	
	for trialnum = 2:length(presentationorder)
		trialexemplar = exemplars(presentationorder(trialnum),:);
		trialtarget = targets(presentationorder(trialnum),:);
		correctclass = assignments(presentationorder(trialnum));		
		
		% run forward pass through the existing network
		[probabilities, classactivation, clusteroutput,distances] = ...
			FORWARDPASS(trialexemplar,clusters,featuretuning,association,...
			attentionfocus,clustercomp,decisionconsis);

		% store performance
		training(trialnum,modelnum) = probabilities(correctclass);
		
		% recruit a cluster if the response was incorrect
		if probabilities(correctclass) ~= max(probabilities)
			clusters		= cat(1,clusters,trialexemplar);
			association	= cat(1,association,zeros(1,numcategories));
			
			% recalculate pass
			[~, classactivation, clusteroutput,distances] = ...
				FORWARDPASS(trialexemplar,clusters,featuretuning,association,...
				attentionfocus,clustercomp,decisionconsis);
		end
		
		
		% update position of cluster center, association, and feature tunings
		[clusters, association, featuretuning] = ...
			UPDATE(trialtarget, trialexemplar, clusters, featuretuning, association,...
			classactivation, clusteroutput,distances,...
			learningrate);
		
	end
end

result.training = mean(blockrows(training,numexemplars*2),2);
	



end