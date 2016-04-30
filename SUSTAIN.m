function result = SUSTAIN(model)
% ------------------------------------------------------------------------
% This trains the SUSTAIN network given the design specification in the 
% sole argument: model. This argument contains critical information about 
% the model, and must include:
% 
%      exemplars: coordinates for the training stimuli
%      numblocks: number of times to iterate over the training set
%      numinitials: number of random initializations
%      assignments: integer class assignment for each exemplar
%      params: [attn, comp, decision, lrate]
% 
% The model struct may contain more fields (such as test items), but those
% are the necessary fields.
% 
% The sole output, result, is a struct containing the following:
%      training: accuracy for each block,(averaged across presentation orders)
% ------------------------------------------------------------------------

% unpack input struct
v2struct(model)
rng('shuffle') %get random seed


% define constants
% --------------------------------
targets         = dummyvar(assignments);
numexemplars    = size(exemplars,1);
numfeatures     = size(exemplars,2);
numcategories   = numel(unique(assignments));

% unpack parameters
attentionfocus  = params(1);
clustercomp     = params(2);
decisionconsis  = params(3);
learningrate    = params(4);

%-----------------------------------------------------------%
% iterate over presentation orders
training = zeros(numblocks*numexemplars,numinits);
training(1,:) = 1/numcategories; % first trial is chance
for modelnum = 1:numinits
    
    % get random presentation order
    presentationorder = shuffletrials(numexemplars,numblocks);
    
    % initialize feature tunings
    featuretuning = ones(1, numfeatures) * numexemplars;
    
    % set up first-trial cluster and association
    clusters    = exemplars(presentationorder(1),:);
    association = zeros(1,numcategories);
    association(assignments(presentationorder(1))) = learningrate;

    %  iterate over trials
    for trialnum = 2:length(presentationorder)
        trialexemplar   = exemplars(presentationorder(trialnum),:);
        trialtarget     = targets(presentationorder(trialnum),:);
        correctclass    = assignments(presentationorder(trialnum));        
        
        % pass activations through the network
        %--------------------------------------------------------------
        [classactivation, clusteroutput, distances] = FORWARDPASS(...
            trialexemplar, clusters, featuretuning, association,...
            attentionfocus, clustercomp);
        
        % compute response probabilities and store performance
        probabilities = RESPONSERULE(classactivation, decisionconsis);
        training(trialnum,modelnum) = probabilities(correctclass);
        
        
        % recruit a cluster if the response was incorrect
        %--------------------------------------------------------------
        if classactivation(correctclass) ~= max(classactivation)
            clusters     = cat(1, clusters, trialexemplar);
            association  = cat(1 ,association, zeros(1,numcategories));
            
            % recalculate forward pass
            [classactivation, clusteroutput, distances] = FORWARDPASS(...
                trialexemplar, clusters, featuretuning, association, ...
                attentionfocus, clustercomp);
        end
        
        % update cluster center, association, and feature tunings
        %-------------------------------------------------------------
        [clusters, association, featuretuning] = UPDATE(...
            trialtarget, trialexemplar, clusters, featuretuning, association,...
            classactivation, clusteroutput, distances, learningrate);
        
    end
    
%     % SAMPLE TEST PHASE CODE 
%     TEST_OUT = FORWARDPASS(exemplars, clusters, featuretuning, association,...
%             attentionfocus, clustercomp);
%     TEST_PS = RESPONSERULE(TEST_OUT, decisionconsis);
%     % ----------------------
    
end

% aggregate accuracy across inits and within blocks
result = struct;
result.training = mean(blockrows(training,numexemplars),2);
    

end