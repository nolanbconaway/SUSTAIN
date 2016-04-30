close all;clear;clc
addpath utils


% load inputs and class assignments
load shj

% set up model
model = struct;
  model.exemplars	= stimuli;	% training stimuli
  model.numblocks	= 32;	% number of iterations over the exemplars
  model.numinits	= 1;	% number of random initializations
  model.params		= [9.01, 1.25, 16.9, 0.09];	% [attn, comp, decision, lrate];
	
training = zeros(model.numblocks,6);
for i = 1:6
	model.assignments = assignments(:,i);
	result = SUSTAIN(model);
	training(:,i) = result.training;
	
end

disp(training)
