close all;clear;clc
addpath utils


% load inputs and class assignments
load shj
stimuli(stimuli==-1) = 0;

model = struct;
	
% network parameters
	model.attentionfocus	= 9.01;
	model.clustercomp		= 1.25;
	model.decisionconsis	= 16.9;
	model.learningrate		= 0.09;
	model.numinits			= 100;
	
% set up categorization problem
	model.exemplars			= stimuli;
	model.numblocks			= 32;	
	
training = zeros(model.numblocks/2,6);
for i = 1:6
	model.assignments = assignments(:,i);
	result = SUSTAIN(model);
	training(:,i) = result.training;
	
end

figure
fsize=12;
pdfsize = [1.8 1]*230;
hold on
for  i = 1:6
	plot(1-training(:,i),'k--')
	text(1:model.numblocks/2,1-training(:,i),num2str(i),'horizontalalignment','center',...
		'fontsize',fsize+2,'fontname','courier')
end

box on
axis([0.5 model.numblocks/2+0.5 -0.025 0.55])
set(gca,'ytick',0:0.1:1,'ygrid','on','fontsize',fsize,'xtick',2:2:model.numblocks)
xlabel('Training Block')
ylabel('Error')

set(gcf, 'PaperUnits','points', 'PaperPosition',[0 0 pdfsize],...
		'papersize',[pdfsize],'position',[500 500 pdfsize],'color','w')
export_fig shj.pdf