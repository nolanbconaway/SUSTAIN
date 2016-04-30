function probabilities=RESPONSERULE(classactivation,decisionconsis)
%--------------------------------------------------------------------------
% This script simply calculates classification probabilities for a set of
% category activations, with a decision consistency parameter.
%	 
% -------------------------------------
% --INPUT ARGUMENTS		 	DESCRIPTION
% 	classactivation			observed class activations from FORWARDPASS
% 	decisionconsis			decision consistency parameter
%--------------------------------------------------------------------------

classactivation	= exp(classactivation .* decisionconsis);
probabilities	= bsxfun(@rdivide,classactivation,sum(classactivation,2));

end