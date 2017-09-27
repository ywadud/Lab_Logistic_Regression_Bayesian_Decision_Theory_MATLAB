%%% multiclass_predict
%
% Function that performs one-vs-all classification given a data matrix
% and a matrix of parameters given by the one_vs_all MATLAB function
%
% Inputs:
%  X - Data matrix of points of size m x (n + 1)
%      Each row is a point and each column is a feature
%      Note: If using the linear algebra approach, the FIRST column is
%      assumed to be 1
%      Make sure you prepend a column of 1s to the first column
%      before running
%
% Theta - A (n + 1) x N matrix where N is the total number of classes
% expected in the data. Each column i of this matrix represents the
% parameters for the logistic regression binary classifier. The ith column
% considered instance from class i to be positive while the rest were
% negative.
%
% Outputs:
% classes - A m x 1 output vector that determines the output class for each
% row / instance of the data matrix X

function classes = multiclass_predict(X, Theta)
%%% FILL IN CODE HERE
    h = sigmoid(X*Theta);
    [~,classes] = max(h,[],2);
    classes = classes-1;
end