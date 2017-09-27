%%% binary_predictor
%
% Function that performs binary classification given a data matrix and
% associated logistic regression parameters
%
% Inputs:
%  X - Data matrix of points of size m x (n + 1)
%      Each row is a point and each column is a feature
%      Note: If using the linear algebra approach, the FIRST column is
%      assumed to be 1
%      Make sure you prepend a column of 1s to the first column
%      before running
%
% theta - A (n + 1) vector of parameters learned from your training
%
% Outputs:
% out - A m x 1 output vector that determines the output class for each row
% / instance of the data matrix X

function out = binary_predictor(X, theta)
    %%% FILL IN YOUR CODE HERE
    m = size(X, 1); 
    out = zeros(m, 1);
    out = (sigmoid(X*theta) >= 0.5);
end