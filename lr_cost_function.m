%%% lr_cost_function
%
% Function that computes the cost and gradient vector for logistic
% regression that is to be used with MATLAB's optimization toolbox
%
% Inputs:
%  x - Data matrix of points of size m x (n + 1)
%      Each row is a point and each column is a feature
%      Note: If using the linear algebra approach, the FIRST column is
%      assumed to be 1
%      Make sure you prepend a column of 1s to the first column
%      before running
%  y - Column vector of expected classes for each instance
%  theta - An input set of parameters used to evaluate the cost and the
%          gradient vector
%
%  Outputs:
%  cost_val - The cost incurred when using the parameters theta to denote
%             the decision boundary separating the two classes. Single
%             value.
%  grad     - A (n + 1) x 1 column vector that denotes the derivative with
%             respect to each parameter evaluated at the input theta

function [cost_val, grad] = lr_cost_function(X, y, theta)
    % Get the total number of examples
    m = numel(y);

    %%% FILL IN REST OF CODE HERE
    grad = zeros(size(theta));
    w = sigmoid(X*theta);
    
    % Cost
    cost_val = (-1/m)*sum(y.*log(w) + (1-y).*log(1 - w));
    
    % Gradient 
    for i = 1:m
            grad = grad + (w(i)-y(i))* X(i, :)';
    end
    
    grad = (1/m)*grad;
end