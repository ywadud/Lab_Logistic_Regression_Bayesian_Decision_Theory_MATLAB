%%% one_vs_all
%
% Function that computes the matrix of logistic regression parameters where
% each column i denotes the parameters assuming that examples with class i
% are positive while the rest are negative
%
% Inputs:
%  x - Data matrix of points of size m x (n + 1)
%      Each row is a point and each column is a feature
%      Note: If using the linear algebra approach, the FIRST column is
%      assumed to be 1
%      Make sure you prepend a column of 1s to the first column
%      before running
%  y - Column vector of expected classes for each instance
%  num_classes - Expected number of classes to be seen in the data matrix
%  lambda - The regularization parameter to prevent overfitting
%
%  Outputs:
%  cost_val - The cost incurred when using the parameters theta to denote
%             the decision boundary separating the two classes. Single
%             value.
%  grad     - A (n + 1) x 1 column vector that denotes the derivative with
%             respect to each parameter evaluated at the input theta

function Theta = one_vs_all(X, y, num_classes, lambda)
% Get total number of parameters
    n = size(X,2);

    % Initialize matrix of parameters
    Theta = zeros(n,num_classes);

    % Set up optimization options
    options = optimset('GradObj', 'on', 'MaxIter', 100);

    %%% FILL IN THE REST OF THE CODE HERE

    for i=1:num_classes

        y_c=(y==(i-1));

       initial_theta=zeros(n,1); 
       func = @(theta) lr_cost_function_reg(X, y_c,lambda, theta);

       theta=fmincg(func, initial_theta, options);
       Theta(:,i)=theta;
    end
end