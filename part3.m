% Clear all variables, close all figures and add the helper directory to
% MATLAB's system path
clearvars;
close all
addpath('helper');

% Load in data
load('data/lab2flowers.mat');

% Preliminaries - Extract out the Sepal Width then separate into the
% different classes 
% Class 1 is the Iris Setosa and Class 2 is the Iris Versicolour
%%% FILL IN YOUR CODE HERE
sepal_width= X(:,2);

% 1. Compute the prior probabilities for the class 1 and class 2
%%% FILL IN YOUR CODE HERE
Prob_setosa = sum(y == 1)/size(y(1:100,1),1);
Prob_vers = sum(y == 2)/size(y(1:100,1),1);

% 2. Compute the mean and variance for the Sepal Width for each class
%%% FILL IN YOUR CODE HERE
Mean_setosa=mean(sepal_width(1:51,1));
Var_setosa=var(sepal_width(1:51,1));
 
Mean_vers = mean(sepal_width(51:100,1));
Var_vers = var(sepal_width(51:100,1));

% 3. Determine the decision boundary that separates between the two classes
%%% FILL IN YOUR CODE HERE
Boundary = calculate_decision_boundary(Mean_setosa, Var_setosa, Prob_setosa, Mean_vers, Var_vers, Prob_vers);

% 4. Using the decision boundary, use this to classify between the two 
% classes for new inputs.
x = [1, 2, 3, 4, 5]; % Define new inputs (in cm)
%%% FILL IN YOUR CODE HERE
output = Boundary(x);

% 5. Calculate the classification accuracy.
%%% FILL IN YOUR CODE HERE
predictions = Boundary(sepal_width(1:101,1));
accuracy= mean((predictions == y(1:101,1))) * 100;
