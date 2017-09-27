% Clear all variables in the workspace and close all figures
clearvars;
close all;
addpath('helper');
addpath('data');

% Load the data for the first part
load('data/lab2dataq1b.mat');

%%% Part 1 - Plotting the data
% Get the positions that have labels 0 and 1
ind_label0 = y == 0;
ind_label1 = y == 1;

xlabel0 = X(ind_label0,:);
xlabel1 = X(ind_label1,:);

% Plot the points
plot(xlabel0(:,1), xlabel0(:,2), 'bo', xlabel1(:,1), xlabel1(:,2), ...
    'rx', 'MarkerSize', 12);
xlabel('x_1'); ylabel('x_2');
legend('Negative Class - y = 0', 'Positive Class - y = 1');

%%% To be used for later - DON'T MODIFY
xmin = min(X(:,1));
xmax = max(X(:,1));
ymin = min(X(:,2));
ymax = max(X(:,2));

%%% Part 2 -  Get the logistic regression parameters
% Define regularization parameter
% You will need to change this to answer some of the questions in this part
lambda = 0; % Fill in here yourself

% Introduce polynomial features
degree = 6;
%%% PLACE CODE FOR CREATING POLYNOMIAL FEATURES HERE
X = create_polynomial_features(X(:,1),X(:,2),degree);
[m, n] = size(X);
X = [ones(m, 1) X];

%%% FILL IN YOUR CODE HERE TO FIND PARAMETERS
initial_theta = zeros(n + 1, 1);
opt = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(lr_cost_function_reg(X, y, lambda, t)), initial_theta, opt);

%%% ENSURE THAT THE OUTPUT PARAMETERS ARE STORED IN A VARIABLE CALLED theta

%%% Part 3 - Plot the decision boundary
[xx,yy] = meshgrid(linspace(xmin, xmax), linspace(ymin, ymax));
XX = create_polynomial_features(xx(:), yy(:), degree);

XX = [ones(10000, 1) XX];

zz = reshape(XX*theta, size(xx));
hold on;
contour(xx, yy, zz, [0, 0], 'LineWidth', 2);
title(['Training examples and labels - \lambda = ' num2str(lambda)]);

%%% Part 4 - Calculate the classification accuracy
%%% PLACE CODE HERE
predictions = binary_predictor(X, theta);
accuracy= mean(double(predictions == y)) * 100;