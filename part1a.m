% Clear all variables in the workspace and close all figures
clearvars;
close all;
addpath('helper');
addpath('data');

% Load the data for the first part
load('data//lab2dataq1a.mat');

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

%%% Part 2 -  Get the logistic regression parameters
%%% FILL IN YOUR CODE HERE
%%% ENSURE THAT THE OUTPUT PARAMETERS ARE STORED IN A VARIABLED CALLED theta
[m, n] = size(X);
X = [ones(m, 1) X];
init_theta = zeros(n + 1, 1);

opt = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(lr_cost_function(X, y, t)), init_theta, opt);

%%% Part 3 - Plot the line
% theta_0 + theta_1 x_1 + theta_2 x_2
% x_2 = -theta_0/theta_2 - theta_1/theta_2
xx = linspace(xmin, xmax);
yy = -theta(1)/theta(3) - theta(2)*xx/theta(3);
hold on;
plot(xx, yy, 'k');

%%% Part 4 - Calculate the classification accuracy
%%% PLACE CODE HERE
predictions = binary_predictor(X, theta);
accuracy = mean(double(predictions == y)) * 100;

