% Clear all variables, close all figures and add the helper directory to
% MATLAB's system path
clearvars;
close all
addpath('helper');

%%% Part 1 
% Load in digits
load('data/lab2digits.mat');

%%% Part 2
% Reshape the 3D matrix of training digits so that each row is a training 
% examples
[rows,cols,numImages] = size(trainImages);
Xtrain = reshape(trainImages, rows*cols, numImages).';
Xtrain = [ones(numImages,1) Xtrain];
show_digits(trainImages,144)

%%% Part 3
% Perform training for each digit
lambda = 1; % Define regularization parameter

% FILL IN YOUR CODE HERE
classes = 10;
theta=one_vs_all(Xtrain, trainLabels, classes, lambda);

%%% Part 4
% Find predicted labels for training set
% FILL IN YOUR CODE HERE
trainPred = multiclass_predict(Xtrain, theta);

%%% Part 5
% Reshape the 3D matrix of test digits so that each row is a test
% example
numTestImages = size(testImages, 3);
Xtest = reshape(testImages, rows*cols, numTestImages).';
Xtest = [ones(numTestImages,1) Xtest];

%%% Part 6
% Find predicted labels for the test set
% FILL IN YOUR CODE HERE
testPred = multiclass_predict(Xtest, theta);

%%% Part 7
% Find the classification accuracy for the training and test data set
% FILL IN YOUR CODE HERE
mean((trainPred == trainLabels)*100)
mean((testPred == testLabels)*100)

%%% Part 8
% Show some misclassified images - 9 for each digit
% FILL IN YOUR CODE HERE
for digit = 0:9 
    show_misclassified_digits(testImages, testPred, testLabels, digit, 9);
end
