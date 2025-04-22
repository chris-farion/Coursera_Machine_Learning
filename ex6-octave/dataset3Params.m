function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

options = [0.01,0.03,0.1,0.3,1,3,10,30];
number_of_options = size(options)(2);

results = [];

%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

%row and column for combinations of C and sigma for predicted results
row = [];
col = [];
% Default the current best prediction to a bad value (0xFF)
best = 255;

for C_temp = 1:number_of_options
  for sigma_temp = 1:number_of_options
    %Uncomment if the values of C and sigma want to be known
    %fprintf('Training with C:%f and sigma: %f',options(C_temp),options(sigma_temp));
    model = svmTrain(X, y, options(C_temp), @(X, Xval) gaussianKernel(X, Xval, options(sigma_temp)));
    predictions = svmPredict(model,Xval);
    train2cross_comparison = double(predictions~=yval);
    train2_cross_mean = mean(train2cross_comparison);
    results = [results train2_cross_mean];
    if train2_cross_mean < best % If there is a better prefiction, save it
      row = C_temp;
      col = sigma_temp;
      best = train2_cross_mean;
    elseif train2_cross_mean == best % In case the best has more than 1 option
      row = [row; C_temp];
      col = [col; sigma_temp];
    endif
  endfor
  results = [results;];
endfor

% Best mean of the train2cross mean
best_mean_result = min(results);

% Return row (C) and col (sigma) of the best results
C = options(row);
sigma = options(col);
% =========================================================================
end
