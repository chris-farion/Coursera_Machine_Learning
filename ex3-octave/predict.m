function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias unit to Training Examples
X = [ones(m, 1) X];

% Perform first layer calculation
first_layer = sigmoid(X * Theta1');

% Add bias unit to first layer
first_layer = [ones(size(first_layer,1),1) first_layer];

%perform second/output layer calculation
second_layer = sigmoid(first_layer * Theta2');

% Set p to index of guess
[value, p] = max(second_layer, [], 2);

% =========================================================================
end
