function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Forward Propogation with Gradient Computation
a1 = X;
a1 = [ones(size(a1,1),1) a1];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

hx = a3;

% Create matrix with y as only 1 and 0 for classification
Y = [];

for k=1:num_labels
  y_vector = [];
  for i = 1:size(y,1)
    if y(i)==k
      y_vector = [y_vector;1];
    else
      y_vector = [y_vector;0];
    endif
  endfor
  Y = [Y y_vector];
endfor
clear y_vector; clear i;

J_sum = 0;

for k = 1:num_labels
  yt = Y(:,k);
  ht = hx(:,k);
  J_sum = J_sum + (-yt' * log(ht)-(1-yt')*log(1-ht));
endfor
hyp = J_sum/m;

t1 = Theta1(:,2:end)(:); % Roll out and do not include the bias term
t2 = Theta2(:,2:end)(:); % Roll out and do not include the bias term

r1 = t1'*t1;
r2 = t2'*t2;

reg = (lambda/(2*m)).*(r1+r2);

J = hyp + reg;

% -------------------------------------------------------------

DELTA1 = 0;
DELTA2 = 0;
z2 = [ones(size(z2,1),1) z2];

for t = 1:m
  a3 = hx(t,:);
  yt = Y(t,:);

  delta3 = a3-yt;

  delta2 = (delta3 * Theta2).*sigmoidGradient(z2(t,:));
  delta2 = delta2(:,2:end);

  DELTA2 = DELTA2 + delta3' * a2(t,:);
  DELTA1 = DELTA1 + delta2' * a1(t,:);
endfor

Theta1_grad = (DELTA1 / m) + (lambda/m) * Theta1;
Theta2_grad = (DELTA2 / m) + (lambda/m) * Theta2;

Theta1_grad(:,1) = (DELTA1(:,1) / m);
Theta2_grad(:,1) = (DELTA2(:,1) / m);
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
