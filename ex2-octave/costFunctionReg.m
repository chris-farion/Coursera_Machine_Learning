function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost Function
h_of_theta = sigmoid(X * theta);

theta(1)=0;

hypothesis = (-y'*log(h_of_theta)-(1-y)'*log(1-h_of_theta))/m;
regularization_term = (lambda/(2*m))*(theta'*theta);

J = hypothesis+regularization_term;


grad = ((h_of_theta-y)'*X)/m+(lambda/m)*theta';
grad(1) = (((h_of_theta-y)'*X)/m)(1);
% =============================================================

end
