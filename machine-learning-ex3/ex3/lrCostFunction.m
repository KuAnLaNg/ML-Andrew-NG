function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h=sigmoid(theta'*X');         % h=hΘ(x) 是一个m维的行向量

lambda_J_pre=sum(theta.^2)-theta(1)^2;
J_up=(log(h)*-y)-log(1-h)*[1-y]+lambda/2*lambda_J_pre;     %这里用了"1-y"这种自适应写法，“y-1"与它同原理
J=J_up/m;


b=theta;                      % 想法是用0来代替“不加第一项”的结果；不能直接令Θ的第一个元素为0，因为这样会影响迭代结果，抹去历次迭代计算出theta0的成果，但是我们可以折腾theta的替身b
b(1)=0;
theta_pre=b;
grad_up=(h-y')*X+lambda*theta_pre';
grad=grad_up'/m;








% =============================================================

grad = grad(:);

end
