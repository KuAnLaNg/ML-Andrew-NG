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

end
