function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);              %X是5000*400的矩阵，所以第一个维度：行数的值是5000
n = size(X, 2);              %X是5000*400的矩阵，所以第二个维度：列数的值是400

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);       %num_labels=10，所以是10个分类器，每个的θ值都分布在一行，所以共10行

% Add ones to the X data matrix
X = [ones(m, 1) X];                         %现在X是5000*401的矩阵

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.           要求是训练“标签个数”数值大小的带正则化的逻辑分类器
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you     将1-10标签的数值变为"y==c,1"与“与y!=c,0”两种
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.    建议用循环训练这10个分类器
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
for c=1:num_labels
 initial_theta = zeros(n + 1, 1);              %每个分类器的θ需先设为成列向量
 options = optimset('GradObj', 'on', 'MaxIter', 50);
 [theta] =fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta, options);    %没训练一个分类器都需要全部样本；且最后只需要返回最优theta值即可，所以只写了[theta]
 all_theta(c,:)=theta';
end








% =========================================================================


end
