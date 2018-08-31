function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1).                                  这里K是10,因为all_theta里面有10个分类器的参数
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions               对X中每个数据样本都返回一个值，这些值最后构成一个向量形式的结果
%  for each example in the matrix X. Note that X contains the examples in              ***X中每个样本是以**行向量**的形式存在的***
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 
% 很明显，需要我们对每一个样本得到的10个分类器结果做出最大值的选择，并将最大值所在行数作为标签结果返回，充当P中的一个元素，并最终构成P向量；
% 因为我们实际上是用y==c这种只由“1”与“0”构成的向量做的标签，所以理论上一个属于第i行的数据会在第i行形成接近1的值，而其它9行的结果会趋于0
m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);                                                             %每个样本一个结果，这个结果就是所在类的标签值

% Add ones to the X data matrix
X = [ones(m, 1) X];                                                                   %现在X是5000*401的矩阵

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.                %因为一个样本对于10个分类器是10个结果，而我们要选最大的那个
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. 

result_in_array=X*all_theta';
[x,ix]=max(result_in_array');
p=ix';                                                                               



% =========================================================================


end
