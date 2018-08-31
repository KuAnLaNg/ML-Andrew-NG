function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network            %这个predict.m函数实际上是在神经网络训练完毕后，询问新的结果用的
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)                         % Theta1, Theta2来自ex3weights.mat， X来自ex3data1.mat，要用Theta1, Theta2当参数去预测X 

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly                           %这里对于p的设置能看出，也要选神经网络10个输出结果的最大值
p = zeros(size(X, 1), 1);                                                        %m由于每个神经网络的输出部分也是10个输出，和predictOneVsAll.m里的对每个样本采用10个正则化的线性回归分类器计算出结果再选出个最大结果一样

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. 
X = [ones(m, 1) X];                                                              %现在X是5000*401的矩阵 ,每个样本是占一行；够坏，本周其他练习是帮着添加这项的，这次由于要中间层也要加偏置，所以没加，练习从一开始就给输入加偏置的意识


z_1=Theta1*X';                                                                   %现在z_1是（25*401）*（401*5000）=25*5000的矩阵,所以现在hidden_result也是25*5000的矩阵 
hidden_result=sigmoid(z_1);                                                      



%这两步是为了加到26*5000，且加的第一列全为1
hidden_result= [ones(m, 1) hidden_result'];                                      %第一步：竖直放置的5000根韭菜全横着躺平，然后每根韭菜都加个值为1的头
hidden_result=hidden_result';                                                    %第二步：将横置且加完头的5000韭菜重新竖直放置，以便后续工作 现在新的韭菜是26*5000





z_2=Theta2*hidden_result;
output_result=sigmoid(z_2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
[x,ix]=max(output_result);
p=ix';



% =========================================================================


end
