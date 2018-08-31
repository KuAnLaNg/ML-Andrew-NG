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
%说明：您应该通过以下部分完成代码。
%第1部分：前馈神经网络并在变量J中返回成本。实现第1部分后，您可以通过验证ex4.m中计算的成本来验证您的成本函数计算是否正确
%第2部分：实现反向传播算法来计算梯度Theta1_grad和Theta2_grad。您应该分别返回与Theta1_grad和Theta2_grad中的Theta1和Theta2相关的成本函数的偏导数。实现第2部分后，您可以通过运行checkNNGradients来检查您的实现是否正确
%注意：传递给函数的向量y是包含1..K值的标签向量。您需要将此向量映射到1和0的二进制向量，以与神经网络成本函数一起使用。
%提示：如果您是第一次实施训练示例，我们建议使用for循环实现反向传播。
%第3部分：使用成本函数和梯度实现正则化。
%提示：您可以围绕反向传播的代码实现此功能。也就是说，您可以单独计算***正则化的***梯度，然后将它们添加到第2部分的Theta1_grad和Theta2_grad中。

X = [ones(m, 1) X];                                                              %现在X是5000*401的矩阵 ,每个样本是占一行；够坏，以前帮着添加这项的，这次没加


z_1=Theta1*X';                                                                   %现在z_1是（25*401）*（401*5000）=25*5000的矩阵,所以现在hidden_result也是25*5000的矩阵 
hidden_result=sigmoid(z_1);                                                      



%这两步是为了加到26*5000，且加的第一列全为1
hidden_result= [ones(m, 1) hidden_result'];                                      %第一步：竖直放置的5000根韭菜全横着躺平，然后每根韭菜都加个值为1的头
hidden_result=hidden_result';                                                    %第二步：将横置且加完头的5000韭菜重新竖直放置，以便后续工作 现在新的韭菜是26*5000





z_2=Theta2*hidden_result;
output_result=sigmoid(z_2);                                                      %现在结果矩阵output_result是10*5000                                                                                       


%现在从文件中读取的y只是5000个标签值，即每个样本只对应一个输出，不是10个输出，这样无法计算神经网络的代价函数，所以现在要转换标签值y的形式
temp_y=zeros(num_labels,m);


 for i=1:m
  temp_y(y(i),i)=1;                                                              %运用单个元素的索引，进行标签值的“列向量化”                                                      
 end

% -------------------------------------------------------------
%一、无正则化的成本函数计算

A=sum(sum(temp_y.*log(output_result)));   
B=sum(sum((1-temp_y).*log(1-output_result)));
J_up=A+B;
J= -J_up/m;
% -------------------------------------------------------------






% -------------------------------------------------------------
%二、正则化的成本函数计算
a=Theta1;
a(:,1)=0;
b=Theta2;
b(:,1)=0;

J_up=A+B-lambda/2*(sum(sum(a.^2))+sum(sum(b.^2)));
J= -J_up/m;
% -------------------------------------------------------------





% -------------------------------------------------------------
%五、实现反向传播：返回参数的偏导数   
%***根据课程可知，求代价函数和求梯度完全是两个问题，求梯度是采用循环而不是代价函数所用的向量化及矩阵乘法法则来实现”批量“的效果， 
%***所以每次***只取一组输入数据***，不是像在“一、无正则化的成本函数”与在“二、正则化的成本函数计算”中计算代价函数时一次就用满了5000个样本


temp_Theta1_grad = zeros(size(Theta1));                                                %用在215行
temp_Theta2_grad = zeros(size(Theta2));                                                %用在216行

Theta1_grad_temp = zeros(size(Theta1));                                                %用在215、224、225行
Theta2_grad_temp = zeros(size(Theta2));                                                %用在216、228、229行

for i=1:m
  %利用“向前传播”计算一个样本的中间层输出和输出层输出
  
  z_1=Theta1*X(i,:)';                                                                   %由于第72行，这里的X已经是5000*401的矩阵了，但我们每次只用一组数据，所以每次***只提取一行***
  hidden_result=sigmoid(z_1);                                                           %hidden_result是中间层输出（列向量形式），但未加偏置单元的输出“1”       25*1的形式
  hidden_result=[ones(1,1);hidden_result];                                              %现在hidden_result是了加偏置单元的输出“1”的中间层输出      26*1的形式
  z_2=Theta2*hidden_result;
  output_result=sigmoid(z_2);                                                           %10*1的形式
  
  
  %下面是利用“向后传播”（反向传播）计算
  %计算各层输出误差
  
  e_3=output_result-temp_y(:,i);                                                         %输出层误差，最正儿八经的输出误差        10*1
  e_2=(Theta2'*e_3).*(hidden_result.*(1-hidden_result));                                 %或 t=[1;sigmoid(z_1)];   e_2=(Theta2'*e_3).*sigmoidGradient(t);  这里的hidden_result已经是加了截距——“1”的列向量了
                                                                                          
  % Theta2'*e_3是26*10*10*1=26*1（这26个值里面，第一个值为中间层的偏置单元由于对输出层有贡献，而对10个输出值造成的总麻烦）     26*1  
  %*****但是e_2的第一个值是没有用的，因为他是中间层偏置单元对于输出层各个单元犯下的累计错，也可视为”输出层各个单元对于中间层偏置单元的期待”或“输出层各个单元对于中间层偏置单元评价出的它所需要改变的量”
  %(Theta2'*e_3)实质上是简单版本的“期待”=“权重*简单版需要改变的量”，不是3blue1brown里的复杂版本的“期待”（虽然我觉得3blue1brown里才是对的）
  
  
  
  %****
  %e_2实际上是 “中间层每个单元的"错"（也可以理解为是输出层对于中间层每个单元需要改变量的评价或单个样本的代价函数对于中间层每个单元的敏感程度）*中间层每个单元的非线性化步骤的导数”，但是e_2的这种组合方式也决定了它只针对Theta1里的权重梯度计算有用
  %同理，e_3也只针对Theta2里的权重梯度计算有用，即：代价函数对J+1层单元的敏感程度只能帮助求解J层权重（J层的权重指的是和J层的输出值配合形成J+1层单元的输入值的权重）的梯度
  %*****
  %****"期待”/“评价”/“敏感程度”才是真正反向传播的唯一东西***
  
  %因为e_2只针对Theta1里的权重梯度计算，但是输入层并没有任何单元（哪怕偏置单元）连接中间层的偏置单元，没有连接就没有权重，所以就算e_2的第一个值算出了代价函数对其的敏感程度，这个值也不能为计算Theta1里的权重梯度做出贡献
  %但是因为Theta2'*e_3和(hidden_result.*(1-hidden_result))都是26*1，所以为了好计算，先算出来再说，大不了不用就行
  
  
   %根据Andrew ng那种独有的（与3blue1brown和《Python神经网络》均不一样的）的对于L层的求梯度方式写出的Theta_2的全部权重的梯度
 % -------------------------------------------------------------—------------------------------------------------------------------------------------- 
  temp_Theta2_grad=e_3*hidden_result';                                                 %这里的hidden_result不加截距——“1”就会更新不了中间层偏置单元对于输出层10个输出单元的10个权重了，很明显，这样做是错的，因为不更新偏置肯定是错的
  
  
  %或for i=1:(hidden_layer_size + 1)                                                   %"hidden_layer_size + 1"是Theta2的列数，必须为hidden_layer_size + 1，否则就会更新不了中间层偏置单元对于输出层10个输出单元的10个权重了
    %temp_Theta2_grad(:,i)= hidden_result(i)*e_3;                                      %考虑到Theta2_grad和Theta2的造型是一样的，所以要考虑Theta2中权重的摆放结构；再结合Andrew对于最后一层权重的求解公式，综合考虑到这些后写出的Theta2_grad的表达式。
   %end                                                                                %其中“hidden_result(i)”实际上是一个数值，而且hidden_result是指加了hidden_result层偏置单元输出——“1”的hidden_result，所以实际上是
	
  
  %这里因为是有10个输出单元，所以e_3也是10维的，而hidden_result是26维的（因为加了hidden_result层的偏置单元的输出——1）***这样中间层偏置单元对于输出层10个“真实”单元的权重才能得到更新*** 
	
	
	
	
  %根据Andrew ng（与《Python神经网络》一样，与3blue1brown不一样）的对于l<L层的求梯度方式写出的Theta_1的全部权重的梯度
% -------------------------------------------------------------—------------------------------------------------------------------------------------- 	
 
                                                 
 



 temp_e_2=e_2(2:end);                                                                 %153行已经解释过，第一个值（代价函数对于中间层的偏置单元的敏感度）没有用，所以我们把e_2的第一个值去掉，也必须去掉，否则就不是算Theta1的25*401个权重了，
 %或temp_e_2=zeros(hidden_layer_size,1);                                              %就会变成算26*401个权重了，而输入层的401个单元（加上了输入层偏置单元）是没有跟中间层的偏置单元相连的，所以26*401的第一行并不存在   
 %for i=1:hidden_layer_size
  %temp_e_2(i)=e_2(i+1);
  %end
 	
	
  
  
 %****Theta1的权重里有“输入层偏置单元与中间层各个真实单元相连”的权重，但没有中间层偏置单元与输入层的偏置单元或真实单元相连的权重，因为没有所以没得更新这些权重，这也是把e_2的第一个值扔掉的原因***
 
 temp_Theta1_grad=temp_e_2*X(i,:);                                               %注意X(i,:)的size是401*1，而hidden_result的size是25*1，故此处的不需要对X(i,:)进行转置
 %或temp_input=X(i,:)';                                                          %为了求解Theta1_grad时能使用元素索引；现在temp_input是了加偏置单元的输出“1”的输入，72行干的，***因为这样做的话输入层偏置单元对于中间层25个“真实”单元的权重才能得到更新***      401*1的形式
 %for i=1:(input_layer_size + 1)                                                 %"(input_layer_size + 1"是Theta1的列数
	%temp_Theta1_grad(:,i)=temp_input(i)*temp_e_2;                               %其中“temp_input(i)”实际上是一个数值
	%end

	
%累加每个样本的对梯度值得所有改变的部位	 	 
 Theta1_grad_temp+=temp_Theta1_grad;
 Theta2_grad_temp+=temp_Theta2_grad;
end  	 

% -------------------------------------------------------------

Theta1_grad=Theta1_grad_temp/m;
Theta2_grad=Theta2_grad_temp/m;
% -------------------------------------------------------------
Theta1_grad(:,1)=Theta1_grad_temp(:,1)/m;
Theta1_grad(:,[2:end])=Theta1_grad_temp(:,[2:end])/m + lambda/m*Theta1(:,[2:end]);


Theta2_grad(:,1)=Theta2_grad_temp(:,1)/m;
Theta2_grad(:,[2:end])=Theta2_grad_temp(:,[2:end])/m + lambda/m*Theta2(:,[2:end]);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];                                   %返回的梯度是以size(Theta1)的尺寸返回的Theta1_grad向量和以size(Theta2)的尺寸返回的Theta2_grad向量


end
