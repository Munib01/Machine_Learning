function [W_2, b_2, W_1, b_1] = Backward_Propagation(X_train, Y_train, A_2, A_1, examples, W_1, b_1, W_2, b_2, learning_rate, lambda)
  
%d_W_2 = ((A_2-Y_train).*(sigmoidGradient(A_2)))'*(A_1)/examples + (W_2.*(lambda/examples));
%d_b_2 = sum((A_2-Y_train).*(sigmoidGradient(A_2)))'/examples;
d_W_2 = ((A_2-Y_train))'*(A_1)/examples;
d_b_2 = sum((A_2-Y_train))'/examples;

d_d = ((A_2-Y_train).*(sigmoidGradient(A_2)))*W_2;
d_W_1 = ((d_d).*(sigmoidGradient(A_1)))'*(X_train)/examples + (W_1.*(lambda/examples));
d_b_1 = sum((d_d).*(sigmoidGradient(A_1)))'/examples;

W_2 = W_2 - (learning_rate*d_W_2);
b_2 = b_2 - (learning_rate*d_b_2);
W_1 = W_1 - (learning_rate*d_W_1);
b_1 = b_1 - (learning_rate*d_b_1);
end