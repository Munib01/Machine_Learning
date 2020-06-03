function [A_1, A_2] = Forward_Propagation(X_train, W_1, b_1, W_2, b_2)
  
Z_1 = Product(X_train, W_1, b_1);
A_1 = sigmoid(Z_1);

Z_2 = Product(A_1, W_2, b_2);
A_2 = sigmoid(Z_2);
end