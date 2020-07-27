function performance = accuracy(W_2, W_1, b_2, b_1, X_train, Y_train_1)

summy = 0;

Z_1 = Product(X_train, W_1, b_1);
A_1 = tan_func(Z_1);

Z_2 = Product(A_1, W_2, b_2);
A_2 = sigmoid(Z_2);

[a,b] = max(A_2,[],2);

for i =1:1:size(X_train,1)

  if(b(i) == Y_train_1(i))
  summy = summy+1;
end

end

performance = (summy*100)/size(X_train,1);
end
