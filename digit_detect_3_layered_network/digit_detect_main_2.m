%%%%% Data Extration %%%%%

[X_train Y_train_1 X_test Y_test_1] = Data_extraction();

Y_train = zeros(size(Y_train_1),10);

Y_test = zeros(size(Y_test_1),10);

Y_train = Y_extraction(Y_train_1); 

Y_test = Y_extraction(Y_test_1);



%%%%% Initializations %%%%%

examples = 4000;        units_layer_1 = 400;          units_layer_2 = 20;       units_layer_3 = 10;

[W_1 b_1 Z_1 A_1] = Initialization(examples, units_layer_1, units_layer_2);

[W_2 b_2 Z_2 A_2] = Initialization(examples, units_layer_2, units_layer_2);

[W_3 b_3 Z_3 A_3] = Initialization(examples, units_layer_2, units_layer_3);

iteration = 3000;        cost = 0;       cost_trace = zeros(iteration,1);       learning_rate = 1;      lambda = 0;



for i=1:1:iteration

%%%%% Forward Propagation %%%%%

Z_1 = Product(X_train, W_1, b_1);
A_1 = sigmoid(Z_1);

Z_2 = Product(A_1, W_2, b_2);
A_2 = sigmoid(Z_2);

Z_3 = Product(A_2, W_3, b_3);
A_3 = sigmoid(Z_3);

%%%%% Error Correction %%%%%

cost_trace(i,1) = 0.5*sum(sum((A_3 - Y_train).^2))/examples + (lambda*0.5/examples)*(sum(W_3(:).^2)+sum(W_2(:).^2)+sum(W_1(:).^2));

cost = cost + cost_trace(i,1);



%%%%% Backward Propagation %%%%%


%
d_d_3 = (A_3-Y_train).*(sigmoidGradient(A_3));

d_W_3 = d_d_3'*(A_2)/examples;

d_b_3 = sum(d_d_3)'/examples;

%d_W_2 = ((A_2-Y_train))'*(A_1)/examples;

%d_b_2 = sum((A_2-Y_train))'/examples;

d_d_2 = (d_d_3*W_3).*(sigmoidGradient(A_2));

d_W_2 = d_d_2'*(A_1)/examples;

d_b_2 = sum(d_d_2)'/examples;

%

d_d_1 = (d_d_2*W_2).*(sigmoidGradient(A_1));

d_W_1 = d_d_1'*(X_train)/examples;

d_b_1 = sum(d_d_1)'/examples;



W_3 = W_3 - (learning_rate*d_W_3);

b_3 = b_3 - (learning_rate*d_b_3);

W_2 = W_2 - (learning_rate*d_W_2);

b_2 = b_2 - (learning_rate*d_b_2);

W_1 = W_1 - (learning_rate*d_W_1);

b_1 = b_1 - (learning_rate*d_b_1);

%}

end

cost_trace

plot([1:1:iteration],cost_trace)

xnxxx = 0;

for i = 1:1:iteration-1
  
if(cost_trace(i)<cost_trace(i+1))

xnxxx = xnxxx + 1;

end
end
xnxxx
performance_with_training_data = accuracy_2(W_3, W_2, W_1, b_3, b_2, b_1, X_train, Y_train_1)
performance_with_test_data = accuracy_2(W_3, W_2, W_1, b_3, b_2, b_1, X_test, Y_test_1)