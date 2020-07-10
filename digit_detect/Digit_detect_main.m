%%%%% Data Extration %%%%%
[X_train Y_train_1 X_test Y_test_1] = Data_extraction();
Y_train = zeros(size(Y_train_1),10);
Y_test = zeros(size(Y_test_1),10);
Y_train = Y_extraction(Y_train_1); 
Y_test = Y_extraction(Y_test_1);

%%%%% Initializations %%%%%
examples = 4000;        units_layer_1 = 400;          units_layer_2 = 20;       units_layer_3 = 10;
[W_1 b_1 Z_1 A_1] = Initialization(examples, units_layer_1, units_layer_2);
[W_2 b_2 Z_2 A_2] = Initialization(examples, units_layer_2, units_layer_3);
iteration = 1000;        cost = 0;       cost_trace = zeros(iteration,1);       learning_rate = 1;      lambda = 0.05;

for i=1:1:iteration
%%%%% Forward Propagation %%%%%
[A_1, A_2] = Forward_Propagation(X_train, W_1, b_1, W_2, b_2);

%%%%% Error Correction %%%%%
cost_trace(i,1) = cost_function(A_2, Y_train, examples, lambda, W_1, W_2);
cost = cost + cost_trace(i,1);

%%%%% Backward Propagation %%%%%
[W_2, b_2, W_1, b_1] = Backward_Propagation(X_train, Y_train, A_2, A_1, examples, W_1, b_1, W_2, b_2, learning_rate, lambda);

%{
d_W_2 = ((A_2-Y_train).*(sigmoidGradient(A_2)))'*(A_1)/examples;
d_b_2 = sum((A_2-Y_train).*(sigmoidGradient(A_2)))'/examples;
%d_W_2 = ((A_2-Y_train))'*(A_1)/examples;
%d_b_2 = sum((A_2-Y_train))'/examples;

d_d = ((A_2-Y_train).*(sigmoidGradient(A_2)))*W_2;
d_W_1 = ((d_d).*(sigmoidGradient(A_1)))'*(X_train)/examples;
d_b_1 = sum((d_d).*(sigmoidGradient(A_1)))'/examples;

W_2 = W_2 - (learning_rate*d_W_2);
b_2 = b_2 - (learning_rate*d_b_2);
W_1 = W_1 - (learning_rate*d_W_1);
b_1 = b_1 - (learning_rate*d_b_1);
%}

end
cost_trace
plot([1:1:iteration],cost_trace)

performance_with_training_data = accuracy(W_2, W_1, b_2, b_1, X_train, Y_train_1)
performance_with_test_data = accuracy(W_2, W_1, b_2, b_1, X_test, Y_test_1)
