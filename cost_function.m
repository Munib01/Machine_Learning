function g = cost_function(A_2, Y_train, examples, lambda, W_1, W_2)

g = 0.5*sum(sum((A_2 - Y_train).^2))/examples + (lambda*0.5/examples)*(sum(W_2(:).^2)+sum(W_1(:).^2));
end