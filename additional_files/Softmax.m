function g = Softmax(z)
z_exp = exp(z);
z_sum = sum(z_exp);
g = z_exp/z_sum; 
end
