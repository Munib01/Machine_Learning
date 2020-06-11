function g = Softmax(z)
g = zeros(size(z,1),size(z,2));
z_exp = exp(z);
z_sum = sum(z_exp);
g = z_exp/z_sum; 
end