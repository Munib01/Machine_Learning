function g = Relu_func(z)
g = zeros(size(z,1),size(z,2))
g = z>0;
g = abs(g.*z);
end
