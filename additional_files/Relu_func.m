function g = Relu_func(z)
g = z>0;
g = abs(g.*z);
end
