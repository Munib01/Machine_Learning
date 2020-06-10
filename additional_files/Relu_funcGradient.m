function g = Relu_funcGradient(z)
g = zeros(size(z,1),size(z,2))
g = z>0;
end
