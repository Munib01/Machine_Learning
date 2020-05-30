function g = Y_extraction(y)
iter = size(y)
g = zeros(iter,10);

for i =1:1:iter 
  g(i,y(i)) = 1; 
end
end