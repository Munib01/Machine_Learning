function g = Hardmax(z)
g = zeros(size(z,1),size(z,2));
[val,ind] = max(z);
g(ind) = 1
end