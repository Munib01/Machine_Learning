function [W b Z A] = Initialization(examples, units_hidden_layer_previous, units_hidden_layer)

W = rand(units_hidden_layer,units_hidden_layer_previous);
b = rand(units_hidden_layer,1);
Z = zeros(examples,units_hidden_layer);
A = zeros(examples,units_hidden_layer);
size(W);
size(b);
size(Z);
size(A);
end


%W_1 = rand(25,400);
%b_1 = rand(25,1);
%Z_1 = zeros(4000,25);
%A_1 = zeros(4000,25);
%W_2 = rand(10,25);
%b_2 = rand(10,1);
%Z_2 = zeros(4000,10);
%A_2 = zeros(4000,10);
