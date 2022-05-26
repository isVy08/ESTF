function edq = f_edq(x, q_tile)
[~,n] = size(x);
edq = zeros(size(q_tile,1),2); 
parfor ii = 1:size(q_tile,2)  
 obj_val = zeros(n,1);
for i = 1:n
x_diff = x(:,1:end ~=i) - x(:,i);
obj_val(i) = sum(sum(q_tile(ii).*abs(x_diff.*(x_diff>0))+(1-q_tile(ii)).*abs(x_diff.*(x_diff<0))));
end
 edq(ii) = find(obj_val == min(obj_val),1);
end
end