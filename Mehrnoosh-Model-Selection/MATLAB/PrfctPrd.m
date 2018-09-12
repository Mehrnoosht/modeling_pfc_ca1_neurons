function [design_matrix,pp,idx_pp] = PrfctPrd(design_matrix,spike)

pp = [];
idx_rw = [];;
for j=1:size(design_matrix,2)
   idx_rw = find(design_matrix(:,j)~=0);
   pp(j) = sum(spike(idx_rw));  
end
idx_pp = find(pp==0);
design_matrix(:,idx_pp) = [];

end

