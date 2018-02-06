function [ spline ] = CardinalSpline( data,cpt,num_cpt,s )

spline = zeros(length(data),length(cpt));

for i=1:length(data)
    nearest_c_pt_index = max(find(cpt<data(i)));
    nearest_c_pt_time = cpt(nearest_c_pt_index);
    next_c_pt_time = cpt(nearest_c_pt_index+1);
    u = (data(i)-nearest_c_pt_time)./(next_c_pt_time-nearest_c_pt_time);
    p=[u^3 u^2 u 1]*[-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];
    spline(i,:) = [zeros(1,nearest_c_pt_index-2) p zeros(1,num_cpt-4-(nearest_c_pt_index-2))];
end



end

