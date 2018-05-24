function [ rbf ] = gaussian_rbf_2d( x,mux,muy,variance )

[Nx,~] = size(mux);
[Ny,~] = size(muy);
rb = [];
rbf  =[];

for i=1:Nx 
    
    for j=1:Ny
         rb(:,j) = exp(-sum((x - [mux(i),muy(j)]).^2,2) / (2 * variance));
    end
    
    rbf = [rbf , rb];
    
end


end

