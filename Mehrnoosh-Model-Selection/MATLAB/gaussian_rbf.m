function [ rbf ] = gaussian_rbf( x,mean,variance )

rbf = exp(-((x - mean).^2) / (2 * variance));

end

