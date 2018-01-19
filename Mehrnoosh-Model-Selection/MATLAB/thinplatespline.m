function [ y ] = thinplatespline( x,mu )

y = (x-mu).^2.*log(abs(x-mu));

end

