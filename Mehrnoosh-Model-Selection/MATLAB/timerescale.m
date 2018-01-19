function [ Z ] = timerescale( spike,lambda )


N=length(spike);
spikeindex = find(spike);
Z(1) = sum(lambda(1:spikeindex(1)));
for i=2:N
    
   Z(i) = sum(lambda(spikeindex(i-1):spikeindex(i)));
    
end


end

