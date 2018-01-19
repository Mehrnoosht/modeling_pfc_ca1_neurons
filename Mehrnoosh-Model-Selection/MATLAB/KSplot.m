function KS = KSplot(lambda, spike)

j = 0;
lambdaInt = 0;
for t=1:length(spike)
    lambdaInt = lambdaInt + lambda(t);
     if (spike(t))
         j = j + 1;
         KS(j) = 1-exp(-lambdaInt);%time rescaling therom (converts to uniform)
         lambdaInt = 0;
     end
end
KSSorted = sort( KS );
N = length( KSSorted);
plot( KSSorted, ([1:N]-.5)/N, 'b', 0:.01:1,0:.01:1, 'k',0:.01:1, ...
      [0:.01:1]+1.36/sqrt(N), 'k:', 0:.01:1,[0:.01:1]-1.36/sqrt(N), 'k:' ); 
axis( [0 1 0 1] );
