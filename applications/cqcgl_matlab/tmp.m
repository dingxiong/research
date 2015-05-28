h = 0.1; nstp = 50;
N = 512; A0 =zeros(N); sigma = 0.002;
for i = 1:N, 
    for j = 1:N,
        A0(i,j)=exp(-sigma*((i-N/2)^2+(j-N/2)^2)); 
    end,
end