function vep = ProjVec(ve, x, th, phi)
%ProjVec() project a vector at x to its SO2 reduced space.
% input:
%   x: point inside the template
%   ve: vector need to be projected. It could be eigenvector at x or 
%       velocity at x (not \hat{x}).
%   th: angle at x.

%projection matrix is h= (I - |tx><tp|/<tx|tp>) * g(-th), so eigenvector |ve> is
%projected to |ve> - |tx>*(<tp|ve>|/<tx|tp>), before which, g(-th) is 
%performed.

n = size(x,1);
tp_rho = zeros(n,1); tp_rho(2) = 1;
tp_tau = zeros(n,1); tp_tau(4) = 1;

vep = GroupTrans(ve, -th, -phi);

tx_tau = GroupTantau(x); 
tx_rho = GroupTanrho(x);

vep = vep - kron((tp_tau'*vep)/(tp_tau'*tx_tau), tx_tau) ...
    - kron((tp_rho'*vep)/(tp_rho'*tx_rho), tx_rho - tx_tau);



end