function [mu, doChangh, doAccept] = adapth(s, mumax, mumin, mue, muc)

mu = 1;
doChangh = true;
doAccept = true;

if s > mumax
    mu = mumax;
elseif s > mue
    mu = s;
elseif s >= 1
    mu = 1;
    doChangh = false;
else
    doAccept = false;
    if s > muc
        mu = muc;
    elseif s > mumin
        mu = s;
    else
        mu = mumin;
    end
end

end