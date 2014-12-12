for ix in $(seq 600); do
    h5copy -i ks22h02t100120EV.h5 -o ks22h02t100EV.h5 -s /ppo/$ix -d /ppo/$(($ix+240))
done

for ix in $(seq 595); do
    h5copy -i ks22h02t100120EV.h5 -o ks22h02t100EV.h5 -s /rpo/$ix -d /rpo/$(($ix+239))
done
