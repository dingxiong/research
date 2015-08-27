# ./copyh5v2.sh rpo 1 100 input output
for ix in $(seq $2 $3); do
    h5copy -i $4 -o $5 -s /$1/$ix -d /$1/$(($ix+240))
done

