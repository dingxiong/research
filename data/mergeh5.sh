# to merge two h5 files by selection rule
fileIn1=ks22h001t120x64_v2.h5
fileIn2=ks22h001t120x64_v3.h5
fileOut=zz2.h5
 
ppoStart=85
ppoEnd=85
ppoInter=(85)

rpoStart=1
rpoEnd=50
rpoInter=(5 20 26 41)

ix=0
for i in $(seq $ppoStart $ppoEnd); do
    if [ "$i" != "${ppoInter[$ix]}" ]; then
	h5copy -i $fileIn1 -o $fileOut -s /ppo/$i -d /ppo/$i
    else
	h5copy -i $fileIn2 -o $fileOut -s /ppo/$i -d /ppo/$i
	echo ${ppoInter[$ix]}
	ix=$(($ix+1))
    fi
done

#ix=0
#for i in $(seq $rpoStart $rpoEnd); do
#    if [ "$i" != "${rpoInter[$ix]}" ]; then
#	h5copy -i $fileIn1 -o $fileOut -s /rpo/$i -d /rpo/$i
#    else
#	h5copy -i $fileIn2 -o $fileOut -s /rpo/$i -d /rpo/$i
#	echo ${rpoInter[$ix]}
#	ix=$(($ix+1))
#    fi
#done