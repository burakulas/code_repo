#!/bin/sh
#export OMP_NUM_THREADS=5

## Author: Burak Ulas -  github.com/burakulas
## 2024, Konkoly Observatory, COMU

cp lcin.active lctmp1

mod=2  # DETACHED BINARY

ec=0

while read -r per q t1 t2 pot1 pot2 incl; do
echo $per,$q,$t1,$t2,$pot1,$pot2,$incl
##done < det_parameters.lst


for i in $(seq $incl 1.8 85);do
echo $per,$q,$t1,$t2,$pot1,$pot2,$i


awk -v mod=$mod 'FNR == 4 {$1=mod};1' lctmp1 > lctmp2_$mod
moda=$(awk 'FNR == 4 {printf(" %s %s %s %s  %s  %s     %s  %s %s  %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}' lctmp2_$mod)
sed -i "4s/.*/$moda/" lctmp1


perp=$(echo "scale=4; $per / 10" | bc)
printf -v pern '%.10fd+01' "$perp"
awk -v pern=$pern 'FNR == 2 {$3=pern};1' lctmp1 > lctmp2_$pern
pera=$(awk 'FNR == 2 {printf("%s   %s %s  %s    %s %s  %s %s %s %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}' lctmp2_$pern)
sed -i "2s/.*/$pera/" lctmp1

printf -v in '%.3f' "$i"
awk -v inc=$in 'FNR == 5 {$6=inc};1' lctmp1 > lctmp2_$in
ia=$(awk 'FNR == 5 {printf("%s %s    %s    %s   %s   %s  %s  %s   %s    %s    %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)}' lctmp2_$in)
sed -i "5s/.*/$ia/" lctmp1

printf -v ecnp '%.5f' "$ec"
ecn=$(echo $ecnp | sed 's/^0*//')
awk -v ecn=$ecn 'FNR == 5 {$1=ecn};1' lctmp1 > lctmp2_$ecn
eca=$(awk 'FNR == 5 {printf("%s %s    %s    %s   %s   %s  %s  %s   %s    %s    %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)}' lctmp2_$ecn)
sed -i "5s/.*/$eca/" lctmp1

printf -v qn '%.6fd+00' "$q"
awk -v qn=$qn 'FNR == 6 {$7=qn};1' lctmp1 > lctmp2_$qn
qa=$(awk 'FNR == 6 {printf(" %s  %s  %s  %s %s %s %s  %s  %s  %s  %s %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' lctmp2_$qn)
sed -i "6s/.*/$qa/" lctmp1

t1p=$(echo "scale=4; $t1 / 10000" | bc)
printf -v t1n '%.4f' "$t1p"
awk -v t1n=$t1n 'FNR == 6 {$1=t1n};1' lctmp1 > lctmp2_$t1n
t1a=$(awk 'FNR == 6 {printf(" %s  %s  %s  %s %s %s %s  %s  %s  %s  %s %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' lctmp2_$t1n)
sed -i "6s/.*/$t1a/" lctmp1

t2p=$(echo "scale=4; $t2 / 10000" | bc)
printf -v t2n '%.4f' "$t2p"
awk -v t2n=$t2n 'FNR == 6 {$2=t2n};1' lctmp1 > lctmp2_$t2n
t2a=$(awk 'FNR == 6 {printf(" %s  %s  %s  %s %s %s %s  %s  %s  %s  %s %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' lctmp2_$t2n)
sed -i "6s/.*/$t2a/" lctmp1

pot1p=$(echo "scale=4; $pot1 / 10" | bc)
printf -v pot1n '%.6fd+01' "$pot1p"
awk -v pot1n=$pot1n 'FNR == 6 {$5=pot1n};1' lctmp1 > lctmp2_$pot1n
pot1a=$(awk 'FNR == 6 {printf(" %s  %s  %s  %s %s %s %s  %s  %s  %s  %s %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' lctmp2_$pot1n)
sed -i "6s/.*/$pot1a/" lctmp1

pot2p=$(echo "scale=4; $pot2 / 10" | bc)
printf -v pot2n '%.6fd+01' "$pot2p"
#echo Mod: $mod, Per: $pern, inc: $in, ecc: $ecn, q: $qn, T1: $t1n, T2: $t2n, pot1: $pot1n, pot2: $pot2n
awk -v pot2n=$pot2n 'FNR == 6 {$6=pot2n};1' lctmp1 > lctmp2_$pot2n
pot2a=$(awk 'FNR == 6 {printf(" %s  %s  %s  %s %s %s %s  %s  %s  %s  %s %s\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' lctmp2_$pot2n)
sed -i "6s/.*/$pot2a/" lctmp1

rm lctmp2*

cp lctmp1 ../wd/lcin.active

cd ../wd
./lc &&lnl1=$(awk '/JD               Phase/{ print NR; exit }' lcout.active)
lnl2=$(( lnl1 + 1 ))

lnu1=$(awk '/star     r pole/{ print NR; exit }' lcout.active)
lnu2=$(( lnu1 - 4 ))

lum=$(awk '/L1           L2/{ print NR; exit }' lcout.active)
luma=$(( lum + 1 ))
l1n=$(awk -v luma=$luma 'NR==luma {print $2}' lcout.active)
l2n=$(awk -v luma=$luma 'NR==luma {print $3}' lcout.active)

awk -v lnl2=$lnl2 -v lnu2=$lnu2 'NR==lnl2, NR==lnu2 {printf "%.6f %.6f\n", $2, $6}' lcout.active > lc_p_${pern}_m_${mod}_i_${i}_ec_${ec}_${qn}_${t1n}_${t2n}_${pot1n}_${pot2n}.dat
mv lc_p_${pern}_m_${mod}_i_${i}_ec_${ec}_${qn}_${t1n}_${t2n}_${pot1n}_${pot2n}.dat ../lcplay/det_lc/
echo $pern"_"$mod"_i_"$i"_ec_"$ec"_"$qn"_"$t1n"_"$t2n"_"$pot1n"_"$pot2n
cd ../lcplay/

#echo "...OK..."

done

done < det_parameters.lst # To acces the list: https://drive.google.com/file/d/1iJaQgg18MHhnG1papI-41QEZURFzw7b1/view?usp=sharing
