#!bin/sh

cd IN

for i in $(seq 1 1 `ls | grep -c defs`)
do 
	sed -i "s/,/ /g" defs_"$i".txt
	sed -i "s/,/ /g" vecs_"$i".txt
done

