ff=dang
for i in 10 20 30 40 50 75 100 125 150 200 250 500 1000
do
	#sbatch submit_scripts/${i}mM_adapUS_${ff}_submit.sh
	sbatch submit_scripts/${i}mM_${ff}_submit.sh
done
