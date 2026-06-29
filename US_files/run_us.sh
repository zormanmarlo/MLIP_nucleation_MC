#!/bin/bash

for i in $(seq 32 2 32); do
	sed "s/CENTER/${i}/g" ../configs/nacl_us/100mM_nacl_template_dang_test.yaml > ../configs/nacl_us/100mM_nacl_${i}mer_dang_test.yaml
    	sed "s/CENTER/${i}/g" submit_template.sh > submit_tmp.sh
	sbatch submit_tmp.sh
#	rm submit_tmp.sh
done
