#!/bin/bash

make -f Makefile2 clean
perl -pe 's/PRINT = yes|no/PRINT = no/; s/RPO_OMP = yes|no/RPO_OMP = no/' Makefile2 > tmp
make -f tmp && make move -f Makefile2
echo "============"

make -f Makefile2 clean
perl -pe 's/PRINT = yes|no/PRINT = yes/; s/RPO_OMP = yes|no/RPO_OMP = no/' Makefile2 > tmp
make -f tmp && mv libcqcglRPO.so libcqcglRPO_print.so \
    && mv libcqcglRPO.a libcqcglRPO_print.a && make move -f Makefile2
echo "============"

make -f Makefile2 clean
perl -pe 's/PRINT = yes|no/PRINT = yes/; s/RPO_OMP = yes|no/RPO_OMP = yes/' Makefile2 > tmp
make -f tmp && mv libcqcglRPO.so libcqcglRPO_omp.so \
    && mv libcqcglRPO.a libcqcglRPO_omp.a && make move -f Makefile2
						   
rm tmp
