#!/usr/bin/env bash
# Installs rollRegres 0.1.4 (archived on CRAN) with a compatibility patch for R >= 4.5.
# The patch only adds headers/macros for the Fortran character-length interface;
# the algorithm is not modified.
set -euo pipefail
Rscript -e 'install.packages(c("Rcpp", "RcppArmadillo", "checkmate"), repos="https://cloud.r-project.org")'
cd /tmp
rm -rf rollRegres rollRegres_0.1.4.tar.gz
curl -sLO https://cran.r-project.org/src/contrib/Archive/rollRegres/rollRegres_0.1.4.tar.gz
tar xzf rollRegres_0.1.4.tar.gz
sed -i '1i #define USE_FC_LEN_T\n#include <Rconfig.h>\n#include <R_ext/BLAS.h>\n#ifndef FCONE\n# define FCONE\n#endif' rollRegres/src/BLAS_LINPACK.cpp
R CMD INSTALL rollRegres
Rscript -e 'library(rollRegres); cat("rollRegres", as.character(packageVersion("rollRegres")), "OK\n")'
