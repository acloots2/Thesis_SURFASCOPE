#---
#{
#"hostname": "lemaitre3",
#"author": "J-M Beuken",
#"date": "2020-06-22",
#"description": [
#   "Configuration file for lemaitre3 based on easy-build and the intel toolchain 2019b"
#],
#"qtype": "slurm",
#"keywords": ["linux", "intel", "easybuild"],
#"pre_configure": [
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi
module purge
module load releases/2020a
module load intel/2020a
module load Python/3.8.2-GCCcore-9.3.0
module load ABINIT/9.2.2-intel-2020a
module load gompi/2020a
module load FFTW/3.3.8-gompi-2020a
module load matplotlib/3.2.1-foss-2020a-Python-3.8.2
export PYTHONPATH=/
# ]
#}
#---

#install architecture-independent files in PREFIX
#prefix="~/local/"
#
FC="mpiifort"
CC="mpiicc"
CXX="mpiicpc"

# MPI/OpenMP
#with_mpi="${EBROOTIIMPI}"
with_mpi="yes"
enable_openmp="no"

CFLAGS="-O2 -g"
CXXFLAGS="-O2 -g"
FCFLAGS="-O2 -g"

# BLAS/LAPACK with MKL
with_linalg_flavor="mkl"
#LINALG_CPPFLAGS="-I${MKLROOT}/include"
#LINALG_FCFLAGS="-I${MKLROOT}/include"
LINAGL_LIBS="-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl"
#LINALG_LIBS="-L${MKLROOT}/lib/intel64 -Wl,--start-group  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group"

# FFT from MKL
with_fft_flavor="dfti"
#FFT_LIBS="-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl"

# libxc
with_libxc="${EBROOTLIBXC}"

# hdf5/netcdf4. Use nc-config and nf-config to get installation directory
with_netcdf="`nc-config --prefix`"
with_netcdf_fortran="`nf-config --prefix`"
with_hdf5="${EBROOTHDF5}"

#Add UTF-8
export LC_CTYPE=en_US.UTF-8
