#!/bin/bash

#SBATCH -A p_argumentation
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --time 0-06:00:00
#SBATCH --mem 16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=s7550245@msx.tu-dresden.de
#SBATCH -o /scratch/s7550245/Dr.-L.-Bird/log.%j
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu1,gpu2


# module load eb
# module load tensorflow

export TESSDATA_PREFIX=/sw/taurus/libraries/tesseract/3.04/share/tesseract

PYTHONPATH=/sw/taurus/libraries/tesseract/3.04/lib64/:/home/s7550245/.local/lib/python2.7/site-packages /sw/taurus/eb/tensorflow/0.8.0/lib/x86_64-linux-gnu/ld-2.17.so --library-path /sw/taurus/eb/tensorflow/0.8.0/lib/x86_64-linux-gnu:/sw/taurus/eb/cuDNN/5.1/lib64:/sw/taurus/libraries/cuda/8.0.44/lib64:/sw/taurus/eb/Python/2.7.11-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/GMP/6.1.0-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/Tk/8.6.4-intel-2016.03-GCC-5.3-no-X11/lib:/sw/taurus/eb/SQLite/3.9.2-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/Tcl/8.6.4-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/libreadline/6.3-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/ncurses/6.0-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/zlib/1.2.8-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/bzip2/1.0.6-intel-2016.03-GCC-5.3/lib:/sw/taurus/eb/imkl/11.3.3.210-iimpi-2016.03-GCC-5.3.0-2.26/mkl/lib/intel64:/sw/taurus/eb/imkl/11.3.3.210-iimpi-2016.03-GCC-5.3.0-2.26/lib/intel64:/sw/taurus/eb/impi/5.1.3.181-iccifort-2016.3.210-GCC-5.3.0-2.26/lib64:/sw/taurus/eb/ifort/2016.3.210-GCC-5.3.0-2.26/compilers_and_libraries_2016.3.210/linux/mpi/intel64:/sw/taurus/eb/ifort/2016.3.210-GCC-5.3.0-2.26/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/sw/taurus/eb/ifort/2016.3.210-GCC-5.3.0-2.26/lib/intel64:/sw/taurus/eb/ifort/2016.3.210-GCC-5.3.0-2.26/lib:/sw/taurus/eb/icc/2016.3.210-GCC-5.3.0-2.26/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/sw/taurus/eb/icc/2016.3.210-GCC-5.3.0-2.26/lib/intel64:/sw/taurus/eb/icc/2016.3.210-GCC-5.3.0-2.26/lib:/sw/taurus/eb/binutils/2.26-GCCcore-5.3.0/lib:/sw/taurus/eb/GCCcore/5.3.0/lib/gcc/x86_64-unknown-linux-gnu/5.3.0:/sw/taurus/eb/GCCcore/5.3.0/lib64:/sw/taurus/eb/GCCcore/5.3.0/lib:/sw/taurus/libraries/cuda/8.0.44/extras/CUPTI/lib64:/sw/taurus/libraries/cuda/8.0.44/lib64:/sw/taurus/libraries/cuda/8.0.44/lib64:/sw/taurus/libraries/cuda/8.0.44/extras/CUPTI/lib64 /sw/taurus/eb/Python/2.7.11-intel-2016.03-GCC-5.3/bin/python testProg.py /scratch/s7550245/Dr.-L.-Bird/runsDDPG/1485099264
