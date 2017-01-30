#!/bin/bash

srun -A p_argumentation -N 1 --ntasks=1 -c 1 --mem=4G --time=1:00:00 -w taurusi1215 --pty --x11=first zsh
cd ~/AppDirx11vnc
bin/ld-linux-x86-64.so.2 --library-path bin bin/x11vnc -display :1 -localhost &
vncviewer :0
