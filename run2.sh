#!/bin/bash

#SBATCH --job-name=drlbird-server
#SBATCH -A p_argumentation
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --time 2-00:00:00
#SBATCH --mem 4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=s7550245@msx.tu-dresden.de
#SBATCH -o /scratch/s7550245/Dr.-L.-Bird/log.%j
#SBATCH -x taurusi1108,taurusi1071,taurusi1008

rm /scratch/s7550245/chromium/SingletonLock
cp -r  ~/Application\ Cache /scratch/s7550245/chromium/Default
# cp -r  ~/Application\ CacheBackup/* /scratch/s7550245/chromium/Default/Application\ Cache


cd ~/abV1.32/plugin

export DISPLAY=:1
Xvfb :1 -screen 0 1024x768x16 &

../../AppDir51/bin/ld-linux-x86-64.so.2 --library-path ../../AppDir51/bin ../../AppDir51/bin/chrome --disable-setuid-sandbox --single-process --disable-gpu --load-extension="." --disable-infobars --disable-session-crashed-bubble  --user-data-dir=/scratch/s7550245/chromium  http://chrome.angrybirds.com  &

/sw/global/tools/java/jdk1.8.0_66/bin/java -jar ~/ABServer.jar -vv
