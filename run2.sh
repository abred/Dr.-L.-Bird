#!/bin/bash
# set -o xtrace

#SBATCH --job-name=drlbird-server
#SBATCH -A p_argumentation
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --time 0-01:00:00
###SBATCH --time 0-01:00:00
#SBATCH --mem 8G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=s7550245@msx.tu-dresden.de
#SBATCH -o /scratch/s7550245/Dr.-L.-Bird/log.%j
###SBATCH -x taurusi1162,taurusi1227

echo $SLURMD_NODENAME > /scratch/s7550245/Dr.-L.-Bird/host-$SLURM_JOB_ID
rm /scratch/s7550245/chromiums/chromium$3/SingletonLock
cp -r  ~/aibird_stuff/Application\ Cache /scratch/s7550245/chromiums/chromium$3/Default
# cp -r  ~/Application\ CacheBackup/* /scratch/s7550245/chromium/Default/Application\ Cache


cd ~/aibird_stuff/abV1.32/plugin

export DISPLAY=:1
Xvfb :1 -screen 0 1024x768x16 &


portS=$1
portC=$2

cp -r /home/s7550245/aibird_stuff/abV1.32/plugin /scratch/s7550245/chromiums/chromium$3
# cp /home/s7550245/abV1.32/plugin/script.jsBackup /scratch/s7550245/chromiums/chromium$3/plugin/script.js
sed -i "s/localhost:9000/localhost:$portS/g" /scratch/s7550245/chromiums/chromium$3/plugin/script.js
../../AppDir51/bin/ld-linux-x86-64.so.2 --library-path ../../AppDir51/bin ../../AppDir51/bin/chrome --disable-setuid-sandbox --single-process --disable-gpu --load-extension=/scratch/s7550245/chromiums/chromium$3/plugin --always-authorize-plugins --disable-infobars --disable-session-crashed-bubble  --user-data-dir=/scratch/s7550245/chromiums/chromium$3  http://chrome.angrybirds.com  &

/sw/global/tools/java/jdk1.8.0_66/bin/java -jar ~/aibird_stuff/Dr.-L.-Bird/ABServer.jar -s $portS -c $portC
