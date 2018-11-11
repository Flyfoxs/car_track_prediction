#Switch to china timezone
env  TZ='/usr/share/zoneinfo/Asia/Chongqing' date >> sub.log
ls ./output/sub/*.csv  >> sub.log
echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" >> sub.log
mv ./output/sub/*.csv /home/code/result.csv
wget -q -O - http://122.112.238.70:10008/submit?ssh=32919
rm /home/code/result.csv