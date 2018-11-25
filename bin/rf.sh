cd "$(dirname "$0")"

cd ..

#rm -rf ./cache/*3gp237*
#rm -rf ./cache/*4gp95*
nohup python code_felix/car/val_rf.py $* > rf_$1.log 2>&1 &
