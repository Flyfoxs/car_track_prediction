cd "$(dirname "$0")"

cd ..

#rm -rf ./cache/*3gp237*
#rm -rf ./cache/*4gp95*
python ./code_felix/ensemble/stacking.py $* >> st_$1.log 2>&1 &
