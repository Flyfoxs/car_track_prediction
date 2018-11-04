cd "$(dirname "$0")"

cd ..

python ./code_felix/car/split.py $1
rm ./input/*_new.csv
cp ./input/train_train_$1.csv    ./input/train_new.csv
cp ./input/train_validate_$1.csv ./input/test_new.csv

rm -rf cache/*train_train*.*
rm -rf cache/*train_tvalidate*.*