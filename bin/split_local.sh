cd "$(dirname "$0")"

cd ..

python ./code_felix/car/split.py $1
rm ./input/*_new.csv
cp ./input/train_train.csv    ./input/train_new.csv
cp ./input/train_validate.csv ./input/test_new.csv

rm -rf cache/*train_train*.*
rm -rf cache/*train_tvalidate*.*