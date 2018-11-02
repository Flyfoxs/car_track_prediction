cd "$(dirname "$0")"

cd ..

python ./code_felix/car/split.py $1

rm -rf cache/*train_train*.*
rm -rf cache/*train_tvalidate*.*