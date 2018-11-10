cd "$(dirname "$0")"

cd ..

python ./code_felix/split/cut.py $1

rm -rf cache/*train_$1*.*
rm -rf cache/*test_$1*.*
