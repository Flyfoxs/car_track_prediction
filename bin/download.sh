cd "$(dirname "$0")"

cd ..

rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/track_prediciton/output/merge*.csv ./output/sub/

rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/track_prediciton/output/500/*all*.* ./output/500/


rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/track_prediciton/output/ensemble/*.h5 ./output/ensemble/


rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/track_prediciton/cache/*worse* ./cache/

rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/felix/track_prediciton/cache/reduce_address_*500*all*.* ./cache/

