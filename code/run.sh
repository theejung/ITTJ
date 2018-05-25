#!/bin/bash



CS=("GOOG" "AAPL" "AMZN" "MSFT"  )
TYPES=("last_price" "volatil")
GPU=0
for C in "${CS[@]}"
do
  for TYPE in "${TYPES[@]}"
  do
    echo "==================="
    echo "Runing..." $C $TYPE
    echo "==================="
    CUDA_VISIBLE_DEVICES="$GPU" \
      python week3_context_predict.py --companies $C --timeseries_type $TYPE

  done
done

