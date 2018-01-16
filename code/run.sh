#!/bin/bash



CS=("GOOG" "AAPL" "AMZN" "MSFT"  )
TYPES=("last_price" "volatil")

for C in "${CS[@]}"
do
  for TYPE in "${TYPES[@]}"
  do
    echo "==================="
    echo "Runing..." $C $TYPE
    echo "==================="
    python week3_context_predict.py --companies $C --timeseries_type $TYPE
  done
done

