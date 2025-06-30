# agreement eval
python plots.py --eval test --schema all --results_path static/results_maged --ignore_length --group_by language 

# main results by language
python plots.py --eval test --schema all  --group_by language --non_browsing

# results each year
python plots.py --eval test --schema all  --year  --non_browsing

# other metrics (f1, precision, recall)
python plots.py --eval test --schema all  --other_metrics  --non_browsing

# results by cutoff
python plots.py --eval test --schema all  --group_by language --results_path static/results_cutoff --ignore_length