# agreement eval
uv run plots.py --eval test --schema all --results_path static/results_maged --ignore_length --group_by language 

# main results by language
uv run plots.py --eval test --schema all  --group_by language --non_browsing

# context length
uv run plots.py --eval test --schema all  --group_by language --type context_length --non_browsing

# length
uv run plots.py --eval test --schema all --length --results_path static/results_latex --non_browsing
uv run plots.py --eval test --schema all --length --results_path static/results_length_mid --non_browsing
uv run plots.py --eval test --schema all --length --results_path static/results_length_high --non_browsing

# results each year
uv run plots.py --eval test --schema all  --year  --non_browsing

# other metrics (f1, precision, recall)
uv run plots.py --eval test --schema all  --other_metrics  --non_browsing

# results by cutoff
python plots.py --eval test --schema all  --group_by language --results_path static/results_cutoff --ignore_length