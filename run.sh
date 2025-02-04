for schema in ar en fr ru jp
do
    python evaluate.py --models 'DeepSeek-V3' -mt --schema $schema
done


