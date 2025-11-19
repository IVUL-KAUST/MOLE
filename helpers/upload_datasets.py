from datasets import Dataset, DatasetDict

mextract_dpo = DatasetDict({
    'train': Dataset.from_parquet('datasets/dpo_train.parquet'),
    'validation': Dataset.from_parquet('datasets/dpo_valid.parquet')
})

mextract_dpo.push_to_hub('IVUL-KAUST/mextract_dpo')

mextract_sft = DatasetDict({
    'train': Dataset.from_parquet('datasets/sft_train.parquet'),
    'validation': Dataset.from_parquet('datasets/sft_valid.parquet'),
    'test': Dataset.from_parquet('datasets/sft_test.parquet')
})

mextract_sft.push_to_hub('IVUL-KAUST/mextract_sft')

mextract_papers = DatasetDict({
    'train': Dataset.from_parquet('datasets/all_papers.parquet'),
})

mextract_papers.push_to_hub('IVUL-KAUST/mextract_papers')