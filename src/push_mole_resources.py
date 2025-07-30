from datasets import load_dataset, DatasetDict, concatenate_datasets

train_dataset = load_dataset('csv', data_files='data/train_dataset.csv', split='train')
test_dataset = load_dataset('csv', data_files='data/test_dataset.csv', split='train')
test_dataset = test_dataset.add_column('reasoning', ['human annotated']*len(test_dataset))

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

print("mole resources")
print(dataset)
dataset.push_to_hub('IVUL-KAUST/mole-resources', private=True)
papers_arxiv = load_dataset('csv', data_files='data/all_papers.csv', split='train')

concatenated_dataset = concatenate_datasets([papers_arxiv])
concatenated_dataset = concatenated_dataset.remove_columns(['id'])
print("mole papers")
print(concatenated_dataset)
concatenated_dataset.push_to_hub('IVUL-KAUST/mole-papers', private=True)
