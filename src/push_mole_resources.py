from datasets import load_dataset, DatasetDict, concatenate_datasets

train_dataset = load_dataset('csv', data_files='data/train_dataset.csv', split='train')
test_dataset = load_dataset('csv', data_files='data/test_dataset.csv', split='train')
test_dataset = test_dataset.add_column('reasoning', ['']*len(test_dataset))

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

print(dataset)
# dataset.push_to_hub('IVUL-KAUST/mole-resources', private=True)
papers_arxiv = load_dataset('csv', data_files='data/arxiv_papers.csv', split='train')
papers_acl = load_dataset('csv', data_files='data/acl_papers.csv', split='train')

concatenated_dataset = concatenate_datasets([papers_arxiv, papers_acl])

print(concatenated_dataset)
# concatenated_dataset.push_to_hub('IVUL-KAUST/mole-papers', private=True)
