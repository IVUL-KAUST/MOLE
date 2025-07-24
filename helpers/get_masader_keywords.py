from datasets import load_dataset
import glob
import json
from nltk.corpus import stopwords
from tqdm import tqdm

def preprocess_word(word):
    for char in '?!.,:':
        word = word.replace(char, '')
    return word.lower().strip()

def get_words_from_paper(paper):
    words = []
    for word in paper.split(' '):
        word = preprocess_word(word)
        if word not in stop_words and word != '':
            words.append(word)
    return words


keywords = []
files = glob.glob('/ibex/ai/home/alyafez/masader/datasets/*.json')
stop_words = set(stopwords.words('english'))
for file in tqdm(files):
    data = json.load(open(file, 'r'))
    if data['Paper Title'] is None or data['Abstract'] is None:
        continue
    paper_text = data['Paper Title'] + ' ' + data['Abstract']
    paper_words = get_words_from_paper(paper_text)
    keywords.extend(paper_words)

from collections import Counter
counts = Counter(keywords).most_common(100)
counts = {word: count for word, count in counts if word not in stop_words}

# join the counts for words with an s
new_counts = {}
words = list(counts.keys())
words = sorted(words)
no_s_words = []
i = 0
while i < len(words):
    no_s_words.append(words[i])
    if i+1 < len(words) and words[i+1] == words[i]+'s':
        i += 2
    else:
        i += 1

for word in no_s_words:
    new_counts[word] = counts[word]
    if word+'s' in counts:
        new_counts[word] += counts[word+'s']
sorted_counts = sorted(new_counts.items(), key=lambda x: x[1], reverse=True)
print(sorted_counts[:100])



