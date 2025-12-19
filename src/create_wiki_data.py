
from search import run
import arg_utils
import json

model_name = 'moonshotai/kimi-k2'
schema_name = 'bib'
links = [
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://en.wikipedia.org/wiki/Charles_Darwin",
    "https://en.wikipedia.org/wiki/Nikola_Tesla",
    "https://en.wikipedia.org/wiki/Stephen_Hawking",
    "https://en.wikipedia.org/wiki/Michael_Faraday",
    "https://en.wikipedia.org/wiki/James_Clerk_Maxwell",
    "https://en.wikipedia.org/wiki/Max_Planck",
    "https://en.wikipedia.org/wiki/Richard_Feynman",
    "https://en.wikipedia.org/wiki/Niels_Bohr",
    "https://en.wikipedia.org/wiki/Enrico_Fermi",
    "https://en.wikipedia.org/wiki/Paul_Dirac",
    "https://en.wikipedia.org/wiki/John_von_Neumann",
    "https://en.wikipedia.org/wiki/Linus_Pauling",
    "https://en.wikipedia.org/wiki/Dmitri_Mendeleev",
    "https://en.wikipedia.org/wiki/Gregor_Mendel",
    "https://en.wikipedia.org/wiki/Alexander_Fleming",
    "https://en.wikipedia.org/wiki/Rosalind_Franklin",
    "https://en.wikipedia.org/wiki/Thomas_Edison",
    "https://en.wikipedia.org/wiki/Alan_Turing",
    "https://en.wikipedia.org/wiki/Abdus_Salam"
]

# Call your processing function with the file content and link
_args = arg_utils.args
_args.model_name = model_name
_args.schema_name = schema_name
_args.format = 'pdf_plumber'
_args.overwrite = True
_args.log = True

for link in links:
    results = run(link, _args)
    metadata = results[model_name]['metadata']
    metadata['Paper_Link'] = link
    keys = metadata.keys()
    metadata['annotations_from_paper'] = {}
    for key in keys:
        if key == 'annotations_from_paper':
            continue
        metadata['annotations_from_paper'][key] = 1
    with open(f'evals/bib/test/{link.split("/")[-1]}.json', 'w') as f:
        json.dump(metadata, f, indent=4)