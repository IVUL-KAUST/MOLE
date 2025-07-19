import json
import random

template = """
{name}: A {task} dataset for {schema}
{authors}
{affs}
{name}, is a {schema} {task} dataset, that contains {volume} {unit}.
{language_table}
{provider_stmt} The dataset was collected from {collection_style} of {domain} in {year}. 
The dataset is publically available through this link {link}. {license_stmt}.
{hf_stmt}
"""
langs = {"ru": "Russian", "jp": "Japanese", "fr": "french", "en": "English", "ar": "Arabic", "multi": "Multilingual"}

author_names = {
    "en": ["Emily Carter", "James Richardson", "Olivia Bennett", "Daniel Foster", "Sophia Mitchell"],
    "ru": ["Ivan Petrov", "Olga Sokolova", "Dmitry Ivanov", "Natalia Smirnova", "Sergey Volkov"],
    "jp": ["Haruto Takahashi", "Yuki Nakamura", "Souta Yamamoto", "Hina Tanaka", "Rina Fujimoto"],
    "ar": ["Omar Al-Mansouri", "Layla Al-Farsi", "Khalid Al-Najjar", "Fatima Al-Zayani", "Youssef Al-Tamimi"],
    "fr": ["Louis Dupont", "Camille Laurent", "Émile Moreau", "Sophie Dubois", "Jean-Pierre Martin"],
    "multi": ["Emily Carter", "James Richardson", "Olivia Bennett", "Daniel Foster", "Sophia Mitchell"]
}

aff_names = {
    "ru": ["Moscow State University", "Saint Petersburg Polytechnic University", 
           "Novosibirsk State University", "Tomsk Polytechnic University", "Kazan Federal University"],
    
    "jp": ["University of Tokyo", "Kyoto University", "Osaka University", 
           "Tohoku University", "Nagoya University"],
    
    "ar": ["King Fahd University of Petroleum and Minerals", "American University of Beirut", 
           "Cairo University", "Qatar University", "United Arab Emirates University"],
    
    "fr": ["Sorbonne University", "École Polytechnique", "Université de Lyon", 
           "Université de Bordeaux", "Université de Strasbourg"],
    
    "en": ["Harvard University", "University of Oxford", "Massachusetts Institute of Technology", 
           "Stanford University", "University of Cambridge"],
    "multi": ["Harvard University", "University of Oxford", "Massachusetts Institute of Technology", 
           "Stanford University", "University of Cambridge"]
}


def sample(l, size=1):
    for r in ["images", "hours"]:
        if r in l:
            l.remove(r)
    if "other" in l:
        l.remove("other")
    sampled = random.sample(l, size)
    if size == 1:
        return sampled[0]
    return sampled

def createPorivder():
    if random.randint(0,1):
        return ''.join([random.choice('abcdefghijklmnopqrstuvwxyz').capitalize() for _ in range(4)])
    return ''

def createLanguageTable(languages, unit):
    # use latex table format
    subsets = []
    table = """
    \\begin{table}[h]
    \centering
    \\begin{tabular}{|c|c|}
    \hline
    Language & Size \\\\
    \hline
    """
    total_volume = 0.0
    for lang in languages:
        volume = random.randint(1000, 10000)
        total_volume += volume
        table += f"{lang} & {volume} \\\ \n"
        subsets.append({"Name": lang, "Volume": volume, "Unit": unit, "Language": lang})
    table += """ \\hline
    \end{tabular}
    \end{table}
    """
    return table, total_volume, subsets
def createHFLink(provider, name):
    if random.randint(0, 1) and provider:
        return f'https://huggingface.co/datasets/{provider}/{name}'
    else:
        return ''
for lang in ["en", "fr", "jp", "ru", "multi"]:
    for ex in range(1, 5+1):
        with open(f"schema/{lang}.json", "r") as f:
            schema = json.load(f)
        task = sample(schema["Tasks"]["options"])
        domain = sample(schema["Domain"]["options"], random.randint(1, 3))
        if not isinstance(domain, list):
            domain = [domain]
        collection_style = sample(schema["Collection Style"]["options"])
        if 'speech' in task:
            unit = 'hours'
            form = "spoken"
        elif 'task' in ['named entity recognition']:
            unit = 'tokens'
        else:
            unit = 'sentences'
        
        form = "text"

        license = sample(schema["License"]["options"])
        if license == "unknown":
            license_stmt = ""
        else:
            license_stmt = f"This dataset is licensed under {license}."
        volume = float(random.randint(100, 10000))
        authors = sample(author_names[lang], 2)
        affs = sample(aff_names[lang], 2)
        name = lang+"".join([t.capitalize() for t in task.split(" ")])
        corr = "".join([t for t in authors[0].lower().split(" ")])
        link = f"https://github.com/{corr}/{name}"
        if lang == "multi":
            languages = sample(schema["Language"]["options"], random.randint(2, 5))
            language_table, volume, subsets = createLanguageTable(languages, unit)
            language = "Multilingual"

        else:
            language = langs[lang]
            language_table = ""
        year = random.randint(2010, 2025)
        provider = createPorivder()
        
        provider_stmt = ""
        if provider:
            provider_stmt = f"This dataset is provided by {provider}."

        hf_stmt = ""
        hf_link = createHFLink(provider, name)
        if hf_link:
            hf_stmt = f"The dataset is also avaiable in HuggingFace through this link {hf_link}"

        applied_tempalte = template.format(
            name=name,
            license_stmt=license_stmt,
            schema=language,
            task=task,
            volume=volume,
            unit=unit,
            link=link,
            domain=", ".join(domain),
            collection_style=collection_style,
            authors=' '.join([f'{auth}^{i+1}' for i,auth in enumerate(authors)]),
            affs=' '.join([f'{i+1}. {aff}' for i,aff in enumerate(affs)]),
            year = year,
            provider_stmt = provider_stmt,
            hf_stmt = hf_stmt,
            language_table = language_table
        )
        out_json = {
            "Name": name,
            "Link": link,
            "HF Link": hf_link,
            "License": license,
            "Year": year,
            "Language": lang if lang != "multi" else languages,
            "Domain": domain,
            "Form": form,
            "Collection Style": [collection_style],
            "Description": f"{name} is a {task} dataset, that contains {volume} {unit}",
            "Volume": volume,
            "Unit": unit,
            "Ethical Risks": "Low",
            "Provider": [provider],
            "Derived From": [],
            "Paper Title": f"{name}: A {task} dataset for {lang}",
            "Paper Link": "",
            "Tokenized": False,
            "Host": "GitHub",
            "Access": "Free",
            "Cost": "",
            "Test Split": False,
            "Tasks": [task],
            "Venue Title": "arXiv",
            "Venue Type": "preprint",
            "Venue Name": "",
            "Authors": authors,
            "Affiliations": affs,
            "Abstract": ""
        }
        

        if lang == "ar":
            dialect = sample(schema["Dialect"]["options"])
            script  = sample(schema["Script"]["options"])
            out_json["Dialect"] = dialect
            out_json["Script"] = script
            out_json["Subsets"] = []
            print(out_json.keys())
            assert len(out_json.keys()) == 32
        elif lang == "jp":
            out_json["Script"] = sample(schema["Script"]["options"])
            assert len(out_json.keys()) == 30
        elif lang == "multi":
            out_json["Subsets"] = subsets
            assert len(out_json.keys()) == 30
        else:
            assert len(out_json.keys()) == 29
        
        import os

        os.makedirs(f"examples/{lang}", exist_ok=True)
        with open(f"examples/{lang}/example{ex}.tex", "w") as f:
            f.write(applied_tempalte)

        with open(f"examples/{lang}/example{ex}.json", "w") as f:
            json.dump(out_json, f, indent= 4)
