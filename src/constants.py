import json
from glob import glob

MODEL_NAMES = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "jury",
    "composer",
    "gemini-2.0-flash-exp",
    "gemini-exp-1206",
]

examplers = []

TEST_DATASETS_IDS_AR = [
    "1709.07276",  # MGB-3
    "1610.00572",  # TED Arabic-Hebrew
    "1509.06928",  # adi-5
    "2201.06723",  # EmojisAnchors
    "2402.07448",  # araspider
    "2210.12985",  # maknuune
    "2106.10745",  # calliar
    "1812.10464",  # laser
    "2103.09687",  # doda
    "2004.14303",  # tunizi
    "2005.06608",  # AraDangspeech
    "1808.07674",  # arap-tweet
    "2106.03193",  # flores-101
    "1612.08989",  # shamela
    "1610.09565",  # transliteration
    "1809.03891",  # OpenITI-proc
    "1411.6718",   # labr
    "1410.3791",   # polygot-ner
    "2309.12053",  # acva
    "1907.03110",  # anetac
    "2407.19835",  # athar
]
VALID_DATASETS_IDS_AR = [
    "1609.05625",  # mgb-2
    "2402.03177",  # cidar
    "2405.01590",  # 101 billion
    "2402.12840",  # arabicmmlu
    "1906.00591",  # winomt
    "2308.16884",  # belebele
]

TEST_DATASETS_IDS_JP = [
    "2202.01764",  # jequad
    "2404.09260",  # JaFIn
    "1911.10668",  # JParaCrawl
    "1710.10639",  # JESC
    "1711.00354",  # JUST
    "2403.19454",  # jdocqa
    "1705.00823",  # stair captions
    "1703.05916",  # jsimilarity
    "2002.08595",  # kaokore
    "2305.12720",  # llm-japaense
    "2305.11444",  # arukikata
    "2112.09323",  # jtubespeech
    "2404.17733",  # ljwc
    "2306.10727",  # jamp
    "2204.02718",  # fakenews-japanese
    "2306.17399",  # JaLeCoN
    "2201.08038",  # japanese-gec
    "2403.17319",  # JMultiWOZ
    "2309.12676",  # JCoLA
    "2107.13998",  # JAFFE
    "2403.19259",  # J-CRe3

]

TEST_DATASETS_IDS_EN = [
    "2501.14249",  # hle
    "2110.14168",  # gsm8k
    "2009.03300",  # mmlu
    "1905.07830",  # hellaswag
    "1705.03551",  # triviaqa
    "2306.01116",  # refineweb
    "1911.11641",  # piqa
    "1907.10641",  # winograd
    "1606.05250",  # squad
    "2406.01574",  # mmlu-pro
    "2311.12022",  # gpqa
    "2305.07759",  # tinystories
    "1806.03822",  # squadv2
    "1905.10044",  # boolq
    "2407.12883",  # bright
    "2504.13161",  # climbmix
    "1606.06031",  # lambada
    "1707.06209",  # sciq
    "1704.04683",  # race
    "1809.09600",  # hotpotqa
    "2005.00547",  # goemotions
]

TEST_DATASETS_IDS_FR = [
    "2002.06071",  # fquad
    "2311.16840",  # cfdd
    "2108.11792",  # bsard
    "2007.00968",  # piaf
    "2304.04280",  # FrenchMedMCQA
    "2504.15941",  # FiarTranslate
    "2207.08292",  # PxSLU
    "2302.07738",  # Alloprof
    "2403.19727",  # intent-media
    "2005.05075",  # lockdown-fr
    "2309.10770",  # FRASIMED
    "2109.13209",  # FQuAD2.0
    "2312.04843",  # FREDSum
    "2501.17117",  # HISTOIRESMORALES
    "2202.09452",  # FREEM
    "1809.00388",  # MTNT
    "2406.17566",  # FrenchToxicityPrompts
    "2403.16099",  # obsinfox
    "2407.11828",  # vigravox
    "2504.21677",  # 20mins-xd
    "2204.05208",  # FIJO
]

TEST_DATASETS_IDS_RU = [
    "2005.10659",  # rubq
    "2010.02605",  # DaNetQA
    "2106.10161",  # golos
    "2108.13112",  # NEREL
    "2210.12814",  # rucola
    "2403.17553",  # RuBia
    "2405.07886",  # russiansummarization
    "2108.12626",  # HeadlineCaust
    "2112.02325",  # Russian Jeopardy
    "2204.08009",  # WikiOmnia
    "2209.13750",  # RuDSI
    "2503.13102",  # REPA
    "2406.19232",  # RuBLiMP
    "2305.14527",  # Solvo
    "2303.16531",  # RusTitW
    "2010.06436",  # RuSemShift
    "2006.11063",  # gazeta
    "1912.09723",  # SberQuAD
    "2502.07455",  # RusCode
    "2501.05841",  # RFSD
    "2505.04851",  # CRAFT
]

TEST_DATASETS_IDS_MULTI = [
    "2010.11856",  # xor-tydi
    "1809.05053",  # xnli
    "1910.07475",  # mlqa
    "2004.06465",  # DE-LIMIT
    "2010.02573",  # marc
    "2103.01910",  # multi-subs
    "2303.08954",  # PRESTO
    "2210.01613",  # mintaka 
    "2306.17674",  # X-RiSAWOZ
    "2211.05958",  # minion 
    "2205.10400",  # multi2woz
    "2211.05955",  # MEE
    "2104.08655",  # DiS-ReX
    "2305.13194",  # SEAHORSE
    "2304.00913",  # LAHM
    "2008.09335",  # MTOP
    "2005.00333",  # XCOPA
    "2004.14900",  # MLSUM
    "2003.08385",  # X-stance
    "2407.12336",  # M2DS
    "2010.09381",  # RELX
]

TEST_DATASETS_IDS_MOD = [
    "2309.16609",  # qwen
]

non_browsing_models = [
    "human",
    "dummy",
    "jury",
    "composer",
    "baseline-first",
    "baseline-last",
    "baseline-random",
    "baseline-keyword",
    "baseline-qa",
]

open_router_costs = {
    "x-ai/grok-4": {"input_tokens": 3, "output_tokens": 15},
    "moonshotai/kimi-k2": {"input_tokens": 0.14, "output_tokens": 2.49},
    "google/gemini-2.5-pro": {"input_tokens": 1.25, "output_tokens": 10},
    "openai/gpt-5": {"input_tokens": 1.25, "output_tokens": 10},
    "openai/gpt-5-chat": {"input_tokens": 1.25, "output_tokens": 10},

}



OPENROUTER_MODELS = [
    # "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "google/gemini-2.5-pro-exp-03-25:free",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-r1:free",
    "google/gemma-3-27b-it",
    "deepseek/deepseek-chat",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    "qwen/qwq-32b:free"
]