import pandas as pd
from datasets import load_dataset

df_extra_train = pd.DataFrame(columns=['text', 'label'])
extra_sources = []

try:
    sarcasm_raw = load_dataset('liamvbetts/sarcastic-news-headlines-1')
    first_key = list(sarcasm_raw.keys())[0]
    rows = []
    for item in sarcasm_raw[first_key]:
        text = str(item.get('headline', '')).strip()
        if text:
            rows.append({'text': text, 'label': int(item.get('is_sarcastic', 0))})
    df_sarcasm = pd.DataFrame(rows)
    extra_sources.append(df_sarcasm)
    print(f'Sarcasm: {len(df_sarcasm)} examples')
    print(df_sarcasm['label'].value_counts())
except Exception as e:
    print(f'Sarcasm failed: {e}')

try:
    misinfo_raw = load_dataset('daviddaubner/misinformation-detection')
    split_key = 'train' if 'train' in misinfo_raw else list(misinfo_raw.keys())[0]
    rows = []
    for item in misinfo_raw[split_key]:
        text = str(item.get('text', '')).strip()
        if text:
            rows.append({'text': text, 'label': int(item.get('label', 0))})
    df_misinfo = pd.DataFrame(rows)
    extra_sources.append(df_misinfo)
    print(f'\nMisinformation: {len(df_misinfo)} examples')
    print(df_misinfo['label'].value_counts())
except Exception as e:
    print(f'Misinformation failed: {e}')

try:
    scam_raw = load_dataset('FredZhang7/all-scam-spam')
    first_key = list(scam_raw.keys())[0]
    rows = []
    for item in scam_raw[first_key]:
        text = str(item.get('text', '')).strip()
        if text and len(text) <= 200:
            rows.append({'text': text, 'label': int(item.get('is_spam', 0))})
    df_scam = pd.DataFrame(rows)
    extra_sources.append(df_scam)
    print(f'\nScam/spam: {len(df_scam)} examples')
    print(df_scam['label'].value_counts())
except Exception as e:
    print(f'Scam/spam failed: {e}')

if extra_sources:
    df_extra_train = pd.concat(extra_sources, ignore_index=True)
    print(f'\nTotal extra: {len(df_extra_train)}')
    print(df_extra_train['label'].value_counts())
