from pathlib import Path
import csv
import json
import re

PATH = Path(__file__).resolve().parents[1] / 'data' / 'intents_merged.csv'
if not PATH.exists():
    raise SystemExit(f'Missing {PATH}')

langs = {'uk':0, 'en':0, 'other':0}
intents = {}
sources = {}
unique_texts = set()

cyrillic_re = re.compile('[\u0400-\u04FF]')
latin_re = re.compile('[A-Za-z]')

with PATH.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = (row.get('text') or '').strip()
        intent = (row.get('intent') or '').strip()
        source = (row.get('source') or '').strip()
        if not text:
            continue
        unique_texts.add(text)
        intents[intent] = intents.get(intent, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        if cyrillic_re.search(text):
            langs['uk'] += 1
        elif latin_re.search(text):
            langs['en'] += 1
        else:
            langs['other'] += 1

summary = {
    'merged_path': str(PATH),
    'total_rows': sum(intents.values()),
    'unique_texts': len(unique_texts),
    'by_intent': intents,
    'by_source': sources,
    'by_language_guess': langs,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
