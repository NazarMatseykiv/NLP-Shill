from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.menu import MenuMatcher
from src.slots import SlotExtractor
from src.preprocess import tokenize

m = MenuMatcher(Path('data/menu.json'))
se = SlotExtractor(m)
text = '2 burgers and 1 fries to 12 Main Street, card'
text_n = text.lower()
toks = tokenize(text_n)
print('tokens:', toks)
print('max_alias_len', se.max_alias_len)
print('\nTrying matches:')
for i in range(len(toks)):
    for n in range(1, se.max_alias_len+1):
        if i+n>len(toks):
            continue
        phrase = ' '.join(toks[i:i+n])
        item = m.match_dish(phrase)
        if item:
            print('match', i, n, phrase, '->', item['name'])

print('\nExtract result:')
print(se.extract(text).to_dict())
