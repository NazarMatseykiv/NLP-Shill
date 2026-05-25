from pathlib import Path
import sys
# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.menu import MenuMatcher

MROOT = Path(__file__).resolve().parent.parent
m = MenuMatcher(MROOT / 'data' / 'menu.json')
short = [k for k in m.choices if len(k.split()) <= 1 and len(k) <= 2]
print('Count short:', len(short))
for k in short:
    print(repr(k), '->', m.alias_to_item[k]['name'])

print('\nCheck for "на":', 'на' in m.alias_to_item)
if 'на' in m.alias_to_item:
    print('на ->', m.alias_to_item['на']['name'])
