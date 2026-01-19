import sys, os
sys.path.insert(0, os.path.abspath('.'))
from polymerSubject import normalize_name
print(normalize_name('poly(propylene glycol)'))
print(normalize_name('poly(ethylene glycol)'))
print(normalize_name('Poly(propylene glycol)'))
