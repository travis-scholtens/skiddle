from community import community_louvain
from dataclasses import dataclass 
import itertools
import networkx
import os
import trueskill
from typing import *

@dataclass
class Url:
  url: str

def bootstrap(url: Url):
  pass

def updated_skills():
  pass

def updated_pti():
  pass

if __name__ == '__main__':
  if os.environ.get('BOOTSTRAP'):
    bootstrap(Url(os.environ['HOME_URL']))
  else:
    update_skills()
    update_pti()
