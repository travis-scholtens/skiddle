from community import community_louvain #type:ignore
from dataclasses import dataclass 
import itertools
import networkx #type:ignore
import os
import trueskill #type:ignore
from typing import *

@dataclass
class Url:
  url: str

def bootstrap(url: Url):
  pass

def updated_skills() -> None:
  pass

def updated_pti() -> None:
  pass

if __name__ == '__main__':
  if os.environ.get('BOOTSTRAP'):
    bootstrap(Url(os.environ['HOME_URL']))
  else:
    update_skills()
    update_pti()
