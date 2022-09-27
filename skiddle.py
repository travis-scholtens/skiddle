from bs4 import BeautifulSoup #type:ignore
from community import community_louvain #type:ignore
from dataclasses import dataclass 
import itertools
import networkx #type:ignore
import os
import trueskill #type:ignore
from typing import *
from urllib import parse, request

@dataclass
class Url:
  url: str

@dataclass
class Division:
  name: str

@dataclass
class Player:
  name: str

@dataclass
class Club:
  name: str

@dataclass
class Team:
  name: str
  club: Club
  roster: List[Player]

get_url_page: Callable[[Url], BeautifulSoup] = lambda url: BeautifulSoup(request.urlopen(url.url).read())

def get_rosters() -> Dict[Division, List[Team]]:
  pass

def bootstrap(url: Url) -> None:
  return

def update_skills() -> None:
  return

def update_pti() -> None:
  return

if __name__ == '__main__':
  if os.environ.get('BOOTSTRAP'):
    bootstrap(Url(os.environ['HOME_URL']))
  else:
    update_skills()
    update_pti()
