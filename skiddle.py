import bs4 #type:ignore
from bs4 import BeautifulSoup #type:ignore
from community import community_louvain #type:ignore
from dataclasses import dataclass
import datetime
import dateutil
import firebase_admin #type:ignore
from firebase_admin import credentials
from firebase_admin import firestore
from dateutil import parser
import itertools
import json
import networkx #type:ignore
import os
import re
import trueskill #type:ignore
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple, TypeAlias, Union
from urllib import parse, request

@dataclass(frozen=True)
class Url:
  url: str

@dataclass(frozen=True)
class Division:
  name: str

@dataclass(frozen=True)
class Player:
  name: str

@dataclass(frozen=True)
class Club:
  name: str

@dataclass(frozen=True)
class Team:
  name: str
  club: Club
  roster: List[Player]

SetResult = Tuple[int, int]

Score = Tuple # 2 or 3 SetResults

@dataclass(frozen=True)
class Match:
  date: datetime.date
  home: Tuple[Player, Player]
  away: Tuple[Player, Player]
  score: Score

FirestoreClient: TypeAlias = firestore._FirestoreClient

database: Callable[[Dict], FirestoreClient] = (
    lambda cert: firestore.client(
        firebase_admin.initialize_app(
            credentials.Certificate(cert))))

def get_url_page(url: Url) -> BeautifulSoup:
  print(f'Parsing {url.url}')
  return BeautifulSoup(request.urlopen(url.url).read())

def expand_fn(url: Url) -> Callable[[bs4.Tag], Url]:
  uri = parse.urlparse(url.url)
  host = f'{uri.scheme}://{uri.netloc}'
  return lambda link: Url(host + link['href'])

def get_division_paths(year: BeautifulSoup) -> Generator[bs4.Tag, None, None]:
  for option in year.find_all('option'):    
    form = option.find_parent('form')
    params = []
    for control in form.find_all(True):
      if 'name' not in control.attrs:
        continue
      param = control['name']
      if 'value' in control.attrs:
        value_source = control
      elif any([child == option for child in control.descendants]):
        value_source = option
      if value_source:
        param += '=' + value_source['value']
      params.append(param)
    yield bs4.Tag(name='a', attrs={'href': f'{form["action"]}?{"&".join(params)}'})

def transpose_scores(home_scores: Tuple[str, str, str], away_scores: Tuple[str, str, str]) -> Score:
  third_set_scores = {home_scores[2], away_scores[2]}
  sets = 2 if third_set_scores == {''} or third_set_scores == {'0'} else 3
  return tuple([(int(home_scores[i]), int(away_scores[i])) for i in range(sets)])

def split_partners(lines: str) -> list[Player]:
  return [Player(p) for p in sorted([partner.strip() for partner in lines.split('\n')])]

def get_match(date: datetime.date, home_row: bs4.Tag, away_row: bs4.Tag) -> Optional[Match]:
  (_, _, _, home_partners, *home_sets) = [td.get_text().strip() for td in home_row.find_all('td')]
  (_, _, away_partners, *away_sets) = [td.get_text().strip() for td in away_row.find_all('td')]
  home_partners = split_partners(home_partners)
  away_partners = split_partners(away_partners)
  if not (len(home_partners) == len(away_partners) == 2):
    return None
  if not (len(home_sets) == len(away_sets) == 3):
    return None
  return Match(
      date,
      (home_partners[0], home_partners[1]),
      (away_partners[0], away_partners[1]),
      transpose_scores(
          (home_sets[0], home_sets[1], home_sets[2]), 
          (away_sets[0], away_sets[1], away_sets[2]))
  )

def get_individual_matches(match_link: bs4.Tag, get_link: Callable[[bs4.Tag], BeautifulSoup]) -> Generator[Match, None, None]:
  date = parser.parse(match_link.string).date()
  row = get_link(match_link).find('th', string='Line').find_parent('table').find('tr', class_='printrow')
  while row:
    next = row.find_next_sibling('tr', class_='printrowalt')
    match = get_match(date, row, next)
    if match:
      yield match
    row = next.find_next_sibling('tr', class_='printrow')

bye_pattern = re.compile('.* BYE')

def get_schedule_matches(schedule: BeautifulSoup, get_link: Callable[[bs4.Tag], BeautifulSoup]) -> Generator[List[Match], None, None]:
  for tr in schedule.find('th', string='Date').find_parent('table').find_all('tr'):
    if tr.find('th'):
      continue
    items = tr.find_all('td')
    if len(items) < 3:
      continue
    if items[0].find('a', string=bye_pattern):
      continue
    yield list(get_individual_matches(items[0].find('a'), get_link))

def get_team_matches(team: BeautifulSoup, get_link: Callable[[bs4.Tag], BeautifulSoup]) -> List[Match]:
  return sum(get_schedule_matches(get_link(team.find('a', string='Show Schedule')), get_link), [])

def get_division_matches(division: BeautifulSoup, get_link: Callable[[bs4.Tag], BeautifulSoup]) -> List[Match]:
  return sum([get_team_matches(get_link(a), get_link) for a in division.find('th', string='TeamName').find_parent('table').find_all('a', href=lambda s: s != '#')], [])

def get_matches(paths: Iterable[bs4.Tag], get_link: Callable[[bs4.Tag], BeautifulSoup]) -> List[Match]:
  return sum([get_division_matches(get_link(path), get_link) for path in paths], [])

def unique_matches(matches: Iterable[Match]) -> Generator[Match, None, None]:
  seen = set()
  for match in matches:
    key = (match.date, match.home, match.away)
    if key not in seen:
      yield match
      seen.add(key)

def complete_set(result: SetResult) -> bool:
  bigger, smaller = max(*result), min(*result)
  return ((bigger == 6 or bigger >= 10) and (bigger - smaller) >= 2) or (bigger == 7 and smaller in (5,6))

def set_winner(result: SetResult) -> int:
  return max(range(len(result)), key=lambda i: result[i])

def valid_matches(matches: Iterable[Match]) -> Generator[Match, None, None]:
  for match in matches:
    if not all([p for p in match.home + match.away]):
      # At least one player name is empty
      continue
    if len(set(match.home) | set(match.away)) != 4:
      # At least one duplicated player
      continue
    if not all([complete_set(result) for result in match.score]):
      # At least one set not complete
      continue
    if len(match.score) == 2:
      if len({set_winner(result) for result in match.score}) != 1:
        # Split sets with no decider
        continue
    elif len(match.score) == 3:
      if len({set_winner(result) for result in match.score[0:2]}) != 2:
        # Unnecessary third set
        continue
    else:
      # Bad set count
      continue
    yield match

archive_matches: Callable[[BeautifulSoup, Callable[[bs4.Tag], BeautifulSoup]], List[Match]] = (
    lambda archive, get_link: list(
      valid_matches(
          unique_matches(
              get_matches(
                  get_division_paths(
                      get_link(
                          archive.find('h2', string='Seasons:').parent.find('a')
                      )
                  ),
                  get_link
              )
          )
      ))
)

def get_rosters() -> Dict[Division, List[Team]]:
  pass

def bootstrap(url: Url) -> None:
  get_link: Callable[[bs4.Tag], BeautifulSoup] = (
      lambda expand: lambda link: get_url_page(expand(link))
  )(expand_fn(url))
  
  home = get_url_page(url)
  previous_matches = archive_matches(
      get_link(home.find('a', string='Archives')),
      get_link)
  return

def update_skills() -> None:
  return

def update_pti() -> None:
  return

if __name__ == '__main__':
  print(os.environ)
  db = database(json.loads(os.environ['FIREBASE_CERT']))
  if os.environ.get('BOOT_STRAP'):
    print('Bootstrapping')
    bootstrap(Url(os.environ['HOME_URL']))
  else:
    print('Updating')
    update_skills()
    update_pti()
