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
import math
import networkx #type:ignore
import os
import re
import trueskill #type:ignore
from typing import Callable, Dict, Generator, Iterable, List, NewType, Optional, Tuple, TypeAlias, Union
from urllib import parse, request

@dataclass(frozen=True)
class Url:
  url: str

League = NewType('League', str)

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

Cohort = NewType('Cohort', int)

FirestoreClient: TypeAlias = firestore._FirestoreClient

database: Callable[[Dict], FirestoreClient] = (
    lambda cert: firestore.client(
        firebase_admin.initialize_app(
            credentials.Certificate(cert))))

def get_url_page(url: Url) -> BeautifulSoup:
  print(f'Parsing {url.url}')
  return BeautifulSoup(request.urlopen(url.url).read())

def expand_fn(url: Url) -> tuple[League, Callable[[bs4.Tag], Url]]:
  uri = parse.urlparse(url.url)
  host = f'{uri.scheme}://{uri.netloc}'
  return (League(uri.netloc.split('.')[0]),
          lambda link: Url(host + link['href']))

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

def write_matches(matches: List[Match], league: League, db: FirestoreClient) -> None:
  db.collection('matches').document(league).set({
      'archive': [repr(match) for match in matches]
  })
  print(f'Wrote {len(matches)} matches')

def read_matches(league: League, db: FirestoreClient) -> List[Match]:
  matches = [eval(match) for match in
      db.collection('matches').document(league).get().get('archive')
  ]
  print(f'Read {len(matches)} matches')
  return matches

def identify_cohorts(matches: list[Match]) -> dict[Player, Cohort]:
  edges = set()
  for match in matches:
    edges |= {tuple(sorted(ps, key=lambda p: p.name)) for ps in itertools.product(match.home, match.away)}
  return community_louvain.best_partition(
      networkx.Graph(edges))

def draw(match: Match) -> bool:
  return sum([abs(h-a) for (h,a) in match.score])/len(match.score) < 2

def rating_env(matches: list[Match]) -> trueskill.TrueSkill:
  return trueskill.TrueSkill(draw_probability=len([m for m in matches if draw(m)])/len(matches))

def ranks(match: Match) -> tuple[int, int]:
  if draw(match):
    return (0,0)
  elif set_winner(match.score[-1]) == 0:
    return (0,1)
  else:
    return (1,0)

def cohort_skill(
    env: trueskill.TrueSkill,
    cohorts: dict[Player, Cohort],
    matches: list[Match]
    ) -> dict[Cohort, dict[Player, trueskill.Rating]]:
  skills: Dict[Cohort, Dict[Player, trueskill.Rating]] = {}
  for cohort in sorted(set(cohorts.values())):
    cohort_skill: dict[Player, trueskill.Rating] = {}
    skills[cohort] = cohort_skill
    for match in matches:
      if not any([cohorts.get(player) == cohort
                  for player in (match.home + match.away)]):
        continue
      (home, away) = env.rate(
          (cohort_skill.get(match.home[0], env.create_rating()),
           cohort_skill.get(match.home[1], env.create_rating())),
          (cohort_skill.get(match.away[0], env.create_rating()),
           cohort_skill.get(match.away[1], env.create_rating())),
          ranks=ranks(match))
      cohort_skill[match.home[0]] = home[0]
      cohort_skill[match.home[1]] = home[1]
      cohort_skill[match.away[0]] = away[0]
      cohort_skill[match.away[1]] = away[1]
  return skills

def mean_rating(ratings: list[trueskill.Rating]) -> trueskill.Rating:
  return trueskill.Rating(mu=sum([r.mu for r in ratings])/len(ratings),
                          sigma=math.sqrt(sum([r.sigma * r.sigma for r in ratings])/len(ratings)))

def mean_skill_delta(from_skill: Dict[Player, trueskill.Rating], to_skill: Dict[Player, trueskill.Rating]) -> trueskill.Rating:
  if from_skill == to_skill:
    return trueskill.Rating(mu=0, sigma=0)
  overlap = set(from_skill) & set(to_skill)
  if not overlap:
    return None
  from_mean = mean_rating([from_skill[p] for p in overlap])
  to_mean = mean_rating([to_skill[p] for p in overlap])
  return trueskill.Rating(mu=to_mean.mu - from_mean.mu,
                          sigma=math.sqrt(from_mean.sigma * from_mean.sigma + to_mean.sigma * to_mean.sigma))

def cohort_deltas(
    cohorts: set[Cohort],
    skills: dict[Cohort, dict[Player, trueskill.Rating]]
    ) -> dict[tuple[Cohort, Cohort], trueskill.Rating]:
  return {
    (from_cohort, to_cohort): mean_skill_delta(skills[from_cohort], skills[to_cohort])
    for (from_cohort, to_cohort)
    in itertools.product(cohorts, cohorts)
  }

def rating(
    env: trueskill.TrueSkill,
    skill: Optional[trueskill.Rating],
    delta: Optional[trueskill.Rating]) -> trueskill.Rating:
  if not skill or not delta:
    return env.create_rating()
  return trueskill.Rating(mu=skill.mu + delta.mu,
                          sigma=math.sqrt(skill.sigma * skill.sigma + delta.sigma * delta.sigma))

def main_cohort(cohorts: dict[Player, Cohort], players: set[Player]) -> Cohort:
  cohort_counts = [0 for _ in range(len(set(cohorts.values())))]
  for player in players:
    if player not in cohorts:
      continue
    cohort_counts[cohorts[player]] += 1
  return Cohort(max(range(len(cohort_counts)), key=lambda i: cohort_counts[i]))

def division_ratings(
    divisions: dict[Division, set[Player]],
    matches: list[Match]) -> dict[Division, dict[Player, trueskill.Rating]]:
  cohorts = identify_cohorts(matches)
  env = rating_env(matches)
  skills = cohort_skill(env, cohorts, matches)
  deltas = cohort_deltas(set(cohorts.values()), skills)
  ratings: dict[Division, dict[Player, trueskill.Rating]] = {}
  for (division, players) in divisions.items():
    target_cohort = main_cohort(cohorts, players)
    ratings[division] = {
        player: rating(
            env,
            skills[cohorts[player]].get(player),
            deltas.get((cohorts[player], target_cohort)))
        for player in players if player in cohorts
    }
  return ratings

def read_division(team: BeautifulSoup) -> Division:
  return Division(next(team
      .find('div', class_='team_nav')
      .stripped_strings))

def read_team_and_club_name(team: BeautifulSoup) -> Tuple[str, str]:
  ss = team.find(id='home_right').stripped_strings
  next(ss)
  return (next(ss), next(ss))

def read_players(team: BeautifulSoup, name: str) -> Generator[Player, None, None]:
  for tr in team.find('th', string=lambda s: name in s).find_parent('table').find_all('tr'):
    if tr.find('th'):
      if next(tr.stripped_strings) not in (name, 'Captains', 'Players', 'Players Also Subbing for Other Teams'):
        break
      else:
        continue
    ss = tr.stripped_strings
    while ss:
      s = next(ss)
      if s not in '✔12345' and s != 'Captain':
        yield Player(s)
        break

def get_team(page: BeautifulSoup) -> tuple[Division, Team]:
  (name, club) = read_team_and_club_name(page)
  players = list(read_players(page, name))
  return (read_division(page),
          Team(name, Club(club), players))

def get_teams(links: list[bs4.Tag],
              get_link: Callable[[bs4.Tag], BeautifulSoup]
             ) -> List[Tuple[Division, Team]]:
  return [get_team(get_link(link)) for link in links]

def get_rosters(home: BeautifulSoup,
                get_link: Callable[[bs4.Tag], BeautifulSoup]
               ) -> Dict[Division, List[Team]]:
  return (lambda teams: {
              division: [team for (d, team) in teams if d == division]
              for division in {division for (division, _) in teams}
          })(get_teams([d.find('a') for d in home.find_all('div', class_='div_list_teams_option')], get_link))

def players(teams: list[Team]) -> set[Player]:
  ps = set()
  for team in teams:
    ps |= set(team.roster)
  return ps

def bootstrap(home: BeautifulSoup,
              league: League,
              get_link: Callable[[bs4.Tag], BeautifulSoup],
              db: FirestoreClient) -> None:
   write_matches(
       archive_matches(get_link(home.find('a', string='Archives')),
                       get_link),
       league,
       db)

def update_skills() -> None:
  return

def update_pti() -> None:
  return

if __name__ == '__main__':
  db = database(json.loads(os.environ['FIREBASE_CERT']))
  url = Url(os.environ['HOME_URL'])
  home = get_url_page(url)
  (league, expand) = expand_fn(url)
  get_link = lambda link: get_url_page(expand(link))
  if os.environ.get('BOOT_STRAP'):
    print('Bootstrapping')
    bootstrap(home, league, get_link, db)
  else:
    print('Setting initial skill')
    matches = read_matches(league, db)
    rosters = get_rosters(home, get_link)
    ratings = division_ratings(
        {division: players(teams) for (division, teams) in rosters.items()},
        matches)
    print('Updating')
    update_skills()
    update_pti()
