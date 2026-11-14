"""Room + game state machine for RST.

Rules
-----
* Players take turns saying a word related to the previous word.
* The word may not start with one of the room's three banned letters (R, S, T
  unless the host picks others), must be a real word, and may not have been
  used already in this round (inflections count as the same word).
* A failed move or a timeout costs a life. The next player then answers the
  same word. Lose every life and you are out; the last player standing wins.
* Each accepted word scores 10 points plus up to 5 for answering fast. If the
  host ends the game early, the leader on points wins (ties: lives, then misses).
* Team mode: 2-4 teams share a pool of lives each. Turns go round-robin over the
  teams still alive and each team rotates through its own members, so every team
  gets the same number of turns however uneven the sizes. Last team standing (or
  the team with most points if the host ends early) wins.
* The clock tightens: every accepted word cuts the turn time by 4%, down to
  MIN_TURN seconds, so even perfect players eventually crack.
"""
from __future__ import annotations

import asyncio
import random
import secrets
import string
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

MAX_PLAYERS = 7
MIN_TURN = 4.0
SPEEDUP = 0.96
DEFAULT_BANNED = "rst"
BASE_POINTS, SPEED_POINTS = 10, 5
# Room codes skip R/S/T (house style) and look-alike characters.
CODE_ALPHABET = "".join(c for c in string.ascii_uppercase if c not in "RSTOIL") + "23456789"

DIFFICULTIES = ["rookie", "casual", "sharp", "pro", "grandmaster"]
TEAM_NAMES = ["Ink", "Ember", "Moss", "Tide"]
BOT_NAMES = ["Ada", "Basil", "Cleo", "Dex", "Echo", "Fable", "Gus", "Hopper",
             "Iris", "Juno", "Kip", "Lumen", "Mabel", "Nova", "Otto", "Pip"]


@dataclass
class Player:
    id: str
    name: str
    is_bot: bool = False
    difficulty: str | None = None
    lives: int = 3
    score: int = 0
    misses: int = 0
    team: int | None = None
    connected: bool = True
    away_since: float | None = None
    token: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    @property
    def alive(self) -> bool:
        return self.lives > 0

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "isBot": self.is_bot,
                "difficulty": self.difficulty, "lives": self.lives,
                "score": self.score, "misses": self.misses, "team": self.team,
                "connected": self.connected}


@dataclass
class Settings:
    turn_seconds: int = 15
    lives: int = 3
    strictness: str = "normal"  # lenient | normal | strict
    banned: str = DEFAULT_BANNED  # three distinct lowercase letters, sorted
    mode: str = "solo"  # solo | teams
    teams: int = 2      # number of teams in team mode (2-4)

    def public(self) -> dict:
        return {"turnSeconds": self.turn_seconds, "lives": self.lives,
                "strictness": self.strictness, "banned": self.banned,
                "mode": self.mode, "teams": self.teams}


@dataclass
class Move:
    player_id: str
    word: str
    ok: bool
    reason: str | None = None
    score: float | None = None
    t: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"playerId": self.player_id, "word": self.word, "ok": self.ok,
                "reason": self.reason,
                "score": None if self.score is None else round(self.score, 3)}


class Room:
    def __init__(self, code: str, model, broadcast: Callable[["Room", dict], Awaitable[None]]):
        self.code = code
        self.model = model
        self._broadcast = broadcast
        self.players: list[Player] = []
        self.host_id: str | None = None
        self.settings = Settings()
        self.phase = "lobby"  # lobby | playing | over
        self.chain: list[Move] = []      # accepted words, chain[0] is the seed
        self.history: list[Move] = []    # every attempt, for the feed
        self.used: set[str] = set()      # normalised forms already played
        self.turn_idx = 0
        self.turn_deadline: float | None = None
        self.turn_length: float = 0.0
        self.turn_no = 0
        self.winner_id: str | None = None
        self.ended_by: str | None = None  # "last" (one survivor) | "host" (ended early)
        self.paused = False  # every human dropped mid-game; resumes when one returns
        # Team mode state
        self.team_lives: dict[int, int] = {}
        self.team_ptr: dict[int, int] = {}   # next member to play, per team
        self.winner_team: int | None = None
        self._timer: asyncio.Task | None = None
        self._bot_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.created = time.time()
        self.last_activity = time.time()

    # ---------------------------------------------------------------- lobby
    def add_player(self, name: str, *, is_bot=False, difficulty=None) -> Player:
        if len(self.players) >= MAX_PLAYERS:
            raise GameError("Room is full (7 players max).")
        if self.phase != "lobby":
            raise GameError("Game already started.")
        p = Player(id=secrets.token_hex(4), name=name[:18], is_bot=is_bot,
                   difficulty=difficulty, lives=self.settings.lives)
        if self.teams_mode:
            p.team = self._smallest_team()
        self.players.append(p)
        if self.host_id is None and not is_bot:
            self.host_id = p.id
        return p

    def add_bot(self, difficulty: str) -> Player:
        if difficulty not in DIFFICULTIES:
            raise GameError("Unknown difficulty.")
        taken = {p.name for p in self.players}
        name = next((n for n in random.sample(BOT_NAMES, len(BOT_NAMES)) if n not in taken), "Bot")
        return self.add_player(name, is_bot=True, difficulty=difficulty)

    def disconnect(self, pid: str) -> float | None:
        """Mark a player away and return the disconnect stamp for drop_if_away.
        In the lobby they keep their seat for a grace period (a refresh keeps your
        seat and host status); mid-game a bot covers their turns."""
        p = self.get(pid)
        if p is None:
            return None
        p.connected = False
        p.away_since = time.time()
        if self.phase == "playing":
            if not self.humans_connected():
                # Nobody left to watch: freeze the game until someone comes back.
                self._cancel_tasks()
                self.paused = True
            elif self.current is p:
                self._start_autoplay(p)
        return p.away_since

    async def reconnect(self, pid: str):
        p = self.get(pid)
        if p is None:
            return
        p.connected, p.away_since = True, None
        if self.host_id is None:
            self.host_id = p.id
        if self.phase == "playing" and self.paused:
            self.paused = False
            # Restart the interrupted turn with a fresh clock.
            await self._begin_turn()
        elif self.current is p and self._bot_task and not self._bot_task.done():
            self._bot_task.cancel()  # you're back: your turn is yours again

    def drop_if_away(self, pid: str, since: float | None):
        p = self.get(pid)
        if p is None or p.connected or p.away_since != since:
            return  # came back (and maybe left again) since this timer was set
        if self.phase == "lobby":
            self.players.remove(p)
        elif self.paused:
            self.phase = "over"  # everyone left for good
        if self.host_id == pid:
            humans = [q for q in self.players if not q.is_bot and q.connected]
            self.host_id = humans[0].id if humans else None

    # ---------------------------------------------------------------- teams
    @property
    def teams_mode(self) -> bool:
        return self.settings.mode == "teams"

    def members(self, team: int) -> list[Player]:
        return [p for p in self.players if p.team == team]

    def _smallest_team(self) -> int:
        return min(range(self.settings.teams), key=lambda t: (len(self.members(t)), t))

    def _rebalance(self):
        """Give every player a valid team; called when team mode or count changes."""
        for p in self.players:
            if p.team is None or p.team >= self.settings.teams:
                p.team = None
        for p in self.players:
            if p.team is None:
                p.team = self._smallest_team()

    def set_team(self, pid: str, team: int):
        if self.phase != "lobby":
            raise GameError("Teams are locked once the game starts.")
        if not self.teams_mode:
            raise GameError("Switch to team mode first.")
        p = self.get(pid)
        if p is None:
            raise GameError("No such player.")
        if not (0 <= team < self.settings.teams):
            raise GameError("No such team.")
        p.team = team

    def is_alive(self, p: Player) -> bool:
        if self.teams_mode:
            return self.team_lives.get(p.team, 0) > 0
        return p.lives > 0

    def _lose_life(self, p: Player):
        p.misses += 1
        if self.teams_mode:
            self.team_lives[p.team] -= 1
        else:
            p.lives -= 1

    def active_teams(self) -> list[int]:
        return [t for t in range(self.settings.teams) if self.members(t)]

    def team_stats(self, t: int) -> dict:
        ms = self.members(t)
        return {"team": t, "name": TEAM_NAMES[t], "lives": self.team_lives.get(t, self.settings.lives),
                "score": sum(p.score for p in ms), "misses": sum(p.misses for p in ms),
                "players": [p.id for p in ms]}

    def team_standings(self) -> list[int]:
        def key(t):
            st = self.team_stats(t)
            return (t != self.winner_team if self.ended_by == "last" else 0,
                    -st["score"], -st["lives"], st["misses"])
        return sorted(self.active_teams(), key=key)

    def humans_connected(self) -> bool:
        return any(q.connected and not q.is_bot for q in self.players)

    def get(self, pid: str) -> Player | None:
        return next((p for p in self.players if p.id == pid), None)

    def update_settings(self, data: dict):
        if self.phase != "lobby":
            raise GameError("Settings are locked once the game starts.")
        s = self.settings
        if "turnSeconds" in data:
            s.turn_seconds = int(min(60, max(5, int(data["turnSeconds"]))))
        if "lives" in data:
            s.lives = int(min(5, max(1, int(data["lives"]))))
        if data.get("strictness") in ("lenient", "normal", "strict"):
            s.strictness = data["strictness"]
        if data.get("mode") in ("solo", "teams"):
            s.mode = data["mode"]
        if "teams" in data:
            s.teams = int(min(len(TEAM_NAMES), max(2, int(data["teams"]))))
        if s.mode == "teams":
            self._rebalance()
        if "banned" in data:
            letters = str(data["banned"]).lower()
            if len(letters) != 3 or len(set(letters)) != 3 or not all("a" <= c <= "z" for c in letters):
                raise GameError("Pick three different letters.")
            s.banned = "".join(sorted(letters))
        for p in self.players:
            p.lives = s.lives

    # ----------------------------------------------------------------- game
    @property
    def current(self) -> Player | None:
        if self.phase != "playing" or not self.players:
            return None
        return self.players[self.turn_idx]

    @property
    def last_word(self) -> str | None:
        return self.chain[-1].word if self.chain else None

    async def start(self):
        if self.phase == "playing":
            raise GameError("Already playing.")
        if len(self.players) < 2:
            raise GameError("Need at least 2 players. Add an AI player?")
        if self.teams_mode:
            self._rebalance()
            if len(self.active_teams()) < 2:
                raise GameError("Put players in at least two teams.")
        self.phase = "playing"
        self.paused = False
        self.chain, self.history, self.used = [], [], set()
        self.winner_id = self.winner_team = None
        self.ended_by = None
        for p in self.players:
            p.lives, p.score, p.misses = self.settings.lives, 0, 0
        seed = self.model.random_seed_word(self.settings.banned)
        self.chain.append(Move(player_id="seed", word=seed, ok=True))
        self.used.add(self.model.normalize(seed))
        if self.teams_mode:
            teams = self.active_teams()
            self.team_lives = {t: self.settings.lives for t in teams}
            self.team_ptr = {t: random.randrange(len(self.members(t))) for t in teams}
            self._set_team_turn(random.choice(teams))
        else:
            self.team_lives, self.team_ptr = {}, {}
            self.turn_idx = random.randrange(len(self.players))
        self.turn_no = 0
        await self._begin_turn()

    async def submit(self, pid: str, raw: str):
        if self.phase != "playing":
            raise GameError("No game in progress.")
        cur = self.current
        if cur is None or cur.id != pid:
            raise GameError("Not your turn.")
        word = (raw or "").strip().lower()
        ok, reason, score = self.validate(word)
        move = Move(player_id=pid, word=word or "…", ok=ok, reason=reason, score=score)
        self.history.append(move)
        if ok:
            self.chain.append(move)
            self.used.add(self.model.normalize(word))
            left = max(0.0, (self.turn_deadline or 0) - time.time())
            cur.score += BASE_POINTS + round(SPEED_POINTS * min(1.0, left / max(0.1, self.turn_length)))
        else:
            self._lose_life(cur)
        await self._advance(last_move=move)

    def validate(self, word: str) -> tuple[bool, str | None, float | None]:
        if not word or not word.isalpha():
            return False, "letters only", None
        if word[0] in self.settings.banned:
            return False, f"starts with {word[0].upper()}", None
        if len(word) < 2:
            return False, "too short", None
        if not self.model.in_vocab(word):
            return False, "not in dictionary", None
        norm = self.model.normalize(word)
        if norm in self.used:
            return False, "already used", None
        prob = self.model.relatedness(self.last_word, word)
        if prob < self.model.threshold(self.settings.strictness):
            return False, "not related enough", prob
        return True, None, prob

    async def _timeout(self, turn_no: int):
        try:
            await asyncio.sleep(self.turn_length + 0.4)
        except asyncio.CancelledError:
            return
        async with self.lock:
            if self.phase != "playing" or self.turn_no != turn_no:
                return
            cur = self.current
            self._lose_life(cur)
            move = Move(player_id=cur.id, word="⏱", ok=False, reason="time's up")
            self.history.append(move)
            await self._advance(last_move=move)

    def _set_team_turn(self, team: int):
        ms = self.members(team)
        self.turn_idx = self.players.index(ms[self.team_ptr[team] % len(ms)])
        self.team_ptr[team] += 1

    async def _game_over(self, last_move: Move):
        self.phase = "over"
        self.ended_by = "last"
        self._cancel_tasks()
        await self.push({"type": "move", "move": last_move.public()})
        await self.push_state()

    async def _advance(self, last_move: Move):
        if self.teams_mode:
            alive_teams = [t for t in self.active_teams() if self.team_lives[t] > 0]
            if len(alive_teams) <= 1:
                self.winner_team = alive_teams[0] if alive_teams else None
                return await self._game_over(last_move)
            cur_team = self.players[self.turn_idx].team
            order = self.active_teams()
            i = order.index(cur_team)
            nxt = next(order[(i + k) % len(order)] for k in range(1, len(order) + 1)
                       if self.team_lives[order[(i + k) % len(order)]] > 0)
            self._set_team_turn(nxt)
        else:
            alive = [p for p in self.players if self.is_alive(p)]
            if len(alive) <= 1:
                self.winner_id = alive[0].id if alive else None
                return await self._game_over(last_move)
            n = len(self.players)
            for step in range(1, n + 1):
                idx = (self.turn_idx + step) % n
                if self.is_alive(self.players[idx]):
                    self.turn_idx = idx
                    break
        await self.push({"type": "move", "move": last_move.public()})
        await self._begin_turn()

    async def _begin_turn(self):
        self._cancel_tasks()
        self.turn_no += 1
        base = self.settings.turn_seconds
        self.turn_length = max(min(MIN_TURN, base), base * SPEEDUP ** (len(self.chain) - 1))
        self.turn_deadline = time.time() + self.turn_length
        self._timer = asyncio.create_task(self._timeout(self.turn_no))
        await self.push_state()
        cur = self.current
        if cur and (cur.is_bot or not cur.connected):
            self._start_autoplay(cur)

    def _start_autoplay(self, player: Player):
        from .bots import play_bot_turn  # local import avoids a cycle
        if self._bot_task and not self._bot_task.done():
            self._bot_task.cancel()
        self._bot_task = asyncio.create_task(play_bot_turn(self, player, self.turn_no))

    def _cancel_tasks(self):
        current = asyncio.current_task()
        for t in (self._timer, self._bot_task):
            if t and not t.done() and t is not current:
                t.cancel()

    def standings(self) -> list[Player]:
        """Final order. A survivor of a played-out game comes first; otherwise points,
        then lives left, then fewer misses. In team mode, players are grouped by
        their team's rank."""
        if self.teams_mode and self.team_lives:
            rank = {t: i for i, t in enumerate(self.team_standings())}
            return sorted(self.players, key=lambda p: (rank.get(p.team, 99), -p.score, p.misses))
        def key(p):
            return (p.id != self.winner_id if self.ended_by == "last" else 0,
                    -p.score, -p.lives, p.misses)
        return sorted(self.players, key=key)

    async def end_game(self):
        """Host stops the game early: the leader on points wins."""
        if self.phase != "playing":
            raise GameError("No game in progress.")
        self._cancel_tasks()
        self.phase, self.paused, self.ended_by = "over", False, "host"
        if self.teams_mode:
            self.winner_team = self.team_standings()[0]
        else:
            self.winner_id = self.standings()[0].id
        await self.push_state()

    async def back_to_lobby(self):
        if self.phase != "over":
            raise GameError("Finish the game first.")
        self._cancel_tasks()
        self.phase = "lobby"
        self.players = [p for p in self.players if p.connected or p.is_bot]
        for p in self.players:
            p.lives, p.score, p.misses = self.settings.lives, 0, 0
        self.chain, self.history = [], []
        self.winner_id = self.winner_team = self.ended_by = None
        self.team_lives, self.team_ptr = {}, {}
        if self.teams_mode:
            self._rebalance()
        await self.push_state()

    # ------------------------------------------------------------- output
    def snapshot(self) -> dict:
        return {
            "code": self.code,
            "phase": self.phase,
            "hostId": self.host_id,
            "players": [p.public() for p in self.players],
            "settings": self.settings.public(),
            "chain": [m.public() for m in self.chain],
            "history": [m.public() for m in self.history[-40:]],
            "currentId": self.current.id if self.current else None,
            "deadline": self.turn_deadline,
            "turnLength": self.turn_length,
            "serverTime": time.time(),
            "winnerId": self.winner_id,
            "endedBy": self.ended_by,
            "winnerTeam": self.winner_team,
            "teamNames": TEAM_NAMES[: self.settings.teams],
            "teams": [self.team_stats(t) for t in range(self.settings.teams)] if self.teams_mode else None,
            "teamStandings": self.team_standings() if self.teams_mode and self.phase == "over" else None,
            "standings": [p.id for p in self.standings()] if self.phase == "over" else None,
            "bannedShare": round(self.model.banned_share(self.settings.banned), 3),
            "paused": self.paused,
            "maxPlayers": MAX_PLAYERS,
        }

    async def push_state(self):
        await self.push({"type": "state", "state": self.snapshot()})

    async def push(self, msg: dict):
        self.last_activity = time.time()
        await self._broadcast(self, msg)


class GameError(Exception):
    pass


def new_code(existing) -> str:
    while True:
        code = "".join(random.choice(CODE_ALPHABET) for _ in range(5))
        if code not in existing:
            return code
