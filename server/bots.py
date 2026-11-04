"""AI players.

Each difficulty controls three things:
  * how deep in the model's ranked candidate list the bot picks from,
  * how quickly it answers,
  * how often it makes a human-like blunder (an R/S/T word, a repeat, freezing).

Grandmaster also plays strategically: among strong candidates it prefers words
whose own best associations mostly start with a banned letter, leaving the next
player few good moves.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    rank_lo: int        # pick among candidates ranked [rank_lo, rank_hi)
    rank_hi: int
    think: tuple[float, float]  # seconds
    blunder: float      # chance of a mistake on any turn
    strategic: bool = False
    vocab_cap: int | None = None  # only words this common (frequency rank); None = any


PROFILES = {
    "rookie":      Profile(10, 120, (4.0, 9.0), 0.22, vocab_cap=6000),
    "casual":      Profile(3, 50,  (3.0, 6.5), 0.10, vocab_cap=10000),
    "sharp":       Profile(1, 20,  (2.0, 4.5), 0.04, vocab_cap=18000),
    "pro":         Profile(0, 8,   (1.2, 3.0), 0.012),
    "grandmaster": Profile(0, 30,  (0.9, 2.2), 0.0, strategic=True),
}


def choose_word(model, prev: str, used: set[str], difficulty: str,
                threshold: float, rng: random.Random | None = None,
                pressure: float = 0.0, banned: str = "rst") -> str | None:
    """Return the bot's word, or None to freeze (time out).

    `pressure` in [0, 1] is how far the clock has tightened; mistakes get more
    likely under pressure, like they do for people.
    """
    rng = rng or random
    prof = PROFILES[difficulty]

    if rng.random() < prof.blunder * (1 + 3 * pressure) + (0.01 * pressure if prof.blunder else 0):
        kind = rng.choice(["forbidden", "forbidden", "repeat", "freeze"])
        if kind == "forbidden":
            bad = [w for w, _ in model.candidates(prev, exclude=used, k=40, allow_forbidden=True)
                   if w[0] in banned]
            if bad:
                return rng.choice(bad[:8])
        if kind == "repeat" and used:
            return rng.choice(sorted(used))
        return None

    cands = [(w, p) for w, p in model.candidates(prev, exclude=used, k=max(prof.rank_hi, 60) * 2,
                                                 banned=banned)
             if p >= threshold and (prof.vocab_cap is None or model.index[w] < prof.vocab_cap)]
    if not cands:
        return None
    if prof.strategic:
        pool = cands[: prof.rank_hi]
        pressure_after = model.forbidden_pressure_many([w for w, _ in pool], used, banned=banned)
        scores = [0.55 * p + 0.45 * fp for (_, p), fp in zip(pool, pressure_after)]
        return pool[max(range(len(pool)), key=scores.__getitem__)][0]
    lo = min(prof.rank_lo, max(0, len(cands) - 1))
    pool = cands[lo: prof.rank_hi] or cands
    return rng.choice(pool)[0]


async def play_bot_turn(room, player, turn_no: int):
    prof = PROFILES.get(player.difficulty or "casual", PROFILES["casual"])
    limit = room.turn_length
    # Bots feel the clock too: think time shrinks with the turn length.
    scale = min(1.0, limit / max(1.0, room.settings.turn_seconds))
    delay = min(rng_uniform(*prof.think) * (0.5 + 0.5 * scale), max(0.6, limit - 1.0))
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    # Think outside the lock so other players' messages aren't held up.
    if room.phase != "playing" or room.turn_no != turn_no:
        return
    thr = room.model.threshold(room.settings.strictness)
    # Disconnected humans are auto-played at casual level so the game keeps moving.
    diff = player.difficulty if player.is_bot else "casual"
    word = await asyncio.to_thread(choose_word, room.model, room.last_word, set(room.used), diff,
                                   thr, None, 1.0 - scale, room.settings.banned)
    if word is None:
        return  # freeze: the turn timer will expire
    async with room.lock:
        still_ours = (room.phase == "playing" and room.turn_no == turn_no
                      and room.current is player and (player.is_bot or not player.connected))
        if still_ours and time.time() <= (room.turn_deadline or 0):
            await room.submit(player.id, word)


def rng_uniform(a: float, b: float) -> float:
    return random.uniform(a, b)
