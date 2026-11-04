import asyncio

import pytest

from server.bots import PROFILES, choose_word
from server.game import MAX_PLAYERS, GameError, Room

LINKS = {("ocean", "wave"), ("wave", "hello"), ("wave", "surf"), ("wave", "water"),
         ("ocean", "water"), ("hello", "goodbye"), ("ocean", "blue"), ("ocean", "surf")}


class FakeModel:
    words = ["ocean", "wave", "hello", "goodbye", "surf", "water", "waters", "blue", "piano"]
    index = {w: i for i, w in enumerate(words)}

    def in_vocab(self, w): return w in self.index
    def normalize(self, w): return {"waters": "water"}.get(w, w)
    def threshold(self, s): return 0.5
    def random_seed_word(self, banned="rst"): return "ocean"
    def banned_share(self, banned): return 0.1

    def relatedness(self, a, b):
        return 0.9 if (a, b) in LINKS or (b, a) in LINKS else 0.1

    def candidates(self, prev, exclude=frozenset(), k=50, allow_forbidden=False, banned="rst"):
        out = [(w, self.relatedness(prev, w)) for w in self.words
               if w != prev and self.normalize(w) not in exclude
               and (allow_forbidden or w[0] not in banned)]
        return sorted(out, key=lambda x: -x[1])[:k]

    def forbidden_pressure(self, w, used, banned="rst"): return 0.0
    def forbidden_pressure_many(self, words, used, banned="rst"): return [0.0] * len(words)


async def _noop(room, msg):
    pass


def make_room(n=2):
    room = Room("ABCDE", FakeModel(), _noop)
    players = [room.add_player(f"P{i}") for i in range(n)]
    return room, players


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_validate_rules():
    async def go():
        room, (a, b) = make_room()
        await room.start()
        room._cancel_tasks()
        assert room.last_word == "ocean"
        assert room.validate("surf")[1] == "starts with S"
        assert room.validate("piano")[1] == "not related enough"
        assert room.validate("xyzzy")[1] == "not in dictionary"
        assert room.validate("ocean")[1] == "already used"
        assert room.validate("wave")[0] is True
    run(go())


def test_inflection_counts_as_repeat_and_failed_move_costs_life():
    async def go():
        room, (a, b) = make_room()
        await room.start()
        first = room.current
        await room.submit(first.id, "water")
        second = room.current
        assert second is not first and room.last_word == "water"
        lives = second.lives
        # 'waters' normalises to 'water' -> repeat
        room.chain[-1].word = "ocean"  # make it otherwise related
        await room.submit(second.id, "waters")
        assert second.lives == lives - 1
        assert room.history[-1].reason == "already used"
        room._cancel_tasks()
    run(go())


def test_not_your_turn():
    async def go():
        room, (a, b) = make_room()
        await room.start()
        other = b if room.current is a else a
        with pytest.raises(GameError):
            await room.submit(other.id, "wave")
        room._cancel_tasks()
    run(go())


def test_last_player_standing_wins():
    async def go():
        room, (a, b) = make_room()
        room.update_settings({"lives": 1})
        await room.start()
        loser = room.current
        await room.submit(loser.id, "piano")
        assert room.phase == "over"
        assert room.winner_id == (b.id if loser is a else a.id)
    run(go())


def test_room_caps_at_seven():
    room, _ = make_room(MAX_PLAYERS)
    with pytest.raises(GameError):
        room.add_bot("pro")


def test_bot_never_plays_forbidden_or_used_when_not_blundering():
    m = FakeModel()
    for diff in PROFILES:
        for _ in range(30):
            w = choose_word(m, "wave", {"ocean"}, diff, 0.5, pressure=0.0)
            if PROFILES[diff].blunder == 0:
                assert w in {"hello", "water"}
            if w is not None and w[0] not in "rst" and w != "ocean":
                assert m.relatedness("wave", w) >= 0.5 or PROFILES[diff].blunder > 0


def test_turn_clock_tightens():
    async def go():
        room, (a, b) = make_room()
        room.update_settings({"turnSeconds": 15})
        await room.start()
        t0 = room.turn_length
        await room.submit(room.current.id, "wave")
        assert room.turn_length < t0
        room._cancel_tasks()
    run(go())


def test_disconnect_grace_and_host_transfer():
    async def go():
        room, (a, b) = make_room()
        t = room.disconnect(a.id)
        assert a in room.players and room.host_id == a.id   # grace period
        room.drop_if_away(a.id, t)
        assert a not in room.players and room.host_id == b.id
    run(go())


def test_stale_grace_timer_does_not_drop_returning_player():
    async def go():
        room, (a, b) = make_room()
        t1 = room.disconnect(a.id)
        await room.reconnect(a.id)
        await asyncio.sleep(0.01)
        room.disconnect(a.id)
        room.drop_if_away(a.id, t1)           # first timer fires late
        assert a in room.players and room.host_id == a.id
    run(go())


def test_solo_refresh_pauses_and_resumes():
    async def go():
        room, (a,) = make_room(1)
        room.add_bot("pro")
        await room.start()
        t = room.disconnect(a.id)
        assert room.phase == "playing" and room.paused
        await room.reconnect(a.id)
        assert not room.paused and room.phase == "playing"
        room.disconnect(a.id)
        room.drop_if_away(a.id, room.get(a.id).away_since)
        assert room.phase == "over"           # left for good
        room._cancel_tasks()
    run(go())


def test_back_to_lobby_only_after_game():
    async def go():
        room, (a, b) = make_room()
        await room.start()
        with pytest.raises(GameError):
            await room.back_to_lobby()
        room._cancel_tasks()
    run(go())


def test_current_player_disconnect_gets_autoplay_and_reconnect_cancels_it():
    async def go():
        room, (a, b, c) = make_room(3)
        await room.start()
        cur = room.current
        room.disconnect(cur.id)
        assert room._bot_task is not None and not room._bot_task.done()
        await room.reconnect(cur.id)
        await asyncio.sleep(0)
        assert room._bot_task.cancelled() or room._bot_task.done()
        room._cancel_tasks()
    run(go())


def test_custom_banned_letters():
    async def go():
        room, (a, b) = make_room()
        room.update_settings({"banned": "WHB"})
        assert room.settings.banned == "bhw"
        await room.start()
        room._cancel_tasks()
        assert room.validate("wave")[1] == "starts with W"
        assert room.validate("surf")[0] is True            # S is allowed now
        for bad in ["ab", "aab", "a1c", "abcd"]:
            with pytest.raises(GameError):
                room.settings.banned = "rst"; room.phase = "lobby"; room.update_settings({"banned": bad})
    run(go())


def test_bots_respect_custom_letters():
    m = FakeModel()
    for _ in range(30):
        w = choose_word(m, "ocean", {"ocean"}, "grandmaster", 0.5, banned="wbh")
        assert w is None or w[0] not in "wbh"


def test_host_end_game_ranks_by_points_then_lives_then_misses():
    async def go():
        room, (a, b, c) = make_room(3)
        await room.start()
        room._cancel_tasks()
        a.score, b.score, c.score = 30, 40, 40
        b.lives, c.lives = 2, 2
        b.misses, c.misses = 1, 0
        await room.end_game()
        assert room.phase == "over" and room.ended_by == "host"
        assert [p.id for p in room.standings()] == [c.id, b.id, a.id]
        assert room.winner_id == c.id
        with pytest.raises(GameError):
            await room.end_game()
    run(go())


def test_last_standing_survivor_ranks_first_even_with_fewer_points():
    async def go():
        room, (a, b) = make_room()
        room.update_settings({"lives": 1})
        await room.start()
        loser = room.current
        winner = b if loser is a else a
        loser.score = 999
        await room.submit(loser.id, "piano")
        assert room.standings()[0] is winner
    run(go())


def test_scoring_is_base_plus_speed_bonus():
    async def go():
        room, (a, b) = make_room()
        await room.start()
        cur = room.current
        await room.submit(cur.id, "wave")                  # answered instantly
        assert cur.score == 15
        room._cancel_tasks()
    run(go())
