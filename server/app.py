"""FastAPI app: static site + WebSocket rooms.

Run:  .venv/bin/uvicorn server.app:app --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .game import DIFFICULTIES, GameError, Room, new_code
from .nlp import AssociationModel

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
ARTIFACTS = Path(os.environ.get("RST_ARTIFACTS", ROOT / "artifacts" / "current"))

app = FastAPI(title="RST")
model = AssociationModel(ARTIFACTS)
rooms: dict[str, Room] = {}
sockets: dict[str, dict[str, "Conn"]] = {}  # room code -> player id -> connection


class Conn:
    """One client socket with an ordered outbox drained by its own writer task, so
    broadcasts never interleave and one slow client can't stall the room."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
        self.writer = asyncio.create_task(self._drain())

    def send(self, msg: dict | str):
        data = msg if isinstance(msg, str) else json.dumps(msg)
        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:  # hopelessly behind: cut it loose, it can reconnect
            self.close()

    async def _drain(self):
        try:
            while (data := await self.queue.get()) is not None:
                await self.ws.send_text(data)
        except Exception:
            pass

    def close(self):
        self.writer.cancel()
        asyncio.create_task(self._close_ws())

    async def _close_ws(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def broadcast(room: Room, msg: dict):
    data = json.dumps(msg)
    for conn in list(sockets.get(room.code, {}).values()):
        conn.send(data)


@app.get("/api/health")
def health():
    return {"ok": True, "rooms": len(rooms), "model": model.meta.get("version"),
            "metrics": model.meta.get("metrics", {})}


@app.post("/api/rooms")
def create_room():
    code = new_code(rooms)
    rooms[code] = Room(code, model, broadcast)
    return {"code": code}


@app.get("/api/rooms/{code}")
def room_info(code: str):
    room = rooms.get(code.upper())
    if not room:
        return JSONResponse({"error": "No room with that code."}, 404)
    return {"code": room.code, "phase": room.phase, "players": len(room.players),
            "maxPlayers": 7}


@app.get("/api/check")
def check(prev: str, word: str):
    """Debug/practice endpoint: how related does the model think two words are?"""
    return {"prev": prev, "word": word, "relatedness": model.relatedness(prev.lower(), word.lower())}


@app.websocket("/ws/{code}")
async def ws_endpoint(ws: WebSocket, code: str):
    await ws.accept()
    code = code.upper()
    room = rooms.get(code)
    if room is None:
        await ws.send_text(json.dumps({"type": "error", "message": "No room with that code."}))
        await ws.close()
        return
    player = None
    conn = Conn(ws)
    try:
        hello = json.loads(await ws.receive_text())
        if not isinstance(hello, dict):
            raise GameError("Bad hello.")
        async with room.lock:
            token = hello.get("token")
            player = next((p for p in room.players if token and p.token == token), None)
            if player:
                await room.reconnect(player.id)
            else:
                name = str(hello.get("name") or "").strip() or "Player"
                player = room.add_player(name)
            old_conn = sockets.setdefault(code, {}).get(player.id)
            sockets[code][player.id] = conn
        if old_conn:  # same seat opened in a new tab: the newest connection wins
            old_conn.close()
        conn.send({"type": "welcome", "playerId": player.id, "token": player.token})
        await room.push_state()

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if not isinstance(msg, dict) or len(raw) > 2000:
                    raise GameError("Bad message.")
                await handle(room, player, msg)
            except GameError as e:
                conn.send({"type": "error", "message": str(e)})
            except (ValueError, TypeError, AttributeError):
                conn.send({"type": "error", "message": "Bad message."})
    except GameError as e:
        conn.send({"type": "error", "message": str(e)})
        await asyncio.sleep(0.2)  # let the writer flush the error before closing
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        mine = player is not None and sockets.get(code, {}).get(player.id) is conn
        if mine:
            sockets[code].pop(player.id, None)
        conn.close()
        if mine:
            async with room.lock:
                since = room.disconnect(player.id)
            await room.push_state()
            asyncio.create_task(_drop_later(room, player.id, since))


async def _drop_later(room: Room, pid: str, since: float | None, grace: float = 20.0):
    await asyncio.sleep(grace)
    async with room.lock:
        room.drop_if_away(pid, since)
    await room.push_state()


async def handle(room: Room, player, msg: dict):
    kind = msg.get("type")
    is_host = player.id == room.host_id
    async with room.lock:
        if kind == "submit":
            await room.submit(player.id, str(msg.get("word", ""))[:32])
        elif kind == "typing":
            await room.push({"type": "typing", "playerId": player.id,
                             "text": str(msg.get("text", ""))[:32]})
        elif kind == "chat":
            text = str(msg.get("text", "")).strip()[:200]
            if text:
                await room.push({"type": "chat", "playerId": player.id, "text": text,
                                 "t": time.time()})
        elif kind == "react":
            await room.push({"type": "react", "playerId": player.id,
                             "emoji": str(msg.get("emoji", ""))[:4]})
        elif not is_host:
            raise GameError("Only the host can do that.")
        elif kind == "add_bot":
            diff = msg.get("difficulty", "casual")
            if diff not in DIFFICULTIES:
                raise GameError("Unknown difficulty.")
            room.add_bot(diff)
            await room.push_state()
        elif kind == "remove":
            target = room.get(msg.get("playerId", ""))
            if target and target.id != player.id and room.phase == "lobby":
                room.players.remove(target)
                kicked = sockets.get(room.code, {}).pop(target.id, None)
                if kicked:
                    kicked.send({"type": "kicked"})
                    asyncio.get_running_loop().call_later(0.3, kicked.close)
                await room.push_state()
        elif kind == "settings":
            room.update_settings(msg.get("settings", {}))
            await room.push_state()
        elif kind == "start":
            await room.start()
        elif kind == "lobby":
            await room.back_to_lobby()
        else:
            raise GameError("Unknown action.")


async def reaper():
    """Drop rooms that have been empty or idle for a while."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for code, room in list(rooms.items()):
            live = sockets.get(code) or {}
            idle = now - room.last_activity
            if (not live and idle > 300) or idle > 6 * 3600:
                room._cancel_tasks()
                for conn in live.values():
                    conn.send({"type": "error", "message": "Room closed for inactivity."})
                    asyncio.get_running_loop().call_later(0.3, conn.close)
                rooms.pop(code, None)
                sockets.pop(code, None)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(reaper())


app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    return FileResponse(WEB / "index.html")
