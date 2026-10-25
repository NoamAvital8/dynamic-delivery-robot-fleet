"""Durable append-only SQLite replay; immutable JSON rows remain audit evidence."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .provenance import validate_row


@contextmanager
def connection(root):
    path = Path(root) / "replay.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS rows (seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, "
                   "family TEXT NOT NULL, episode TEXT NOT NULL, step INTEGER NOT NULL, "
                   "science TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL, "
                   "chosen TEXT NOT NULL, actor_kind TEXT NOT NULL, city TEXT NOT NULL, profile TEXT NOT NULL, "
                   "regret REAL NOT NULL, margin REAL NOT NULL, "
                   "UNIQUE(family,episode,step))")
        db.execute("CREATE INDEX IF NOT EXISTS rows_family ON rows(family,seq)")
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.commit()
        yield db
    finally:
        db.close()


def record(root, row):
    validate_row(row)
    if row["split"] != "train":
        raise ValueError("Only training rows enter replay")
    payload = json.dumps(row, sort_keys=True, allow_nan=False)
    with connection(root) as db, db:
        db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES ('science',?)", (row["science_id"],))
        if db.execute("SELECT value FROM metadata WHERE key='science'").fetchone()[0] != row["science_id"]:
            raise ValueError("Replay contains another scientific campaign")
        existing = db.execute("SELECT id,digest,payload FROM rows WHERE family=? AND episode=? AND step=?",
                              (row["family"], row["episode"], row["step"])).fetchone()
        if existing:
            if existing != (row["id"], row["content_sha256"], payload):
                raise ValueError("An immutable replay row changed")
            return
        ranked = sorted(row["costs"].values())
        chosen = row["intent"]["chosen_policy"]
        db.execute("INSERT INTO rows(id,family,episode,step,science,digest,payload,chosen,actor_kind,city,profile,regret,margin) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (row["id"], row["family"], row["episode"], row["step"], row["science_id"], row["content_sha256"], payload,
                    chosen, row["intent"].get("model_kind", "unknown"), row["city"], row.get("profile", "unknown"),
                    row["costs"][chosen] - ranked[0], ranked[1] - ranked[0]))


def statistics(root, science):
    with connection(root) as db:
        stored = db.execute("SELECT value FROM metadata WHERE key='science'").fetchone()
        if stored is not None and stored[0] != science:
            raise ValueError("Incompatible replay science")
        answer = {family: {"count": count, "watermark": watermark, "mean_teacher_regret":regret,
                          "mean_teacher_margin":margin, "bootstrap_states":bootstrap}
                  for family, count, watermark, regret, margin, bootstrap in db.execute(
                      "SELECT family,COUNT(*),MAX(seq),AVG(regret),AVG(margin),SUM(actor_kind='bootstrap') FROM rows GROUP BY family")}
        for family in answer:
            answer[family]["policy_states"] = dict(db.execute("SELECT chosen,COUNT(*) FROM rows WHERE family=? GROUP BY chosen", (family,)))
            answer[family]["city_states"] = dict(db.execute("SELECT city,COUNT(*) FROM rows WHERE family=? GROUP BY city", (family,)))
            answer[family]["profile_states"] = dict(db.execute("SELECT profile,COUNT(*) FROM rows WHERE family=? GROUP BY profile", (family,)))
        return answer


def snapshot(root, family, science, episodes, watermark):
    """A fit sees a fixed committed prefix, not a directory changing underneath it."""
    with connection(root) as db:
        rows = db.execute("SELECT id,digest,payload FROM rows WHERE family=? AND seq<=? ORDER BY seq",
                          (family, watermark)).fetchall()
    result = []
    for key, digest, payload in rows:
        row = validate_row(json.loads(payload), episodes, science, family)
        if row["id"] != key or row["content_sha256"] != digest:
            raise ValueError("Replay catalog identity mismatch")
        result.append(row)
    return result
