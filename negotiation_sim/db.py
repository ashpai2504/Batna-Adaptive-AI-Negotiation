"""SQLite ledger for the mid-tier distributor node.

The environment is the only writer; agents interact through read-only queries
(``nominal_capacity`` / ``effective_capacity``). A downstream [SUBMIT_DEAL] or
[ACCEPT_DEAL] is valid only if a capacity query was issued in the same turn and
the committed quantity fits within nominal capacity.
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY, tier TEXT, name TEXT, lat REAL, lon REAL
);
CREATE TABLE IF NOT EXISTS inventory (
    episode_id INTEGER, node_id TEXT, sku TEXT, on_hand INTEGER,
    PRIMARY KEY (episode_id, node_id, sku)
);
CREATE TABLE IF NOT EXISTS upstream_contracts (
    episode_id INTEGER, turn INTEGER, supplier_id TEXT, sku TEXT,
    qty INTEGER, unit_price REAL, status TEXT
);
CREATE TABLE IF NOT EXISTS downstream_contracts (
    episode_id INTEGER, turn INTEGER, buyer_id TEXT, sku TEXT, qty INTEGER,
    price_level INTEGER, lead_days INTEGER, pay_days INTEGER, status TEXT
);
CREATE TABLE IF NOT EXISTS shock_state (
    episode_id INTEGER, turn INTEGER, z REAL, event_id TEXT, disruption_type TEXT,
    expected_fill REAL, tau_t REAL, spot_multiplier REAL,
    PRIMARY KEY (episode_id, turn)
);
CREATE TABLE IF NOT EXISTS action_log (
    episode_id INTEGER, turn INTEGER, channel TEXT, actor TEXT,
    action TEXT, payload TEXT, valid INTEGER
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY, agent TEXT, condition TEXT, scenario INTEGER,
    scenario_seed INTEGER, buyer_type TEXT, disruption_type TEXT, severity REAL, shock_onset_turn INTEGER
);
CREATE TABLE IF NOT EXISTS settlement (
    episode_id INTEGER PRIMARY KEY, delivered_upstream INTEGER,
    committed_downstream INTEGER, shortfall INTEGER
);
"""

NOMINAL_CAPACITY_SQL = """
SELECT i.on_hand
     + COALESCE((SELECT SUM(u.qty) FROM upstream_contracts u
                 WHERE u.episode_id = i.episode_id AND u.status = 'SECURED'), 0)
     - COALESCE((SELECT SUM(d.qty) FROM downstream_contracts d
                 WHERE d.episode_id = i.episode_id AND d.status = 'COMMITTED'), 0)
FROM inventory i
WHERE i.episode_id = ? AND i.node_id = 'DIST' AND i.sku = ?
"""

# Shock-adjusted capacity: in-transit upstream quantity is discounted by the
# expected fill rate implied by the latest observed shock state.
EFFECTIVE_CAPACITY_SQL = """
SELECT i.on_hand
     + COALESCE((SELECT SUM(u.qty) FROM upstream_contracts u
                 WHERE u.episode_id = i.episode_id AND u.status = 'SECURED'), 0)
       * COALESCE((SELECT s.expected_fill FROM shock_state s
                   WHERE s.episode_id = i.episode_id AND s.turn <= ?
                   ORDER BY s.turn DESC LIMIT 1), 1.0)
     - COALESCE((SELECT SUM(d.qty) FROM downstream_contracts d
                 WHERE d.episode_id = i.episode_id AND d.status = 'COMMITTED'), 0)
FROM inventory i
WHERE i.episode_id = ? AND i.node_id = 'DIST' AND i.sku = ?
"""


class NetworkDB:
    def __init__(self, path: str = ":memory:", log_actions: bool = True):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.log_actions = log_actions
        self.queries = 0
        self.conn.executemany(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?)",
            [
                ("SUP", "tier-2 supplier", "Upstream raw-material supplier", None, None),
                ("DIST", "tier-1 distributor", "Phoenix DC", 33.45, -112.07),
                ("BUY", "customer", "Downstream OEM buyer", None, None),
            ],
        )

    def reset_episode(self, ep: int, on_hand: int, sku: str = "SKU-A"):
        c = self.conn
        for t in ("inventory", "upstream_contracts", "downstream_contracts", "shock_state", "action_log", "settlement",
                  "episodes"):
            c.execute(f"DELETE FROM {t} WHERE episode_id = ?", (ep,))
        c.execute("INSERT INTO inventory VALUES (?,?,?,?)", (ep, "DIST", sku, on_hand))

    def write_shock(self, ep, t, z, event_id, dtype, expected_fill, tau_t, spot_mult):
        self.conn.execute(
            "INSERT OR REPLACE INTO shock_state VALUES (?,?,?,?,?,?,?,?)",
            (ep, t, z, event_id, dtype, expected_fill, tau_t, spot_mult),
        )

    def secure_upstream(self, ep, t, qty, price, sku="SKU-A"):
        self.conn.execute(
            "INSERT INTO upstream_contracts VALUES (?,?,?,?,?,?,?)", (ep, t, "SUP", sku, qty, price, "SECURED")
        )

    def commit_downstream(self, ep, t, pkg, sku="SKU-A"):
        q, p, lead, pay = pkg
        self.conn.execute(
            "INSERT INTO downstream_contracts VALUES (?,?,?,?,?,?,?,?,?)",
            (ep, t, "BUY", sku, int(q), int(p), int(lead), int(pay), "COMMITTED"),
        )

    def nominal_capacity(self, ep, sku="SKU-A") -> float:
        self.queries += 1
        return float(self.conn.execute(NOMINAL_CAPACITY_SQL, (ep, sku)).fetchone()[0])

    def effective_capacity(self, ep, t, sku="SKU-A") -> float:
        self.queries += 1
        return float(self.conn.execute(EFFECTIVE_CAPACITY_SQL, (t, ep, sku)).fetchone()[0])

    def log(self, ep, t, channel, actor, action, payload="", valid=1):
        if self.log_actions:
            self.conn.execute(
                "INSERT INTO action_log VALUES (?,?,?,?,?,?,?)", (ep, t, channel, actor, action, str(payload), valid)
            )

    def register_episode(self, ep, agent, condition, scenario, r: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?)",
            (ep, agent, condition, scenario, r["seed"], r["buyer_type"],
             None if r["disruption_type"] == "none" else r["disruption_type"],
             r["severity"], None if r["t_s"] < 0 else r["t_s"]),
        )

    def settle(self, ep, delivered, committed, shortfall):
        self.conn.execute("INSERT OR REPLACE INTO settlement VALUES (?,?,?,?)", (ep, delivered, committed, shortfall))

    def commit(self):
        self.conn.commit()
