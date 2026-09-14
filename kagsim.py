"""
kagsim.py — a lightweight local re-implementation of the Kaggriculture rules,
written from the competition README so agents can be tested without the
official package. It is NOT the official environment: details may differ.
Use it for fast iteration, then validate with `kaggle_environments`.
"""
import math
import random

CROP = {
    # seed, base, first_yield, max_yield_day, cap, cap_fert, ongoing, window_start, interval
    "WHEAT":      dict(seed=10, first=2, max_day=4, cap=4, cap_fert=6, ongoing=False, win=2),
    "CARROT":     dict(seed=20, first=2, max_day=3, cap=3, cap_fert=4, ongoing=False, win=2),
    "MELON":      dict(seed=80, first=10, max_day=10, cap=6, cap_fert=6, ongoing=False, win=6),
    "TOMATO":     dict(seed=50, first=8, max_day=11, cap=4, cap_fert=4, ongoing=True, interval=1),
    "STRAWBERRY": dict(seed=100, first=10, max_day=16, cap=4, cap_fert=4, ongoing=True, interval=2),
}
ANIMAL = {
    "GOOSE": dict(cost=300, struct="COOP", product="EGG", first=4, interval=1, max_held=4),
    "COW":   dict(cost=400, struct="PASTURE", product="MILK", first=8, interval=2, max_held=6),
    "SHEEP": dict(cost=500, struct="PASTURE", product="WOOL", first=6, interval=3, max_held=6),
}
MARKET = {  # base, I0, T, below_func, below_target, above_func, above_target
    "WHEAT":      (25, 10000, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT":     (35, 10000, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO":     (60, 10000, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON":      (250, 10000, 300, "log", 0.20, "sq", 3.60),
    "EGG":        (50, 10000, 332, "hinge", 0.40, "log", 0.20),
    "MILK":       (160, 10000, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL":       (200, 10000, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 10000, 200, "linear", 0.40, "linear", 0.40),
}
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
QUADS = {"NW": (0, 0), "NE": (1, 0), "SW": (0, 1), "SE": (1, 1)}
LAND_COST = [1000, 2000, 4000]
PRODUCTS = list(MARKET.keys())


def _f(name, x, T):
    if name == "linear":
        return x
    if name == "sq":
        return x * x
    if name == "sqrt":
        return math.sqrt(x)
    if name == "log":
        return math.log1p(x)
    if name == "log10":
        return math.log10(1 + x)
    if name == "hinge":
        u = x / T
        return u + 8 * max(0.0, u - 1) ** 2
    raise ValueError(name)


def price_of(item, inv):
    base, I0, T, bf, bt, af, at = MARKET[item]
    if inv < I0:
        x = I0 - inv
        amp = bt * base / _f(bf, T, T)
        p = base + amp * _f(bf, x, T)
    elif inv > I0:
        x = inv - I0
        amp = at * base / _f(af, T, T)
        p = base - amp * _f(af, x, T)
    else:
        p = base
    return max(1, int(round(p)))


class Farm:
    def __init__(self, n):
        self.n = n
        self.half = n // 2
        self.money = 3000.0
        self.tiles = [[None] * n for _ in range(n)]
        self.unlocked = ["NW"]
        self.farmer = [self.half - 1, self.half - 1]
        self.hands = []
        self.hires_today = 0
        self.shed = {}
        self.seeds = {}
        self.inventories = [{}]

    def locked(self, x, y):
        qx, qy = (1 if x >= self.half else 0), (1 if y >= self.half else 0)
        for q, (a, b) in QUADS.items():
            if (a, b) == (qx, qy):
                return q not in self.unlocked
        return True

    def shed_adjacent(self, pos):
        h = self.half
        return tuple(pos) in {(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)}

    def public(self):
        tiles = []
        for y in range(self.n):
            row = []
            for x in range(self.n):
                if self.locked(x, y):
                    row.append("LOCKED")
                else:
                    t = self.tiles[y][x]
                    row.append(dict(t) if isinstance(t, dict) else t)
            tiles.append(row)
        return dict(money=self.money, tiles=tiles, farmer=list(self.farmer),
                    hands=[list(h) for h in self.hands], unlocked_quadrants=list(self.unlocked),
                    hires_today=self.hires_today)


class Game:
    def __init__(self, seed=None, board=10, steps=720, shed_cap=100, verbose=False):
        self.rng = random.Random(seed)
        self.n = board
        self.steps = steps
        self.shed_cap = shed_cap
        self.farms = [Farm(board), Farm(board)]
        self.inv = {p: 10000 for p in PRODUCTS}
        self.shops = []
        self.step = 0
        self.verbose = verbose
        self.log = []

    # ---------------- observation ----------------
    def obs(self, pid):
        f = self.farms[pid]
        return {
            "player": pid, "day": self.step // 24, "hour": self.step % 24, "step": self.step,
            "farms": [fm.public() for fm in self.farms],
            "market": {"inventory": dict(self.inv), "prices": {p: price_of(p, self.inv[p]) for p in PRODUCTS}},
            "town": {"unlocked_shops": list(self.shops)},
            "private": {"shed": dict(f.shed), "seeds": dict(f.seeds),
                        "inventories": [dict(i) for i in f.inventories]},
        }

    # ---------------- unit actions ----------------
    def _unit_action(self, f, ui, act, day, planted_this_turn):
        if not act:
            return
        name = act[0]
        pos = f.farmer if ui == 0 else f.hands[ui - 1]
        inv = f.inventories[ui]
        x, y = pos
        if name in ("NORTH", "SOUTH", "EAST", "WEST"):
            dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}[name]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                pos[0], pos[1] = nx, ny
            return
        if name == "PICKUP":
            if not f.shed_adjacent(pos):
                return
            item = act[1]
            n = int(act[2]) if len(act) > 2 else 1
            n = min(n, f.shed.get(item, 0))
            if n > 0:
                f.shed[item] -= n
                inv[item] = inv.get(item, 0) + n
            return
        if name == "DROP":
            if not f.shed_adjacent(pos):
                return
            for k, v in list(inv.items()):
                room = self.shed_cap - sum(f.shed.values())
                f.shed[k] = f.shed.get(k, 0) + min(v, max(0, room))
            inv.clear()
            return
        if f.locked(x, y):
            return
        t = f.tiles[y][x]
        if name == "PLANT":
            crop = act[1] if len(act) > 1 else None
            if t is None and crop in CROP:
                planted_this_turn.setdefault(crop, []).append((ui, x, y))
            return
        if name == "PLACE":
            item = act[1] if len(act) > 1 else None
            if item in ANIMAL and isinstance(t, dict) and t.get("kind") == ANIMAL[item]["struct"] and not t.get("animal"):
                if inv.get(item, 0) > 0:
                    inv[item] -= 1
                    t.update(animal=item, placed_day=day, yield_units=0, fed_today=False,
                             consecutive_unfed=0, cared_today=False, fertilizer_available=False, pending_care_bonus=0)
            elif f.shed_adjacent(pos) and item in inv:
                n = min(int(act[2]) if len(act) > 2 else 1, inv[item])
                room = self.shed_cap - sum(f.shed.values())
                n = min(n, max(0, room))
                inv[item] -= n
                f.shed[item] = f.shed.get(item, 0) + n
            return
        if name == "BUILD_COOP" and t is None:
            f.tiles[y][x] = dict(kind="COOP", animal=None, placed_day=-1, yield_units=0, fed_today=False,
                                 consecutive_unfed=0, cared_today=False, fertilizer_available=False, pending_care_bonus=0)
            return
        if name == "BUILD_PASTURE" and t is None:
            f.tiles[y][x] = dict(kind="PASTURE", animal=None, placed_day=-1, yield_units=0, fed_today=False,
                                 consecutive_unfed=0, cared_today=False, fertilizer_available=False, pending_care_bonus=0)
            return
        if name == "DIG":
            if isinstance(t, dict) and (t.get("kind") in ("PLANT", "WEED") or (t.get("kind") in ("COOP", "PASTURE") and not t.get("animal"))):
                f.tiles[y][x] = None
            return
        if not isinstance(t, dict):
            return
        kind = t.get("kind")
        if kind == "PLANT":
            if name == "WATER":
                t["watered_today"] = True
            elif name == "HARVEST":
                yu = t["yield_units"]
                if yu > 0:
                    inv[t["crop"]] = inv.get(t["crop"], 0) + yu
                    t["yield_units"] = 0
                    if not CROP[t["crop"]]["ongoing"]:
                        f.tiles[y][x] = None
            elif name == "FERTILIZE":
                if inv.get("FERTILIZER", 0) > 0 and t["fertilized_until_day"] < day:
                    inv["FERTILIZER"] -= 1
                    t["fertilized_until_day"] = day + 2
        elif kind in ("COOP", "PASTURE") and t.get("animal"):
            if name == "FEED":
                if not t["fed_today"] and inv.get("WHEAT", 0) > 0:
                    inv["WHEAT"] -= 1
                    t["fed_today"] = True
            elif name == "CARE":
                t["cared_today"] = True
            elif name == "HARVEST":
                prod = ANIMAL[t["animal"]]["product"]
                if t["yield_units"] > 0:
                    inv[prod] = inv.get(prod, 0) + t["yield_units"]
                    t["yield_units"] = 0
            elif name == "COLLECT_FERTILIZER":
                if t["fertilizer_available"]:
                    t["fertilizer_available"] = False
                    inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1

    # ---------------- market ----------------
    def _market(self, orders, day):
        """orders: list per player of market order lists. Processed one unit at a time, alternating."""
        queues = [list(o[:10]) for o in orders]
        # non-unit orders (HIRE, LAND, animals, seeds) resolve immediately when reached in queue;
        # SELL/BUY_PRODUCT are interleaved unit by unit.
        cursors = [0, 0]
        remaining = [None, None]  # (type,item,n) currently being processed
        while True:
            progressed = False
            for pid in range(2):
                f = self.farms[pid]
                if remaining[pid] is None:
                    # advance to next order, resolving instant ones
                    while cursors[pid] < len(queues[pid]):
                        o = queues[pid][cursors[pid]]
                        cursors[pid] += 1
                        if not o:
                            continue
                        kind = o[0]
                        if kind in ("SELL", "BUY_PRODUCT"):
                            item = o[1]
                            n = int(o[2]) if len(o) > 2 else 1
                            if item in MARKET and n > 0:
                                remaining[pid] = [kind, item, n]
                                break
                        elif kind == "HIRE":
                            a, b = 1, 1
                            for _ in range(f.hires_today):
                                a, b = b, a + b
                            if f.money >= a:
                                f.money -= a
                                f.hires_today += 1
                                h = f.half
                                cands = [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]
                                occ = {c: 0 for c in cands}
                                for u in [f.farmer] + f.hands:
                                    if tuple(u) in occ:
                                        occ[tuple(u)] += 1
                                spot = min(cands, key=lambda c: occ[c])
                                f.hands.append([spot[0], spot[1]])
                                f.inventories.append({})
                        elif kind == "BUY_LAND":
                            k = len(f.unlocked)
                            if k < 4 and f.money >= LAND_COST[k - 1]:
                                q = o[1] if len(o) > 1 and o[1] in QUADS and o[1] not in f.unlocked else \
                                    [q for q in ("NE", "SW", "SE") if q not in f.unlocked][0]
                                f.money -= LAND_COST[k - 1]
                                f.unlocked.append(q)
                        elif kind == "BUY_ANIMAL":
                            a = o[1]
                            n = int(o[2]) if len(o) > 2 else 1
                            if a in ANIMAL:
                                for _ in range(n):
                                    if f.money >= ANIMAL[a]["cost"]:
                                        f.money -= ANIMAL[a]["cost"]
                                        f.shed[a] = f.shed.get(a, 0) + 1
                        elif kind == "BUY_SEED":
                            c = o[1]
                            n = int(o[2]) if len(o) > 2 else 1
                            if c in CROP:
                                for _ in range(n):
                                    if f.money >= CROP[c]["seed"]:
                                        f.money -= CROP[c]["seed"]
                                        f.seeds[c] = f.seeds.get(c, 0) + 1
                if remaining[pid] is None:
                    continue
                kind, item, n = remaining[pid]
                if kind == "SELL":
                    if f.shed.get(item, 0) <= 0:
                        remaining[pid] = None
                        continue
                    p = price_of(item, self.inv[item])
                    f.shed[item] -= 1
                    f.money += p
                    if p > 1:
                        self.inv[item] += 1
                else:  # BUY_PRODUCT
                    if item not in ("WHEAT", "FERTILIZER"):
                        remaining[pid] = None
                        continue
                    self.inv[item] -= 1
                    p = price_of(item, self.inv[item])
                    if f.money < p:
                        self.inv[item] += 1
                        remaining[pid] = None
                        continue
                    f.money -= p
                    room = self.shed_cap - sum(f.shed.values())
                    if room > 0:
                        f.shed[item] = f.shed.get(item, 0) + 1
                remaining[pid][2] -= 1
                if remaining[pid][2] <= 0:
                    remaining[pid] = None
                progressed = True
            if not progressed and all(r is None for r in remaining) and all(cursors[p] >= len(queues[p]) for p in range(2)):
                break

    # ---------------- town ----------------
    def _town(self):
        if self.step % 4 == 0:
            for shop in self.shops:
                items = SHOPS[shop]
                mult = 2 if len(items) == 1 else 1
                for it in items:
                    self.inv[it] = max(0, self.inv[it] - mult)
        if self.step % 24 == 0:
            for it in PRODUCTS:
                if it != "FERTILIZER":
                    self.inv[it] = max(0, self.inv[it] - 1)

    # ---------------- day refresh ----------------
    def _day_refresh(self, day):
        for f in self.farms:
            for y in range(self.n):
                for x in range(self.n):
                    t = f.tiles[y][x]
                    if f.locked(x, y):
                        continue
                    if t is None:
                        if self.rng.random() < 0.005:
                            f.tiles[y][x] = {"kind": "WEED"}
                        continue
                    if t.get("kind") == "PLANT":
                        c = CROP[t["crop"]]
                        age = day - t["planted_day"]
                        fert = t["fertilized_until_day"] >= day
                        if t["watered_today"]:
                            t["consecutive_unwatered"] = 0
                            if not c["ongoing"] and c["win"] <= age <= c["max_day"]:
                                cap = c["cap_fert"] if fert else c["cap"]
                                t["yield_units"] = min(cap, t["yield_units"] + (2 if fert else 1))
                        else:
                            t["consecutive_unwatered"] += 1
                        if c["ongoing"]:
                            k = age - c["first"]
                            if k >= 0 and k % c["interval"] == 0 and t.setdefault("_prod", 0) < c["cap"]:
                                t["yield_units"] += 2 if (fert and t["watered_today"]) else 1
                                t["_prod"] += 1
                                if t["_prod"] >= c["cap"]:
                                    t["max_lifespan_step"] = (day + 2) * 24
                        if t["consecutive_unwatered"] >= 2:
                            f.tiles[y][x] = {"kind": "WEED"}
                            continue
                        t["watered_today"] = False
                    elif t.get("kind") in ("COOP", "PASTURE") and t.get("animal"):
                        a = ANIMAL[t["animal"]]
                        if t["fed_today"]:
                            t["consecutive_unfed"] = 0
                            if t["cared_today"]:
                                t["pending_care_bonus"] += 1
                        else:
                            t["consecutive_unfed"] += 1
                        if t["consecutive_unfed"] >= 2:
                            f.tiles[y][x].update(animal=None, yield_units=0, pending_care_bonus=0)
                            continue
                        age1 = (day + 1) - t["placed_day"]
                        if age1 >= a["first"] and (age1 - a["first"]) % a["interval"] == 0:
                            add = 1 + (t["pending_care_bonus"] if t["fed_today"] else 0)
                            t["yield_units"] = min(a["max_held"], t["yield_units"] + add)
                            t["pending_care_bonus"] = 0
                        t["fertilizer_available"] = True
                        t["fed_today"] = False
                        t["cared_today"] = False
            # decay for one-time crops past lifespan is applied per-turn in _decay
            # hands vanish; inventories drop into the shed
            for inv in f.inventories:
                for k, v in inv.items():
                    room = self.shed_cap - sum(f.shed.values())
                    f.shed[k] = f.shed.get(k, 0) + min(v, max(0, room))
            f.inventories = [{}]
            f.hands = []
            f.hires_today = 0
            f.farmer = [f.half - 1, f.half - 1]
        nd = day + 1
        if nd % 3 == 0 and nd > 0 and len(self.shops) < 8:
            self.shops.append(self.rng.choice(list(SHOPS.keys())))

    def _decay(self):
        for f in self.farms:
            for y in range(self.n):
                for x in range(self.n):
                    t = f.tiles[y][x]
                    if isinstance(t, dict) and t.get("kind") == "PLANT" and t["max_lifespan_step"] >= 0 \
                            and self.step >= t["max_lifespan_step"] and (self.step - t["max_lifespan_step"]) % 2 == 0 \
                            and self.step > t["max_lifespan_step"]:
                        t["yield_units"] -= 1
                        if t["yield_units"] <= 0:
                            f.tiles[y][x] = {"kind": "WEED"}

    # ---------------- main loop ----------------
    def run(self, agents, on_day=None):
        while self.step < self.steps:
            day, hour = self.step // 24, self.step % 24
            actions = []
            for pid, ag in enumerate(agents):
                try:
                    actions.append(ag(self.obs(pid)) or {})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    actions.append({})
            orders = []
            for pid, act in enumerate(actions):
                f = self.farms[pid]
                planted = {}
                self._unit_action(f, 0, act.get("farmer") or ["PASS"], day, planted)
                hand_acts = act.get("hands") or []
                for hi in range(len(f.hands)):
                    a = hand_acts[hi] if hi < len(hand_acts) else ["PASS"]
                    self._unit_action(f, hi + 1, a, day, planted)
                for crop, lst in planted.items():
                    if f.seeds.get(crop, 0) >= len(lst):
                        for (ui, x, y) in lst:
                            f.seeds[crop] -= 1
                            c = CROP[crop]
                            f.tiles[y][x] = dict(kind="PLANT", crop=crop, planted_day=day, watered_today=False,
                                                 consecutive_unwatered=1, yield_units=1,
                                                 max_lifespan_step=(-1 if c["ongoing"] else (day + c["max_day"] + 1) * 24),
                                                 fertilized_until_day=-1)
                orders.append(act.get("market") or [])
            self._market(orders, day)
            self._town()
            self._decay()
            if hour == 23:
                self._day_refresh(day)
                if on_day:
                    on_day(self, day)
            self.step += 1
        return [f.money for f in self.farms]


# ---------------- baseline agent (README "wheat loop") ----------------
def starter(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]
    market = []
    if private["seeds"].get("WHEAT", 0) == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])
    w = private["shed"].get("WHEAT", 0)
    if w > 0:
        market.append(["SELL", "WHEAT", w])
    if tile is None and private["seeds"].get("WHEAT", 0) > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if tile["yield_units"] >= 4 or obs["day"] - tile["planted_day"] > 4:
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market}
    return {"farmer": ["PASS"], "hands": [], "market": market}
