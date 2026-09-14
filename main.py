"""
Kaggriculture agent — "Melon Rush -> Goose Empire"
====================================================

Architecture (sense -> plan -> tasks -> assign):

  1. perceive()   : turn the raw observation into a tidy State object.
  2. plan_market(): economic layer -> SELL / HIRE / BUY_LAND / BUY_ANIMAL / BUY_SEED orders.
  3. gen_tasks()  : every tile & animal emits prioritized work items
                    (WATER, FEED, HARVEST, PLANT, BUILD_COOP, PLACE, DIG, ...).
  4. assign()     : greedy scheduler -> each unit (farmer + hired hands) takes the
                    best "priority - walking distance" task. Claims are sticky across
                    turns so units don't thrash. Units fetch wheat / animals from the
                    shed when a task needs them, and drop produce when full.

Strategy in one paragraph:
  Melon prices collapse quadratically, so only the FIRST ~150 melons in the shared
  market are worth anything -> plant ~22 melons on day 0, dump them on day 10-11.
  Egg prices use a log curve (never crash) and every animal drops sellable
  fertilizer daily, so the melon money funds land + a large goose flock, plus a
  handful of cows/sheep for the premium milk/wool markets (small caps). Everything
  else grows wheat (feed + stable price). Hands are nearly free (fib cost), so we
  hire aggressively and let the scheduler keep them busy.

Pure Python, no dependencies. Entry point: agent(obs).
"""
import math

SEASON_DAYS = 30
TURNS_PER_DAY = 24

# ----------------------------------------------------------------------------
# Game knowledge tables (from the competition rules)
# ----------------------------------------------------------------------------
CROP = {
    #            seed  base  first max_day cap cap_fert ongoing
    "WHEAT":      (10,  25,   2,   4,      4,  6,       False),
    "CARROT":     (20,  35,   2,   3,      3,  4,       False),
    "MELON":      (80,  250,  10,  10,     6,  6,       False),
    "TOMATO":     (50,  60,   8,   11,     4,  4,       True),
    "STRAWBERRY": (100, 120,  10,  16,     4,  4,       True),
}
ANIMAL = {
    #         cost structure   product  first interval
    "GOOSE": (300, "COOP",    "EGG",   4,    1),
    "COW":   (400, "PASTURE", "MILK",  8,    2),
    "SHEEP": (500, "PASTURE", "WOOL",  6,    3),
}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMAL_NAMES = tuple(ANIMAL.keys())

# ----------------------------------------------------------------------------
# Strategy knobs (tune here)
# ----------------------------------------------------------------------------
P = dict(
    early_geese=3,            # geese bought on day 0 (fertilizer income from day 1)
    melon_tiles=22,           # size of the day-0 melon wave
    melon_min_price=170,      # only plant melons while the market is fresh
    geese_max=24,             # flock size once money arrives
    goose_last_buy_day=23,    # a goose needs ~3 productive days to pay back
    cows=3, cow_last_buy_day=15,
    sheep=3, sheep_last_buy_day=17,
    max_hands=14,
    actions_per_unit=13,      # effective actions/day after walking
    fert_keep_price=20,       # below this price, keep fertilizer for our crops
    fert_keep=30,
    carrot_min_price=30,
    land_reserve=800,
    land_last_day=20,
    drop_threshold=12,
    animal_radius=3,          # animals need 4 visits/day -> closest tiles
    crop_radius=5,            # corners are not worth the walk
    walk_penalty=4.0,
    same_tile_bonus=40,        # produce carried before walking back to the shed
    land_order=("NE", "SW", "SE"),
    land_action_with_arg=True,   # ["BUY_LAND","NE"]; set False if env wants ["BUY_LAND"]
)

_MEM = {}   # per-player persistent memory across turns


def _fib_hire_cost(n):
    """Cost of the n-th hire of the day (n = hires already made). fib: 1,1,2,3,5,8..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hire_cost_for(k):
    return sum(_fib_hire_cost(i) for i in range(k))


# ----------------------------------------------------------------------------
# 1. Perception
# ----------------------------------------------------------------------------
class State:
    pass


def _g(d, k, default=None):
    try:
        v = d.get(k, default) if hasattr(d, "get") else getattr(d, k, default)
    except Exception:
        v = default
    return default if v is None else v


def perceive(obs):
    s = State()
    s.player = int(_g(obs, "player", 0))
    s.day = int(_g(obs, "day", 0))
    s.hour = int(_g(obs, "hour", 0))
    s.step = s.day * TURNS_PER_DAY + s.hour
    farms = _g(obs, "farms", [])
    me = farms[s.player] if s.player < len(farms) else {}
    s.money = float(_g(me, "money", 0))
    s.tiles = _g(me, "tiles", [])
    s.n = len(s.tiles) if s.tiles else 10
    s.half = s.n // 2
    s.unlocked = list(_g(me, "unlocked_quadrants", ["NW"]))
    s.hires_today = int(_g(me, "hires_today", 0))
    priv = _g(obs, "private", {})
    s.shed = dict(_g(priv, "shed", {}))
    s.seeds = dict(_g(priv, "seeds", {}))
    invs = list(_g(priv, "inventories", []))
    market = _g(obs, "market", {})
    s.prices = dict(_g(market, "prices", {}))
    s.market_inv = dict(_g(market, "inventory", {}))
    town = _g(obs, "town", {})
    s.shops = list(_g(town, "unlocked_shops", []))

    farmer = list(_g(me, "farmer", [s.half - 1, s.half - 1]))
    hands = [list(h) for h in _g(me, "hands", [])]
    s.units = []
    positions = [farmer] + hands
    for i, pos in enumerate(positions):
        inv = dict(invs[i]) if i < len(invs) and invs[i] else {}
        s.units.append(dict(idx=i, pos=(int(pos[0]), int(pos[1])), inv=inv))
    s.n_hands = len(hands)

    # shed-adjacent tiles (shed is the center, never a tile)
    h = s.half
    s.shed_tiles = [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]

    # scan tiles
    s.plants, s.structs, s.weeds, s.empty = [], [], [], []
    for y, row in enumerate(s.tiles):
        for x, t in enumerate(row):
            if t is None:
                s.empty.append((x, y))
            elif isinstance(t, str):
                continue  # "LOCKED"
            else:
                kind = _g(t, "kind", "")
                if kind == "PLANT":
                    s.plants.append((x, y, t))
                elif kind == "WEED":
                    s.weeds.append((x, y))
                elif kind in ("COOP", "PASTURE"):
                    s.structs.append((x, y, t))
    s.animals_placed = [(x, y, t) for (x, y, t) in s.structs if _g(t, "animal")]
    s.n_animals = len(s.animals_placed)
    s.days_left = SEASON_DAYS - 1 - s.day   # full days after today
    s.last_day = s.day >= SEASON_DAYS - 1
    return s


def price(s, item, default=1):
    return float(s.prices.get(item, default))


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shed_dist(s, pos):
    return min(dist(pos, t) for t in s.shed_tiles)


def nearest_shed_tile(s, pos):
    return min(s.shed_tiles, key=lambda t: dist(pos, t))


def carried(s, item):
    return sum(int(u["inv"].get(item, 0)) for u in s.units)


# ----------------------------------------------------------------------------
# 2. Economic planner -> market orders
# ----------------------------------------------------------------------------
def desired_hands(s):
    """Estimate today's workload and convert it into a number of hands."""
    if s.last_day:
        work = s.n_animals * 2 + len(s.plants) * 0.6 + 6
    else:
        work = len(s.plants) * 1.7 + s.n_animals * 4.8 + len(s.empty) * 2.6 + 8
        if s.n_animals == 0 and len(s.plants) < 5:
            work += 40  # day 0 rush: plant + water everything before nightfall
    units_needed = math.ceil(work / P["actions_per_unit"])
    hands = max(2, min(P["max_hands"], units_needed - 1))
    # affordability: never spend more than 25% of cash on hands
    while hands > 1 and _hire_cost_for(hands) > s.money * 0.25:
        hands -= 1
    return hands


def fert_reserve(s):
    """Fertilizer to hold back for our melons (fertilize at age 6)."""
    n = 0
    for (_, _, t) in s.plants:
        if _g(t, "crop") == "MELON" and int(_g(t, "fertilized_until_day", -1)) < s.day:
            age = s.day - int(_g(t, "planted_day", s.day))
            if 3 <= age <= 7:
                n += 1
    return n


def wheat_reserve(s):
    if s.last_day:
        return 0
    return s.n_animals * 2 + 2


def plan_market(s, mem):
    orders = []
    money = s.money
    day = s.day

    # --- HIRE (hours 0-1, orders are limited to 10/turn) ---
    if s.hour <= 1:
        want = desired_hands(s)
        have = s.n_hands
        while have < want and len(orders) < 7:
            c = _fib_hire_cost(s.hires_today + (have - s.n_hands))
            if money - c < 20:
                break
            orders.append(["HIRE"])
            money -= c
            have += 1

    # --- SELL everything sellable in the shed ---
    reserve_w = wheat_reserve(s)
    for item, qty in sorted(s.shed.items(), key=lambda kv: -price(s, kv[0]) * kv[1]):
        qty = int(qty)
        if qty <= 0 or item in ANIMAL_NAMES or item not in s.prices:
            continue
        p = price(s, item)
        if item == "WHEAT":
            qty = qty - reserve_w
        elif item == "FERTILIZER" and not s.last_day:
            keep = fert_reserve(s)
            if p < P["fert_keep_price"]:
                keep = max(keep, P["fert_keep"])
            qty = qty - keep
        if qty <= 0:
            continue
        if p < 2 and not s.last_day and item != "WHEAT":
            continue  # floor price: holding costs nothing, opponent may sell first
        orders.append(["SELL", item, qty])
        money += p * qty * 0.9  # rough estimate, prices move as we sell

    # --- BUY WHEAT for feed ---
    if not s.last_day and s.n_animals > 0:
        have_w = int(s.shed.get("WHEAT", 0)) + carried(s, "WHEAT")
        need = s.n_animals * 2 - have_w
        if s.hour >= 18:
            need = s.n_animals - int(s.shed.get("WHEAT", 0))  # tomorrow's breakfast must sit in shed
        wp = price(s, "WHEAT", 25)
        if need > 0 and money > need * wp + 50:
            orders.append(["BUY_PRODUCT", "WHEAT", int(need)])
            money -= need * wp

    # --- Reserves for the rest of the plan ---
    feed_reserve = (s.n_animals + 3) * price(s, "WHEAT", 25) * 1.5 + 60

    # --- BUY LAND ---
    if day >= 2 and day <= P["land_last_day"] and s.hour <= 20:
        locked = [q for q in P["land_order"] if q not in s.unlocked]
        free = sum(1 for p in s.empty if shed_dist(s, p) <= P["crop_radius"])
        if locked and free < 6:
            cost = {1: 1000, 2: 2000, 3: 4000}.get(len(s.unlocked), 8000)
            if money - cost >= P["land_reserve"] + feed_reserve:
                q = locked[0]
                orders.append(["BUY_LAND", q] if P["land_action_with_arg"] else ["BUY_LAND"])
                money -= cost
                mem["pending_land"] = q

    # --- BUY ANIMALS ---
    buy_plan = {}
    if not s.last_day and s.hour <= 20:
        free_tiles = len(s.empty) + len(s.weeds) // 2
        empty_structs = {"COOP": 0, "PASTURE": 0}
        for (_, _, t) in s.structs:
            if not _g(t, "animal"):
                empty_structs[_g(t, "kind")] += 1
        placed = {a: 0 for a in ANIMAL}
        for (_, _, t) in s.animals_placed:
            a = _g(t, "animal")
            if a in placed:
                placed[a] += 1
        unplaced = {a: int(s.shed.get(a, 0)) + carried(s, a) for a in ANIMAL}
        # geese target: small until the melon money lands
        rich = money > 4000 or day >= 10
        targets = {
            "GOOSE": (P["geese_max"] if rich else P["early_geese"]) if day <= P["goose_last_buy_day"] else 0,
            "COW": P["cows"] if (rich and day <= P["cow_last_buy_day"]) else 0,
            "SHEEP": P["sheep"] if (rich and day <= P["sheep_last_buy_day"]) else 0,
        }
        for a in ("COW", "SHEEP", "GOOSE"):   # deadline-first ordering
            cost, struct, *_ = ANIMAL[a]
            want = targets[a] - placed[a] - unplaced[a]
            n = 0
            while want > 0 and n < 4:
                room = free_tiles + empty_structs[struct]
                if room <= 0:
                    break
                if money - cost < feed_reserve:
                    break
                money -= cost
                n += 1
                want -= 1
                if empty_structs[struct] > 0:
                    empty_structs[struct] -= 1
                else:
                    free_tiles -= 1
            if n > 0:
                orders.append(["BUY_ANIMAL", a, n])
                buy_plan[a] = n

    s.buy_plan = buy_plan
    s.money_after_orders = money
    return orders


# ----------------------------------------------------------------------------
# 3. Task generation
# ----------------------------------------------------------------------------
def choose_crop(s, melon_count):
    left = s.days_left
    if (left >= 11 and melon_count < P["melon_tiles"] and s.day <= 1
            and price(s, "MELON", 250) >= P["melon_min_price"]):
        return "MELON"
    wheat_tiles = sum(1 for (_, _, t) in s.plants if _g(t, "crop") == "WHEAT")
    if left >= 5 and wheat_tiles < s.n_animals + 4:
        return "WHEAT"   # feed security first: a wheat tile yields ~1 wheat/day
    if left >= 4 and price(s, "CARROT", 35) >= P["carrot_min_price"]:
        return "CARROT"
    if left >= 5:
        return "WHEAT"
    if left >= 4:
        return "CARROT"
    return None


def crop_ready(s, t):
    crop = _g(t, "crop", "WHEAT")
    if crop not in CROP:
        return int(_g(t, "yield_units", 0)) > 0
    seed, base, first, max_day, cap, cap_fert, ongoing = CROP[crop]
    age = s.day - int(_g(t, "planted_day", s.day))
    yu = int(_g(t, "yield_units", 0))
    if ongoing:
        return yu > 0
    fert = int(_g(t, "fertilized_until_day", -1)) >= 0
    eff_cap = cap_fert if fert else cap
    mls = int(_g(t, "max_lifespan_step", -1))
    if yu >= eff_cap:
        return True
    if age > max_day:
        return True   # overdue: decay starts now
    if age >= max_day and _g(t, "watered_today", False):
        return True   # last bonus watering done -> take it today, never race the decay timer
    if mls >= 0 and s.step >= mls - 1:
        return True
    if s.last_day and s.hour >= 12 and age >= first and yu > 0:
        return True   # cash out before the season ends
    return False


def gen_tasks(s, mem):
    """Return (tasks, seed_orders). Each task: dict(key, pos, prio, action, need)."""
    tasks = []
    seed_orders = []
    hour = s.hour
    urgency = max(0, hour - 10) * 4     # survival tasks get more urgent as the day goes on
    endgame = s.last_day

    # ---- plants ----
    melon_count = 0
    for (x, y, t) in s.plants:
        crop = _g(t, "crop", "")
        if crop == "MELON":
            melon_count += 1
        key = ("plant", x, y)
        if crop_ready(s, t):
            age_now = s.day - int(_g(t, "planted_day", s.day))
            overdue = age_now > CROP.get(crop, (0, 0, 0, 99))[3]
            prio = (110 if overdue else 96) if crop == "MELON" else (105 if overdue else 90)
            tasks.append(dict(key=key + ("harvest",), pos=(x, y), prio=prio, action=["HARVEST"], need=None))
            continue
        if endgame:
            continue
        if not _g(t, "watered_today", False):
            prio = 100 + urgency
            tasks.append(dict(key=key + ("water",), pos=(x, y), prio=prio, action=["WATER"], need=None))
        age = s.day - int(_g(t, "planted_day", s.day))
        unfert = int(_g(t, "fertilized_until_day", -1)) < s.day
        # melons: fertilize at age 6 -> yield caps at age 8, sold 2 days earlier, no decay race
        if crop == "MELON" and unfert and 6 <= age <= 7:
            tasks.append(dict(key=key + ("fert",), pos=(x, y), prio=88, action=["FERTILIZE"], need=("FERTILIZER", 1)))
        # cheap fertilizer -> boost wheat/carrot too
        elif (crop in ("WHEAT", "CARROT") and unfert and age == 2
                and price(s, "FERTILIZER", 100) < P["fert_keep_price"]):
            tasks.append(dict(key=key + ("fert",), pos=(x, y), prio=55, action=["FERTILIZE"], need=("FERTILIZER", 1)))

    # ---- animals ----
    for (x, y, t) in s.structs:
        a = _g(t, "animal")
        key = ("anim", x, y)
        if not a:
            continue
        if int(_g(t, "yield_units", 0)) > 0:
            tasks.append(dict(key=key + ("harvest",), pos=(x, y), prio=82, action=["HARVEST"], need=None))
        if _g(t, "fertilizer_available", False):
            tasks.append(dict(key=key + ("fert",), pos=(x, y), prio=72, action=["COLLECT_FERTILIZER"], need=None))
        if endgame:
            continue
        if not _g(t, "fed_today", False):
            tasks.append(dict(key=key + ("feed",), pos=(x, y), prio=101 + urgency, action=["FEED"], need=("WHEAT", 1)))
        elif not _g(t, "cared_today", False):
            tasks.append(dict(key=key + ("care",), pos=(x, y), prio=60, action=["CARE"], need=None))

    if endgame or hour >= 22:
        return tasks, seed_orders

    # ---- structures to build & animals to place ----
    empty_structs = {"COOP": [], "PASTURE": []}
    for (x, y, t) in s.structs:
        if not _g(t, "animal"):
            empty_structs[_g(t, "kind")].append((x, y))
    unplaced = {a: int(s.shed.get(a, 0)) + carried(s, a) for a in ANIMAL}   # only build for animals we own
    need_build = {"COOP": 0, "PASTURE": 0}
    for a, n in unplaced.items():
        need_build[ANIMAL[a][1]] += n
    for st in need_build:
        need_build[st] = max(0, need_build[st] - len(empty_structs[st]))
    # PLACE tasks (animal must be carried; fetch logic handles pickup)
    for a in ANIMAL:
        st = ANIMAL[a][1]
        avail = int(s.shed.get(a, 0)) + carried(s, a)
        for (x, y) in empty_structs[st][:avail]:
            tasks.append(dict(key=("place", x, y), pos=(x, y), prio=130, action=["PLACE", a], need=(a, 1)))
        empty_structs[st] = empty_structs[st][avail:]

    # empty tiles, nearest to shed first (animals need 4 visits/day -> keep them close)
    # keep the farm compact: every extra step is a wasted action for every visit
    radius = P["crop_radius"] + 1 if s.day <= 1 else P["crop_radius"]   # day-0 melon wave uses the whole quadrant
    empties = sorted((p for p in s.empty if shed_dist(s, p) <= radius), key=lambda p: shed_dist(s, p))
    weeds = sorted((p for p in s.weeds if shed_dist(s, p) <= P["crop_radius"]), key=lambda p: shed_dist(s, p))
    for st, n in need_build.items():
        while n > 0 and empties and shed_dist(s, empties[0]) <= P["animal_radius"]:
            x, y = empties.pop(0)
            tasks.append(dict(key=("build", x, y), pos=(x, y), prio=118, action=["BUILD_" + st], need=None))
            n -= 1
    # ---- planting ----
    if hour <= 20:
        want = {}
        plant_tiles = []
        for (x, y) in empties:
            crop = choose_crop(s, melon_count + want.get("MELON", 0))
            if crop is None:
                break
            want[crop] = want.get(crop, 0) + 1
            plant_tiles.append((x, y, crop))
        # tasks limited by seeds in stock; order seeds for the shortfall
        stock = {c: int(s.seeds.get(c, 0)) for c in want}
        budget = s.money_after_orders - 60
        for crop, n in want.items():
            short = n - stock[crop]
            if short > 0:
                cost = CROP[crop][0]
                buy = min(short, int(budget // cost) if cost > 0 else 0, 30)
                if buy > 0:
                    seed_orders.append(["BUY_SEED", crop, buy])
                    budget -= buy * cost
        for (x, y, crop) in plant_tiles:
            if stock.get(crop, 0) <= 0:
                continue
            stock[crop] -= 1
            tasks.append(dict(key=("plant", x, y), pos=(x, y), prio=86, action=["PLANT", crop], need=None))
    # ---- weeds: clear when we are short on land ----
    if len(empties) < 8:
        for (x, y) in weeds[:6]:
            tasks.append(dict(key=("dig", x, y), pos=(x, y), prio=45, action=["DIG"], need=None))
    return tasks, seed_orders


# ----------------------------------------------------------------------------
# 4. Scheduler
# ----------------------------------------------------------------------------
def move_toward(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def produce_count(inv):
    return sum(int(v) for k, v in inv.items() if k in PRODUCTS and k != "WHEAT")


def assign(s, tasks, mem):
    hour = s.hour
    prev = mem.get("claims", {})
    new_claims = {}
    taken = set()
    actions = {}
    shed_avail = {k: int(v) for k, v in s.shed.items()}
    turns_left = TURNS_PER_DAY - 1 - hour
    by_key = {t["key"]: t for t in tasks}

    unfed = sum(1 for t in tasks if t["action"][0] == "FEED")
    n_units = max(1, len(s.units), desired_hands(s) + 1)

    def feasible(u, t):
        """Return (distance, fetch_item) or None if this unit can't do the task."""
        pos, inv = u["pos"], u["inv"]
        d = dist(pos, t["pos"])
        fetch = None
        if t["need"] is not None:
            item, qty = t["need"]
            if int(inv.get(item, 0)) < qty:
                if shed_avail.get(item, 0) < qty:
                    return None
                fetch = item
                sh = nearest_shed_tile(s, pos)
                d = dist(pos, sh) + dist(sh, t["pos"])
        if d > turns_left - 1 and not (s.last_day and t["action"][0] == "HARVEST"):
            return None
        return d, fetch

    def drop_task(u):
        return dict(key=("drop", u["idx"]), pos=nearest_shed_tile(s, u["pos"]), prio=999, action=["DROP"], need=None)

    total_carried = sum(produce_count(u["inv"]) for u in s.units)
    shed_items = sum(int(v) for k, v in s.shed.items())
    cap_risk = total_carried + shed_items > 80   # shed holds 100; overflow at night is lost

    def needs_drop(u):
        prod = produce_count(u["inv"])
        sd = shed_dist(s, u["pos"])
        if prod <= 0:
            return False
        if s.last_day:
            return turns_left <= sd + 2
        if cap_risk and turns_left <= sd + 3:
            return True
        return prod >= P["drop_threshold"]

    chosen = {}
    # ---- pass 1: units keep the task they claimed last turn (kills thrashing) ----
    for u in s.units:
        k = prev.get(u["idx"])
        t = by_key.get(k)
        if t is None or k in taken or needs_drop(u):
            continue
        f = feasible(u, t)
        if f is None:
            continue
        chosen[u["idx"]] = (t, f[1])
        taken.add(k)

    # ---- pass 2: greedy for the rest ----
    for u in s.units:
        idx = u["idx"]
        if idx in chosen:
            continue
        here = by_key.get(("plant", u["pos"][0], u["pos"][1], "water"))
        if here is not None and here["key"] not in taken:
            chosen[idx] = (here, None)      # never leave a fresh seed unwatered
            taken.add(here["key"])
            continue
        if needs_drop(u):
            chosen[idx] = (drop_task(u), None)
            continue
        best, best_score = None, -1e9
        for t in tasks:
            if t["key"] in taken:
                continue
            f = feasible(u, t)
            if f is None:
                continue
            d, fetch = f
            score = t["prio"] - P["walk_penalty"] * d
            if d == 0:
                score += P["same_tile_bonus"]   # finish everything here before walking away
            if t["need"] is not None and fetch is None:
                score += 30   # already carrying what this task needs -> finish the job
            if score > best_score:
                best_score, best = score, (t, fetch)
        if best is None:
            actions[idx] = ["PASS"]
            continue
        chosen[idx] = best
        taken.add(best[0]["key"])

    # ---- turn choices into actions ----
    for u in s.units:
        idx, pos, inv = u["idx"], u["pos"], u["inv"]
        if idx not in chosen:
            actions.setdefault(idx, ["PASS"])
            continue
        t, fetch = chosen[idx]
        new_claims[idx] = t["key"]
        if fetch:
            target = nearest_shed_tile(s, pos)
            if pos == target:
                if fetch == "WHEAT":
                    qty = min(shed_avail.get("WHEAT", 0), max(1, math.ceil(unfed / n_units) + 1))
                else:
                    qty = 1
                shed_avail[fetch] = shed_avail.get(fetch, 0) - qty
                actions[idx] = ["PICKUP", fetch, int(qty)]
            else:
                actions[idx] = move_toward(pos, target)
            continue
        if pos == t["pos"]:
            actions[idx] = list(t["action"])
            if t["action"][0] == "DROP":
                for k, v in inv.items():
                    shed_avail[k] = shed_avail.get(k, 0) + int(v)
                new_claims.pop(idx, None)
        else:
            actions[idx] = move_toward(pos, t["pos"])

    mem["claims"] = new_claims
    return actions


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def _memory(s):
    m = _MEM.get(s.player)
    if m is None or s.step == 0 or m.get("last_step", -1) > s.step:
        m = {"claims": {}, "last_step": -1}
        _MEM[s.player] = m
    m["last_step"] = s.step
    return m


def agent(obs, config=None):
    try:
        s = perceive(obs)
        mem = _memory(s)
        market = plan_market(s, mem)
        tasks, seed_orders = gen_tasks(s, mem)
        market = (market + seed_orders)[:10]
        acts = assign(s, tasks, mem)
        farmer = acts.get(0, ["PASS"])
        hands = [acts.get(i, ["PASS"]) for i in range(1, len(s.units))]
        return {"farmer": farmer, "hands": hands, "market": market}
    except Exception as e:  # never crash: a crashed agent loses the episode
        import traceback
        traceback.print_exc()
        return {"farmer": ["PASS"], "hands": [], "market": []}
