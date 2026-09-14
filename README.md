# Kaggriculture agent — Melon Rush → Goose Empire 🍈🪿

# 🍈 Kaggriculture AI Agent — Market-Aware Farming Bot (Kaggle Simulation Competition)

Autonomous agent competing live on Kaggle's leaderboard. Models the game as a supply-chain
optimization problem: reads the market's price curves to time sales, allocates ~15 workers per
turn with a greedy priority scheduler, and reinvests capital (land, livestock) on a payback schedule.

**Highlights:** pure-Python planner (no ML libs) · custom local simulator of the game rules for
fast iteration · ~7× the baseline agent in local tests · validated and playing on Kaggle.

## Strategy
1. **Melon rush** — melon price collapses *quadratically*, so only the first ~150 melons market-wide are worth anything. Plant 22 on day 0, fertilize them with our own goose fertilizer at age 6 (caps yield 2 days early), sell on day 9–10 before the opponent.
2. **Goose empire** — eggs use a *log* price curve (never crash), every animal drops 1 sellable fertilizer/day, CARE doubles egg output. Melon money → land + ~24 geese + 3 cows + 3 sheep (small premium markets).
3. **Wheat fields** — feed security; buying wheat pushes its price up against you.
4. **Cheap labor** — hands cost fib(n) coins/day, so hire up to 14 and let the scheduler keep them busy.

## Architecture (`main.py`, pure Python, no deps)
`perceive()` → `plan_market()` (sell/hire/buy land/animals/seeds) → `gen_tasks()` (every tile & animal emits prioritized work)
→ `assign()` (greedy scheduler: `priority − walk_penalty·distance`, sticky claims to stop thrashing, "finish this tile first",
survival tasks escalate through the day, fetch wheat/animals from the shed when a task needs them, drop before the 100-item shed cap).

## Local testing
`kagsim.py` is a from-the-README re-implementation of the rules (market curve, weeds, animal escape, fib hiring, town shops)
for fast offline iteration. `python run_local.py [seed] [mirror]` prints a daily trace. Then validate with the real package:

```bash
pip install -U kaggle-environments && python probe_env.py
```

## Submit
```bash
kaggle competitions submit kaggriculture -f main.py -m "Melon rush + goose empire v1"
```
Results in the local sim: ~$24–28k vs ~$3.8k for the wheat-loop baseline.
