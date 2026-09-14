"""Run the agent in the local simulator: python run_local.py [seed] [mirror]"""
import sys, importlib
import kagsim
import main as agent_mod


def fresh_agent():
    m = importlib.reload(agent_mod)
    return m.agent


def trace(game, day):
    f = game.farms
    line = f"day {day:2d} | P0 ${f[0].money:8.0f} P1 ${f[1].money:8.0f} | hands {f[0].hires_today:2d}"
    counts = {}
    for row in f[0].tiles:
        for t in row:
            if isinstance(t, dict):
                k = t.get("crop") or t.get("animal") or t.get("kind")
                counts[k] = counts.get(k, 0) + 1
    inv = game.inv
    line += f" | {counts} | melon={kagsim.price_of('MELON', inv['MELON'])} egg={kagsim.price_of('EGG', inv['EGG'])} fert={kagsim.price_of('FERTILIZER', inv['FERTILIZER'])} wheat={kagsim.price_of('WHEAT', inv['WHEAT'])} shops={len(game.shops)}"
    print(line)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mirror = len(sys.argv) > 2 and sys.argv[2] == "mirror"
    g = kagsim.Game(seed=seed)
    a0 = fresh_agent()
    opp = fresh_agent() if mirror else kagsim.starter
    res = g.run([a0, opp], on_day=trace)
    print("FINAL:", res, "WIN" if res[0] > res[1] else "LOSS")
