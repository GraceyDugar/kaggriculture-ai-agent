"""Run this once with the REAL package installed to verify things I couldn't check offline:
   pip install -U kaggle-environments && python probe_env.py
"""
import inspect, re, kaggle_environments as ke
from kaggle_environments import make
import kagsim
src = inspect.getsource(ke.envs.kaggriculture.kaggriculture)
print("BUY_LAND handling:\n", "\n".join(l for l in src.splitlines() if "BUY_LAND" in l)[:800])
print("\nProduct keys:", sorted(set(re.findall(r'"(EGG|EGGS|MILK|WOOL|FERTILIZER)"', src))))
env = make("kaggriculture", debug=True)
env.run(["main.py", "starter"])
print("vs starter:", [(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
env = make("kaggriculture", debug=True)
env.run(["main.py", "main.py"])
print("mirror:", [(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
