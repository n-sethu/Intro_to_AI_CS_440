# engine to run the experiments
import random
from ship import ShipGrid
from fire import FireSystem

def run_trial(ship, bot_class, start, button, fire_start, q, seed=None, max_steps=100000, trace=None):
    # trial returns trial success and reason 
    # reasons: 'button', 'walked_into_fire', 'fire_reached_bot', 'timeout'
    # fixed seed ensures the fire's same randomness for each bot
    # trace: optional list; gets the bot position at t=0 and after every move
    rng_state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try: 
        #get fire and bot
        fire = FireSystem(ship, q, fire_start)          
        bot = bot_class(ship, start, button, fire_start)
        if trace is not None:
            trace.append(bot.position)
        for _ in range(max_steps):
            # bot makes a move decision
            bot.choose_move(fire)
            if trace is not None:
                trace.append(bot.position)
            # if bot is in fire cell, end of trial          
            if bot.position in fire.burning_cells:
                return False, 'walked_into_fire', bot
            # if bot in in button cell, yay! end of trial
            if bot.position == button:             
                return True, 'button', bot
            # else, fire spreads
            fire.step()                
            # if fire spread to bot cell, end of trial                
            if bot.position in fire.burning_cells:
                return False, 'fire_reached_bot', bot
        return False, 'timeout', bot
    finally:
        random.setstate(rng_state)

def compare_bots(bot_classes, D, q, n_trials):
    # run every bot on the same trial and return the winning bot
    wins = {cls.__name__: 0 for cls in bot_classes}
    for i in range(n_trials):
        ship = ShipGrid(D)
        start, button, fire_start = random.sample(ship.get_open_cells(), 3)
        seed = random.randrange(2**32)
        for cls in bot_classes:
            ok, how, bot= run_trial(ship, cls, start, button, fire_start, q, seed)
            wins[cls.__name__] += ok
            if cls.__name__ == "Bot4":
                if bot.active_threshold is not None:
                    strategy = f"active_threshold={bot.active_threshold}"
                elif bot.path is not None:
                    strategy = "shortest-path fallback"
                else:
                    strategy = "no available path"
                print(i, ": ", ok, how, cls.__name__, strategy)
            else:
                print(i, ": ", ok, how, cls.__name__)

    return wins

if __name__ == "__main__":
    from bots import Bot1, Bot2, Bot3, Bot4

    n = 100
    for q in (0.3, 0.5,0.7):
        # Compare all four bot strategies under the same trial seeds.
        wins = compare_bots([Bot1, Bot2, Bot3, Bot4], D=50, q=q, n_trials=n)
        print(f"q={q}: " + ", ".join(f"{k}={v}/{n}" for k, v in wins.items()))
