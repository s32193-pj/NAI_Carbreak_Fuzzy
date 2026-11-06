
import argparse
# from argparse_range import range_action
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import PIL


def parsing_arguments():
    parser = argparse.ArgumentParser(
        prog='Car breaking simulation',
        description='Uses fuzzy logic to simulate car breaking',
        suggest_on_error=True
    )

    parser.add_argument('name', help='name of the saved file')
    parser.add_argument('starting_speed',choices=range(0,301),metavar="[0,301]",type=float, help='define starting speed [km/h] ')
    parser.add_argument('distance',choices=range(0,81),metavar="[0,81]",type=float, help='define distance to obstacle [m]')
    parser.add_argument('condition',choices=range(0,11),metavar="[0,11]",type=float, help='define road conditions: 0=dry, 5=wet, 10=icy')
    args = parser.parse_args()

    return args




# ---- Scenario: change these and compare ----
V0 = 90       # starting speed [km/h]
D0 = 80       # distance to obstacle [m]
ROAD = 5      # 0 = dry, 5 = wet, 10 = icy

DT = 0.05        # simulation time step [s]
CAR_LEN = 4.0    # car length [m]

# Fuzzy system

# Universes
speed    = ctrl.Antecedent(np.arange(0, 151, 1), 'speed')
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
road     = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'road')
brake    = ctrl.Consequent(np.arange(0, 101, 1), 'brake')

# Membership functions
speed['low']    = fuzz.trimf(speed.universe, [0, 0, 60])
speed['medium'] = fuzz.trimf(speed.universe, [30, 70, 110])
speed['high']   = fuzz.trimf(speed.universe, [80, 150, 150])

distance['close']  = fuzz.trimf(distance.universe, [0, 0, 30])
distance['medium'] = fuzz.trimf(distance.universe, [15, 45, 75])
distance['far']    = fuzz.trimf(distance.universe, [55, 100, 100])

road['dry'] = fuzz.trimf(road.universe, [0, 0, 4])
road['wet'] = fuzz.trimf(road.universe, [2, 5, 8])
road['icy'] = fuzz.trimf(road.universe, [6, 10, 10])

brake['none']     = fuzz.trimf(brake.universe, [0, 0, 15])
brake['light']    = fuzz.trimf(brake.universe, [5, 25, 45])
brake['moderate'] = fuzz.trimf(brake.universe, [35, 55, 75])
brake['strong']   = fuzz.trimf(brake.universe, [65, 100, 100])

# Rules
rules = [
    # Base table: speed x distance (covers every combination)
    ctrl.Rule(speed['low'] & distance['far'], brake['none']),
    ctrl.Rule(speed['low'] & distance['medium'], brake['light']),
    ctrl.Rule(speed['low'] & distance['close'], brake['moderate']),
    ctrl.Rule(speed['medium'] & distance['far'], brake['light']),
    ctrl.Rule(speed['medium'] & distance['medium'], brake['moderate']),
    ctrl.Rule(speed['medium'] & distance['close'], brake['strong']),
    ctrl.Rule(speed['high'] & distance['far'], brake['moderate']),
    ctrl.Rule(speed['high'] & distance['medium'], brake['strong']),
    ctrl.Rule(speed['high'] & distance['close'], brake['strong']),
    # Road condition modifiers
    ctrl.Rule(road['icy'] & distance['medium'], brake['strong']),
    ctrl.Rule(road['icy'] & speed['high'], brake['strong']),
    ctrl.Rule(road['wet'] & speed['medium'] & distance['medium'], brake['strong']),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)


def braking(v, d, r):
    # Return brake force [%] for speed v [km/h], distance d [m], road r [0-10].""
    sim.input['speed'] = v
    sim.input['distance'] = d
    sim.input['road'] = r
    sim.compute()
    return sim.output['brake']


# Simulation

def simulate(v0_kmh, d0, r, max_steps=5000):
    # Simulate the car approaching an obstacle d0 metres ahead.
    max_decel = 9.0 - 0.75 * r   # available grip: ~9 m/s^2 dry, ~1.5 m/s^2 icy
    v = v0_kmh / 3.6             # km/h -> m/s
    x, t = 0.0, 0.0
    history = []                 # rows of (time, position, speed_kmh, brake_%)

    for _ in range(max_steps):
        dist = d0 - x
        b = braking(min(v * 3.6, 150), np.clip(dist, 0, 100), r)
        history.append((t, x, v * 3.6, b))
        if v <= 0 or dist <= 0:
            break
        v = max(v - (b / 100) * max_decel * DT, 0.0)
        x += v * DT
        t += DT

    return np.array(history)


# Animation

def main():
    args = parsing_arguments()
    t, x, v, b = simulate(args.starting_speed, args.distance, args.condition).T

    fig, (ax_road, ax_plot) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 2]})

    # Road scene
    ax_road.set_xlim(-CAR_LEN - 2, D0 + 6)
    ax_road.set_ylim(0, 4)
    ax_road.set_yticks([])
    ax_road.set_xlabel('position [m]')
    ax_road.axhspan(0.5, 3.5, color='0.85')                         # road
    ax_road.add_patch(Rectangle((D0, 0.5), 2, 3, color='black'))    # obstacle
    car = Rectangle((-CAR_LEN, 1.2), CAR_LEN, 1.6, color='tab:blue')
    ax_road.add_patch(car)
    info = ax_road.text(0.0, 1.08, '', transform=ax_road.transAxes, fontsize=10)

    # Live graph
    ax_plot.set_xlim(0, max(t[-1], DT))
    ax_plot.set_ylim(0, max(V0, 100) * 1.05)
    ax_plot.set_xlabel('time [s]')
    ax_plot.grid(alpha=0.3)
    speed_line, = ax_plot.plot([], [], label='speed [km/h]')
    brake_line, = ax_plot.plot([], [], label='brake force [%]', color='tab:red')
    ax_plot.legend(loc='upper right')

    def update(i):
        pos = min(x[i], D0)                  # don't draw the car through the wall
        car.set_x(pos - CAR_LEN)             # car's front bumper is at `pos`
        speed_line.set_data(t[:i + 1], v[:i + 1])
        brake_line.set_data(t[:i + 1], b[:i + 1])

        text = (f't = {t[i]:4.1f} s   speed = {v[i]:5.1f} km/h   '
                f'distance = {max(D0 - x[i], 0):5.1f} m   brake = {b[i]:4.0f}%')
        if i == len(t) - 1:
            crashed = D0 - x[i] <= 0
            text += '   CRASH!' if crashed else '   STOPPED'
            car.set_color('tab:red' if crashed else 'tab:green')
        info.set_text(text)
        return car, speed_line, brake_line, info

    anim = FuncAnimation(fig, update, frames=len(t),
                         interval=DT * 1000, repeat=False)

    anim.save(args.name, writer='pillow', fps=20)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()