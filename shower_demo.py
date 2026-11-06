"""
Shower Delay Demo
=================
A small Arcade (3.3.x) game for a 5-minute talk on feedback delay.
The user turns a shower knob (LEFT / RIGHT keys) to reach a target
temperature. The water responds after a transport delay T and a
first-order lag tau, so reacting too fast makes the water swing
between too hot and too cold.

Model
-----
    x_{k+1} = alpha * x_k - (1 - alpha) * K * a_{k-d}
    y_k     = T_mid + x_k
    T_mid = (l + h) / 2,  K = (h - l) / pi,  alpha = exp(-DT / tau),  d = round(T / DT)
    Negative knob angle = hotter. Requested temperature = T_mid - K * a.

Run the game:        python shower_demo.py
Run the model test:  python shower_demo.py --test
"""

import math
import random
import sys
from collections import deque
from dataclasses import dataclass

# =====================================================================
# 1. ADJUSTABLE PARAMETERS
# =====================================================================

# --- Window ----------------------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Shower Delay Demo"
ROW_RATIO = (2, 0, 1)  # graph+timer : bar : knob (top to bottom)
GRAPH_WIDTH_FRACTION = 0.75  # share of row 1 used by the graph

# --- Temperature range (random each round) ---------------------------
L_RANGE = (15, 25)  # lowest temperature l, degC
H_RANGE = (38, 48)  # highest temperature h, degC
SPAN_MIN = 20  # minimum h - l, degC
SPAN_MAX = 40  # maximum h - l, degC

# --- Target ----------------------------------------------------------
TARGET_MARGIN = 4.0  # target stays this far from l and h, degC
TARGET_MIN_GAP = 8.0  # target is at least this far from start temp, degC
GOOD_BAND = 0.5  # "comfortable" band is target +/- this, degC
SUCCESS_HOLD = 3.0  # stay in band this long (continuously) to win, s

# --- Dynamics --------------------------------------------------------
DT = 0.01  # model time step, s
DELAY_RANGE = (0.0, 10.0)  # slider range for delay T, s
LAG_RANGE = (0.1, 10.0)  # slider range for lag tau, s
SLIDER_STEP = 0.1  # slider snapping step, s
DEFAULT_DELAY = 2.0  # first-launch delay, s
DEFAULT_LAG = 0.5  # first-launch lag, s

# --- Knob ------------------------------------------------------------
A_MIN = -math.pi / 2  # knob limit (fully hot), rad
A_MAX = math.pi / 2  # knob limit (fully cold), rad
KNOB_STEP = math.pi / 36  # one key step (5 deg), rad
KEY_REPEAT_RATE = 10  # steps per second while a key is held

# --- Round -----------------------------------------------------------
TIMEOUT = 120.0  # round time limit, s
MAX_FRAME_TIME = 0.25  # cap per frame so a frozen window can't cause a burst

# --- Colours ---------------------------------------------------------
BG_COLOR = (24, 28, 36)
PANEL_COLOR = (34, 40, 52)
AXIS_COLOR = (170, 178, 190)
GRID_COLOR = (60, 68, 82)
TEXT_COLOR = (230, 234, 240)
MUTED_TEXT = (150, 158, 170)
COLD_COLOR = (60, 130, 255)
HOT_COLOR = (235, 70, 55)
TEMP_LINE_COLOR = (255, 200, 80)
TARGET_COLOR = (90, 220, 140)
BAND_COLOR = (90, 220, 140, 50)
NOW_COLOR = (255, 255, 255)


# =====================================================================
# 2. MODEL (no Arcade inside)
# =====================================================================


class ShowerEnv:
    """Shower with transport delay + first-order lag."""

    def __init__(self, delay, lag, rng=None):
        if lag <= 0:
            raise ValueError("lag must be > 0")
        self.delay = delay
        self.lag = lag
        self.d = int(round(delay / DT))  # delay in steps (round, not ceil)
        self.alpha = math.exp(-DT / lag)
        self.rng = rng or random.Random()
        self.reset()

    # ----- round setup ---------------------------------------------
    def reset(self):
        rng = self.rng

        # 1. l and h with the allowed span
        while True:
            l = rng.randint(*L_RANGE)
            h = rng.randint(*H_RANGE)
            if SPAN_MIN <= h - l <= SPAN_MAX:
                break
        self.l, self.h = float(l), float(h)
        self.t_mid = (self.l + self.h) / 2
        self.K = (self.h - self.l) / math.pi

        # 2. target first
        self.target = rng.uniform(self.l + TARGET_MARGIN, self.h - TARGET_MARGIN)

        # 3. start temperature y0 only where |y0 - target| >= TARGET_MIN_GAP
        intervals = []
        if self.target - TARGET_MIN_GAP > self.l:
            intervals.append((self.l, self.target - TARGET_MIN_GAP))
        if self.target + TARGET_MIN_GAP < self.h:
            intervals.append((self.target + TARGET_MIN_GAP, self.h))
        total = sum(b - a for a, b in intervals)
        pick = rng.uniform(0, total)
        for a, b in intervals:
            if pick <= b - a:
                y0 = a + pick
                break
            pick -= b - a
        else:
            y0 = intervals[-1][1]
        a0 = (self.t_mid - y0) / self.K  # angle that requests y0

        # 4. start steady at y0
        self.angle = min(max(a0, A_MIN), A_MAX)
        self.x = -self.K * self.angle
        self.buffer = (
            deque([self.angle] * self.d, maxlen=self.d) if self.d > 0 else None
        )
        self.k = 0
        self.times = [0.0]
        self.temps = [self.temperature]

    # ----- simulation ----------------------------------------------
    def step(self, a):
        """Advance one DT with knob angle a (rad). Returns temperature."""
        a = min(max(a, A_MIN), A_MAX)
        self.angle = a
        if self.d == 0:
            a_used = a
        else:
            a_used = self.buffer.popleft()  # oldest angle, d steps ago
            self.buffer.append(a)  # newest angle
        self.x = self.alpha * self.x - (1 - self.alpha) * self.K * a_used
        self.k += 1
        self.times.append(self.time)
        self.temps.append(self.temperature)
        return self.temperature

    # ----- read-only values for the view ---------------------------
    @property
    def time(self):
        return self.k * DT

    @property
    def temperature(self):
        return self.t_mid + self.x

    @property
    def in_band(self):
        return abs(self.temperature - self.target) <= GOOD_BAND

    def _requested_temperature(self, a):
        """Internal only: never drawn (it would give the answer away)."""
        return self.t_mid - self.K * a


# =====================================================================
# 3. MODEL SELF-TEST  (python shower_demo.py --test)
# =====================================================================


def run_model_tests():
    rng = random.Random(1)

    # reset rules, many rounds, several settings
    for delay, lag in [(0.0, 0.1), (2.0, 0.5), (5.0, 5.0), (0.3, 1.2)]:
        env = ShowerEnv(delay, lag, rng)
        for _ in range(5000):
            env.reset()
            span = env.h - env.l
            assert SPAN_MIN <= span <= SPAN_MAX, span
            assert (
                env.l + TARGET_MARGIN - 1e-9
                <= env.target
                <= env.h - TARGET_MARGIN + 1e-9
            )
            assert abs(env.temperature - env.target) >= TARGET_MIN_GAP - 1e-9
            assert A_MIN <= env.angle <= A_MAX
            assert env.l - 1e-9 <= env.temperature <= env.h + 1e-9
    print("reset rules: OK (20 000 rounds)")

    # delay check: output must not move for d steps after a knob change
    for delay in [0.0, 0.3, 2.0]:
        env = ShowerEnv(delay, 0.5, rng)
        y0 = env.temperature
        new_a = 0.0 if abs(env.angle) > 0.1 else 1.0
        moved_at = None
        for k in range(1, 2000):
            env.step(new_a)
            if moved_at is None and abs(env.temperature - y0) > 1e-12:
                moved_at = k
        assert moved_at == env.d + 1, (delay, moved_at, env.d)
        final = env._requested_temperature(new_a)
        assert abs(env.temperature - final) < 0.05, (env.temperature, final)
        print(
            f"delay {delay:.1f} s: first change at step {moved_at} (d = {env.d}), "
            f"settles at {env.temperature:.2f} degC (request {final:.2f}): OK"
        )

    # clipping keeps temperature inside [l, h]
    env = ShowerEnv(1.0, 0.2, rng)
    for k in range(3000):
        env.step(10.0 if (k // 300) % 2 else -10.0)
        assert env.l - 1e-9 <= env.temperature <= env.h + 1e-9
    print("clipping: OK")
    print("All model tests passed.")


# =====================================================================
# 4. HELPERS
# =====================================================================


def snap(value, lo, hi):
    v = round(value / SLIDER_STEP) * SLIDER_STEP
    return round(min(max(v, lo), hi), 1)


def fmt_time(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


@dataclass
class Rect:
    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self):
        return self.left + self.width

    @property
    def top(self):
        return self.bottom + self.height

    @property
    def cx(self):
        return self.left + self.width / 2

    @property
    def cy(self):
        return self.bottom + self.height / 2


def compute_layout():
    """Row rectangles from ROW_RATIO, top to bottom."""
    total = sum(ROW_RATIO)
    heights = [WINDOW_HEIGHT * r / total for r in ROW_RATIO]
    top = WINDOW_HEIGHT
    rows = []
    for hgt in heights:
        rows.append(Rect(0, top - hgt, WINDOW_WIDTH, hgt))
        top -= hgt
    row1 = rows[0]
    gw = WINDOW_WIDTH * GRAPH_WIDTH_FRACTION
    return {
        "graph": Rect(row1.left, row1.bottom, gw, row1.height),
        "timer": Rect(row1.left + gw, row1.bottom, WINDOW_WIDTH - gw, row1.height),
        "bar": rows[1],
        "knob": rows[2],
    }


@dataclass
class Settings:
    """Remembered between rounds (SetupView is recreated each time)."""

    delay: float = DEFAULT_DELAY
    lag: float = DEFAULT_LAG


# =====================================================================
# 5. ARCADE PART
# =====================================================================


def run_game():
    import arcade
    import arcade.gui

    PAD = 8  # gap between panels

    def draw_panel_bg(r):
        arcade.draw_lrbt_rectangle_filled(
            r.left + PAD, r.right - PAD, r.bottom + PAD, r.top - PAD, PANEL_COLOR
        )

    def label(
        text, x, y, size=14, color=TEXT_COLOR, ax="center", ay="center", bold=False
    ):
        return arcade.Text(
            text, x, y, color=color, font_size=size, anchor_x=ax, anchor_y=ay, bold=bold
        )

    # ----------------------------- panels -----------------------------
    class GraphPanel:
        """Row 1 left: current temperature and target over the full run."""

        def __init__(self, rect):
            self.rect = rect
            self.px = rect.left + PAD + 60
            self.py = rect.bottom + PAD + 45
            self.pw = rect.right - PAD - 20 - self.px
            self.ph = rect.top - PAD - 40 - self.py
            self.texts = []

        def rebuild(self, env):
            """Axis labels change each round because l and h are random."""
            self.l, self.h = env.l, env.h
            self.texts = [
                label(
                    "Water temperature (°C)",
                    self.rect.left + PAD + 14,
                    self.rect.top - PAD - 18,
                    14,
                    TEXT_COLOR,
                    ax="left",
                    bold=True,
                ),
                label(
                    "Current",
                    self.rect.right - PAD - 190,
                    self.rect.top - PAD - 18,
                    12,
                    TEMP_LINE_COLOR,
                    ax="left",
                ),
                label(
                    "Target",
                    self.rect.right - PAD - 100,
                    self.rect.top - PAD - 18,
                    12,
                    TARGET_COLOR,
                    ax="left",
                ),
                label(
                    "Time (s)",
                    self.px + self.pw / 2,
                    self.rect.bottom + PAD + 10,
                    11,
                    MUTED_TEXT,
                ),
            ]
            first = int(math.ceil(self.l / 5.0) * 5)
            for t in range(first, int(self.h) + 1, 5):
                self.texts.append(
                    label(str(t), self.px - 10, self.ty(t), 11, MUTED_TEXT, ax="right")
                )
            for s in range(0, int(TIMEOUT) + 1, 20):
                self.texts.append(
                    label(str(s), self.tx(s), self.py - 14, 11, MUTED_TEXT)
                )

        def tx(self, t):
            return self.px + t / TIMEOUT * self.pw

        def ty(self, temp):
            return self.py + (temp - self.l) / (self.h - self.l) * self.ph

        def draw(self, env):
            draw_panel_bg(self.rect)
            # grid
            first = int(math.ceil(self.l / 5.0) * 5)
            for t in range(first, int(self.h) + 1, 5):
                arcade.draw_line(
                    self.px, self.ty(t), self.px + self.pw, self.ty(t), GRID_COLOR, 1
                )
            for s in range(20, int(TIMEOUT) + 1, 20):
                arcade.draw_line(
                    self.tx(s), self.py, self.tx(s), self.py + self.ph, GRID_COLOR, 1
                )
            # good band
            arcade.draw_lrbt_rectangle_filled(
                self.px,
                self.px + self.pw,
                self.ty(env.target - GOOD_BAND),
                self.ty(env.target + GOOD_BAND),
                BAND_COLOR,
            )
            # dashed target line
            y = self.ty(env.target)
            x, dash, gap = self.px, 12, 8
            while x < self.px + self.pw:
                arcade.draw_line(
                    x, y, min(x + dash, self.px + self.pw), y, TARGET_COLOR, 2
                )
                x += dash + gap
            # axes
            arcade.draw_line(
                self.px, self.py, self.px + self.pw, self.py, AXIS_COLOR, 2
            )
            arcade.draw_line(
                self.px, self.py, self.px, self.py + self.ph, AXIS_COLOR, 2
            )
            # temperature line (thinned out so drawing stays fast)
            n = len(env.temps)
            stride = max(1, n // 1500)
            pts = [
                (self.tx(env.times[i]), self.ty(env.temps[i]))
                for i in range(0, n, stride)
            ]
            if pts[-1] != (self.tx(env.times[-1]), self.ty(env.temps[-1])):
                pts.append((self.tx(env.times[-1]), self.ty(env.temps[-1])))
            if len(pts) >= 2:
                arcade.draw_line_strip(pts, TEMP_LINE_COLOR, 3)
            for t in self.texts:
                t.draw()

    class TimerPanel:
        """Row 1 right: count-up timer (buttons are placed here by the view)."""

        def __init__(self, rect, settings):
            self.rect = rect
            self.title = label("Time", rect.cx, rect.top - PAD - 26, 16, MUTED_TEXT)
            self.clock = label(
                "00:00", rect.cx, rect.top - PAD - 72, 40, TEXT_COLOR, bold=True
            )
            self.info = label(
                f"Delay {settings.delay:.1f} s   Lag {settings.lag:.1f} s",
                rect.cx,
                rect.top - PAD - 112,
                12,
                MUTED_TEXT,
            )

        def button_slots(self, w, h, gap):
            """(x, y) for two stacked buttons near the bottom of the panel."""
            x = self.rect.cx - w / 2
            y1 = self.rect.bottom + PAD + 16
            return (x, y1 + h + gap), (x, y1)

        def draw(self, env, paused):
            draw_panel_bg(self.rect)
            text = fmt_time(env.time)
            if self.clock.text != text:
                self.clock.text = text
            self.clock.color = MUTED_TEXT if paused else TEXT_COLOR
            self.title.text = "Paused" if paused else "Time"
            self.title.draw()
            self.clock.draw()
            self.info.draw()

    class BarPanel:
        """Row 2: cold-to-hot bar with target and current markers, no numbers."""

        SLICES = 80

        def __init__(self, rect):
            self.rect = rect
            self.bl = rect.left + 120
            self.br = rect.right - 120
            self.bb = rect.cy - 22
            self.bt = rect.cy + 22
            self.texts = [
                label("COLD", rect.left + 60, rect.cy, 16, COLD_COLOR, bold=True),
                label("HOT", rect.right - 60, rect.cy, 16, HOT_COLOR, bold=True),
            ]
            self.target_txt = label(
                "Target", 0, self.bt + 52, 13, TARGET_COLOR, bold=True
            )
            self.now_txt = label("Now", 0, self.bb - 52, 13, NOW_COLOR, bold=True)

        def bx(self, env, temp):
            f = (temp - env.l) / (env.h - env.l)
            return self.bl + min(max(f, 0.0), 1.0) * (self.br - self.bl)

        def draw(self, env):
            draw_panel_bg(self.rect)
            w = (self.br - self.bl) / self.SLICES
            for i in range(self.SLICES):
                c = lerp_color(COLD_COLOR, HOT_COLOR, i / (self.SLICES - 1))
                arcade.draw_lrbt_rectangle_filled(
                    self.bl + i * w, self.bl + (i + 1) * w + 1, self.bb, self.bt, c
                )
            # good band
            arcade.draw_lrbt_rectangle_filled(
                self.bx(env, env.target - GOOD_BAND),
                self.bx(env, env.target + GOOD_BAND),
                self.bb - 6,
                self.bt + 6,
                (255, 255, 255, 70),
            )
            # target arrow (above, pointing down)
            x = self.bx(env, env.target)
            arcade.draw_triangle_filled(
                x, self.bt + 6, x - 14, self.bt + 32, x + 14, self.bt + 32, TARGET_COLOR
            )
            self.target_txt.x = x
            # current arrow (below, pointing up); green when comfortable
            x = self.bx(env, env.temperature)
            c = TARGET_COLOR if env.in_band else NOW_COLOR
            arcade.draw_triangle_filled(
                x, self.bb - 6, x - 14, self.bb - 32, x + 14, self.bb - 32, c
            )
            self.now_txt.x = x
            self.now_txt.color = c
            for t in self.texts:
                t.draw()
            self.target_txt.draw()
            self.now_txt.draw()

    class KnobPanel:
        """Row 3: knob; clockwise (negative angle) = hotter. No numbers."""

        def __init__(self, rect):
            self.rect = rect
            self.r = rect.height * 0.26
            self.ar = self.r + 14  # radius of the limit arc
            self.cx = rect.cx
            # keep circle and arc inside the panel
            self.cy = rect.bottom + PAD + 14 + self.r
            self.texts = [
                label(
                    "COLD", self.cx - self.ar - 40, self.cy, 13, COLD_COLOR, bold=True
                ),
                label("HOT", self.cx + self.ar + 34, self.cy, 13, HOT_COLOR, bold=True),
                label("←  colder", rect.left + 40, rect.cy, 14, MUTED_TEXT, ax="left"),
                label(
                    "hotter  →", rect.right - 40, rect.cy, 14, MUTED_TEXT, ax="right"
                ),
            ]

        def draw(self, angle):
            draw_panel_bg(self.rect)
            ar = self.ar
            arcade.draw_arc_outline(
                self.cx, self.cy, 2 * ar, 2 * ar, AXIS_COLOR, 0, 180, 3
            )
            arcade.draw_circle_filled(self.cx, self.cy, self.r, (70, 78, 94))
            arcade.draw_circle_outline(self.cx, self.cy, self.r, AXIS_COLOR, 2)
            # angle 0 points up; negative angle turns clockwise (to the right)
            px = self.cx - math.sin(angle) * (self.r - 6)
            py = self.cy + math.cos(angle) * (self.r - 6)
            arcade.draw_line(self.cx, self.cy, px, py, TEXT_COLOR, 5)
            arcade.draw_circle_filled(self.cx, self.cy, 6, TEXT_COLOR)
            for t in self.texts:
                t.draw()

    # ----------------------------- views ------------------------------
    class SetupView(arcade.View):
        """Start screen and end-of-round screen."""

        def __init__(self, settings, message="", success=True):
            super().__init__()
            self.settings = settings
            msg_color = (
                MUTED_TEXT
                if not message
                else (TARGET_COLOR if success else TEMP_LINE_COLOR)
            )
            self.ui = arcade.gui.UIManager()

            box = arcade.gui.UIBoxLayout(vertical=True, space_between=14)
            box.add(
                arcade.gui.UILabel(
                    text="Shower Delay Demo",
                    font_size=32,
                    bold=True,
                    text_color=TEXT_COLOR,
                    width=700,
                    align="center",
                )
            )
            box.add(
                arcade.gui.UILabel(
                    text=message or "Find the comfortable temperature.",
                    font_size=18,
                    text_color=msg_color,
                    width=700,
                    align="center",
                )
            )

            self.delay_label = arcade.gui.UILabel(
                text="", font_size=16, text_color=TEXT_COLOR, width=420, align="left"
            )
            self.delay_slider = arcade.gui.UISlider(
                value=settings.delay,
                min_value=DELAY_RANGE[0],
                max_value=DELAY_RANGE[1],
                step=SLIDER_STEP,
                width=420,
                height=30,
            )
            self.lag_label = arcade.gui.UILabel(
                text="", font_size=16, text_color=TEXT_COLOR, width=420, align="left"
            )
            self.lag_slider = arcade.gui.UISlider(
                value=settings.lag,
                min_value=LAG_RANGE[0],
                max_value=LAG_RANGE[1],
                step=SLIDER_STEP,
                width=420,
                height=30,
            )
            for w in (
                self.delay_label,
                self.delay_slider,
                self.lag_label,
                self.lag_slider,
            ):
                box.add(w)

            start = arcade.gui.UIFlatButton(text="Start shower", width=260, height=56)
            box.add(start)

            @start.event("on_click")
            def _on_start(event):
                env = ShowerEnv(self.settings.delay, self.settings.lag)
                self.window.show_view(ShowerView(self.settings, env))

            anchor = arcade.gui.UIAnchorLayout()
            anchor.add(box, anchor_x="center", anchor_y="center")
            self.ui.add(anchor)
            self._shown = (None, None)
            self._refresh_values()

        def _refresh_values(self):
            # read sliders, snap to SLIDER_STEP, remember in settings
            self.settings.delay = snap(self.delay_slider.value, *DELAY_RANGE)
            self.settings.lag = snap(self.lag_slider.value, *LAG_RANGE)
            shown = (self.settings.delay, self.settings.lag)
            if shown != self._shown:
                self.delay_label.text = f"Delay T: {self.settings.delay:.1f} s"
                self.lag_label.text = f"Lag τ: {self.settings.lag:.1f} s"
                self._shown = shown

        def on_show_view(self):
            self.window.background_color = BG_COLOR
            self.ui.enable()

        def on_hide_view(self):
            self.ui.disable()

        def on_update(self, delta_time):
            self._refresh_values()

        def on_draw(self):
            self.clear()
            self.ui.draw()

    class ShowerView(arcade.View):
        """The simulation: graph + timer, bar, knob."""

        RUNNING, PAUSED = "running", "paused"

        def __init__(self, settings, env):
            super().__init__()
            self.settings = settings
            self.env = env
            lay = compute_layout()
            self.graph = GraphPanel(lay["graph"])
            self.timer = TimerPanel(lay["timer"], settings)
            self.bar = BarPanel(lay["bar"])
            self.knob = KnobPanel(lay["knob"])

            # buttons inside the timer panel
            self.ui = arcade.gui.UIManager()
            bw, bh, gap = 200, 46, 14
            (x1, y1), (x2, y2) = self.timer.button_slots(bw, bh, gap)
            self.pause_btn = arcade.gui.UIFlatButton(
                text="Pause", x=x1, y=y1, width=bw, height=bh
            )
            self.restart_btn = arcade.gui.UIFlatButton(
                text="Restart", x=x2, y=y2, width=bw, height=bh
            )
            self.ui.add(self.pause_btn)
            self.ui.add(self.restart_btn)

            @self.pause_btn.event("on_click")
            def _on_pause(event):
                self.toggle_pause()

            @self.restart_btn.event("on_click")
            def _on_restart(event):
                self.env.reset()
                self.start_round()

            self.held = set()
            self.start_round()

        # ----- round control -----------------------------------------
        def start_round(self):
            self.angle = self.env.angle
            self.accumulator = 0.0
            self.hold_time = 0.0
            self.repeat_timer = 0.0
            self.state = self.RUNNING
            self.pause_btn.text = "Pause"
            self.graph.rebuild(self.env)

        def toggle_pause(self):
            if self.state == self.RUNNING:
                self.state = self.PAUSED
                self.pause_btn.text = "Resume"
            else:
                self.state = self.RUNNING
                self.pause_btn.text = "Pause"
                self.accumulator = 0.0  # no burst of steps after a pause
                self.repeat_timer = 0.0

        def end_round(self, message, success):
            self.window.show_view(SetupView(self.settings, message, success))

        # ----- input -------------------------------------------------
        def knob_direction(self):
            d = 0
            if arcade.key.LEFT in self.held:
                d += 1  # counter-clockwise = colder = angle up
            if arcade.key.RIGHT in self.held:
                d -= 1  # clockwise = hotter = angle down
            return d

        def turn(self, direction):
            self.angle = min(max(self.angle + direction * KNOB_STEP, A_MIN), A_MAX)

        def on_key_press(self, symbol, modifiers):
            if symbol in (arcade.key.LEFT, arcade.key.RIGHT):
                self.held.add(symbol)
                if self.state == self.RUNNING:
                    self.turn(1 if symbol == arcade.key.LEFT else -1)
                    self.repeat_timer = 0.0
            elif symbol == arcade.key.SPACE:
                self.toggle_pause()

        def on_key_release(self, symbol, modifiers):
            self.held.discard(symbol)

        # ----- update ------------------------------------------------
        def on_update(self, delta_time):
            if self.state != self.RUNNING:
                return
            delta_time = min(delta_time, MAX_FRAME_TIME)

            # 1. held keys repeat at KEY_REPEAT_RATE
            direction = self.knob_direction()
            if direction:
                self.repeat_timer += delta_time
                period = 1.0 / KEY_REPEAT_RATE
                while self.repeat_timer >= period:
                    self.repeat_timer -= period
                    self.turn(direction)
            else:
                self.repeat_timer = 0.0

            # 2. step the model in real time
            self.accumulator += delta_time
            while self.accumulator >= DT:
                self.accumulator -= DT
                self.env.step(self.angle)

                # 3. continuous time in the good band
                self.hold_time = self.hold_time + DT if self.env.in_band else 0.0

                # 4. success, then timeout
                if self.hold_time >= SUCCESS_HOLD - 1e-9:
                    self.end_round(
                        f"Well done! Comfortable in {fmt_time(self.env.time)}", True
                    )
                    return
                if self.env.time >= TIMEOUT - 1e-9:
                    self.end_round("Time's up! Try again.", False)
                    return

        # ----- draw --------------------------------------------------
        def on_show_view(self):
            self.window.background_color = BG_COLOR
            self.ui.enable()

        def on_hide_view(self):
            self.ui.disable()

        def on_draw(self):
            self.clear()
            self.graph.draw(self.env)
            self.timer.draw(self.env, self.state == self.PAUSED)
            # self.bar.draw(self.env)
            self.knob.draw(self.angle)
            self.ui.draw()

    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    window.show_view(SetupView(Settings()))
    arcade.run()


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_model_tests()
    else:
        run_game()
