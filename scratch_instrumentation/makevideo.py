"""Composite the captured frames into one annotated side-by-side demo video.

SCRATCH ONLY.

Left  = the refiner's own persistent sim, whose transition function is
        `sim.set_state(x); sim.step(u)`.
Right = the real environment, stepped forward with the identical actions.

Every frame carries the step index, the full qpos state vector (cube pose, base, arm,
and the eight Robotiq finger joints called out on their own line), the action vector,
and -- on the refinement side only -- the `set_state` that fired immediately before
that step and exactly what it did and did not change.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 1640, 1184
PANEL_W, PANEL_H = 780, 585
PANEL_Y = 100
LX, RX = 20, 840
ANN_Y = PANEL_Y + PANEL_H + 8
DIV_Y = 1070

BG = (14, 16, 20)
FG = (232, 234, 238)
DIM = (140, 146, 156)
BLUE = (0x00, 0x72, 0xB2)  # refinement: set_state intervenes every step
ORANGE = (0xD5, 0x5E, 0x00)  # execution: nothing intervenes
RED = (0xE0, 0x30, 0x30)
GREEN = (0x2E, 0xA0, 0x43)
YELLOW = (0xF0, 0xC4, 0x20)

FD = "/usr/share/fonts/truetype/dejavu"
F_TITLE = ImageFont.truetype(f"{FD}/DejaVuSans-Bold.ttf", 27)
F_SUB = ImageFont.truetype(f"{FD}/DejaVuSans.ttf", 18)
F_HEAD = ImageFont.truetype(f"{FD}/DejaVuSans-Bold.ttf", 19)
F_M = ImageFont.truetype(f"{FD}/DejaVuSansMono.ttf", 15)
F_MB = ImageFont.truetype(f"{FD}/DejaVuSansMono-Bold.ttf", 15)
F_BIG = ImageFont.truetype(f"{FD}/DejaVuSansMono-Bold.ttf", 17)
F_TINY = ImageFont.truetype(f"{FD}/DejaVuSansMono.ttf", 12)
F_CARD = ImageFont.truetype(f"{FD}/DejaVuSans.ttf", 25)
F_CARDB = ImageFont.truetype(f"{FD}/DejaVuSans-Bold.ttf", 31)
F_CARDM = ImageFont.truetype(f"{FD}/DejaVuSansMono.ttf", 21)

FINGER_LABELS = ["r_drv", "r_cpl", "r_spr", "r_fol", "l_drv", "l_cpl", "l_spr", "l_fol"]

# qpos layout, read off the MuJoCo joint table for Tossing3D-o1 (nq=39).
QP_CUBE = slice(0, 7)
QP_BASE = slice(21, 24)
QP_ARM = slice(24, 31)
QP_FING = slice(31, 39)


def vecs(v, fmt="%+.4f"):
    return " ".join(fmt % x for x in v)


class Comp:
    def __init__(self, capdir: Path):
        self.cap = capdir
        self.steps = {}
        self.events = []
        for line in open(capdir / "events.jsonl"):
            r = json.loads(line)
            self.events.append(r)
            if r["kind"] == "step":
                self.steps[(r["env"], r["step"])] = r
        self.summary = json.load(open(capdir / "summary.json"))
        self.frames: list[Image.Image] = []

        # aligned pairs: (refinement sim1 step, execution sim0 step)
        self.pairs = [(k, k) for k in range(56)] + [
            (105 + i, 56 + i) for i in range(55)
        ]
        self.div = []
        for p, e in self.pairs:
            a = np.asarray(self.steps[("sim1", p)]["cube"])
            b = np.asarray(self.steps[("sim0", e)]["cube"])
            self.div.append(float(np.linalg.norm(a - b)))
        self.divplot = self._render_divplot()

    # ---------------------------------------------------------------- div plot
    def _render_divplot(self) -> Image.Image:
        fig = plt.figure(figsize=(16.0, 0.86), dpi=100)
        ax = fig.add_axes([0.045, 0.30, 0.945, 0.62])
        d = np.maximum(np.asarray(self.div), 1e-11)
        ax.semilogy(d, color="#D55E00", lw=1.8)
        ax.axvline(55.5, color="#F0C420", lw=1.2, ls="--")
        ax.set_xlim(0, len(d) - 1)
        ax.set_ylim(1e-11, 3)
        ax.set_facecolor("#0e1014")
        fig.patch.set_facecolor("#0e1014")
        for s in ax.spines.values():
            s.set_color("#5a6068")
        ax.tick_params(colors="#c8ccd2", labelsize=7)
        ax.set_yticks([1e-10, 1e-6, 1e-3, 1e0])
        ax.grid(alpha=0.18, which="both", lw=0.4)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img = Image.fromarray(buf)
        plt.close(fig)
        return img.resize((1600, 86))

    # ---------------------------------------------------------------- helpers
    def new_canvas(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        im = Image.new("RGB", (W, H), BG)
        return im, ImageDraw.Draw(im)

    def hold(self, im: Image.Image, n: int) -> None:
        for _ in range(n):
            self.frames.append(im.copy())

    def load_frame(self, name: str) -> Image.Image:
        return Image.open(self.cap / "frames" / name).convert("RGB").resize(
            (PANEL_W, PANEL_H), Image.LANCZOS
        )

    # ---------------------------------------------------------------- cards
    def card(self, title: str, lines: list[tuple[str, str]], n: int) -> None:
        im, d = self.new_canvas()
        y = 74
        words, line = title.split(), ""
        for w in words + [None]:
            trial = (line + " " + w).strip() if w else line
            if w and d.textlength(trial, font=F_CARDB) < W - 120:
                line = trial
                continue
            d.text((60, y), line, font=F_CARDB, fill=FG)
            y += 42
            line = w or ""
        y += 8
        d.line((60, y, W - 60, y), fill=DIM, width=2)
        y += 44
        for kind, txt in lines:
            if kind == "gap":
                y += 26
                continue
            f = {"t": F_CARD, "m": F_CARDM, "b": F_CARDB}[kind[0]]
            col = FG
            if kind.endswith("!"):
                col = YELLOW
            elif kind.endswith("b"):
                col = BLUE
            elif kind.endswith("o"):
                col = ORANGE
            elif kind.endswith("d"):
                col = DIM
            d.text((60, y), txt, font=f, fill=col)
            y += int(f.size * 1.55)
        self.hold(im, n)

    # ---------------------------------------------------------------- panels
    def annotate(
        self,
        d: ImageDraw.ImageDraw,
        x0: int,
        rec: dict | None,
        *,
        side: str,
        colour,
        peer: dict | None,
    ) -> None:
        y = ANN_Y
        if rec is None:
            d.text((x0, y + 60), "(not running -- refinement has not", font=F_SUB, fill=DIM)
            d.text((x0, y + 88), " reached execution yet)", font=F_SUB, fill=DIM)
            return

        # --- step + set_state banner
        d.text((x0, y), f"STEP {rec['step']:>4d}", font=F_BIG, fill=colour)
        ss = rec["set_state"]
        bx = x0 + 130
        if ss is not None:
            big = ss["max_abs_qpos_change"] > 0.1
            box = (bx, y - 3, x0 + PANEL_W, y + 23)
            d.rectangle(box, fill=(90, 20, 20) if big else (30, 42, 58))
            d.text(
                (bx + 8, y + 1),
                "set_state FIRED  max|dqpos|=%.2e  max|dfinger|=%.2e"
                % (ss["max_abs_qpos_change"], ss["max_abs_finger_change"]),
                font=F_MB,
                fill=RED if big else (150, 190, 230),
            )
        else:
            d.rectangle((bx, y - 3, x0 + PANEL_W, y + 23), fill=(22, 34, 24))
            d.text(
                (bx + 8, y + 1),
                "no set_state -- physics runs forward from the last step",
                font=F_MB,
                fill=GREEN,
            )
        y += 32

        # --- the eight finger joints, the star of the show
        diff = None
        if peer is not None:
            diff = np.abs(
                np.asarray(rec["pre_fingers"]) - np.asarray(peer["pre_fingers"])
            )
        hot = diff is not None and diff.max() > 0.05
        d.rectangle(
            (x0, y - 2, x0 + PANEL_W, y + 68),
            fill=(64, 18, 18) if hot else (26, 30, 38),
        )
        hdr = "GRIPPER FINGER JOINTS  qpos[31:39]  at START of step"
        if diff is not None:
            hdr += "   |d| = %.4f rad" % diff.max()
        d.text((x0 + 6, y + 1), hdr, font=F_M, fill=RED if hot else DIM)
        d.text((x0 + 6, y + 21), " " + "   ".join(FINGER_LABELS), font=F_M, fill=DIM)
        d.text(
            (x0 + 6, y + 43),
            vecs(rec["pre_fingers"]),
            font=F_BIG,
            fill=RED if hot else colour,
        )
        y += 76

        # --- state vector
        q = np.asarray(rec["post_qpos"])
        c = rec["cube"]
        d.text((x0, y), "STATE (qpos, after this step)", font=F_M, fill=DIM)
        y += 20
        d.text(
            (x0 + 6, y),
            "cube  x=%+.4f  y=%+.4f  z=%+.4f" % (c[0], c[1], c[2]),
            font=F_MB,
            fill=colour,
        )
        y += 19
        d.text((x0 + 6, y), "base  " + vecs(q[QP_BASE]), font=F_M, fill=FG)
        y += 19
        d.text((x0 + 6, y), "arm   " + vecs(q[QP_ARM]), font=F_M, fill=FG)
        y += 22

        # --- action
        a = np.asarray(rec["action"])
        shape = tuple(rec["action_shape"])
        if a.ndim == 2:
            d.text(
                (x0, y),
                f"ACTION  control schedule {shape[0]}x{shape[1]}  (rows 0 and {shape[0]-1})",
                font=F_M,
                fill=YELLOW,
            )
            y += 20
            for lbl, row in (("row 0 ", a[0]), (f"row{shape[0]-1:>2d} ", a[-1])):
                d.text((x0 + 6, y), lbl + vecs(row[:9], "%+.3f"), font=F_M, fill=FG)
                y += 18
                d.text((x0 + 6, y), "      " + vecs(row[9:], "%+.3f"), font=F_M, fill=FG)
                y += 18
        else:
            d.text((x0, y), f"ACTION  shape {shape}", font=F_M, fill=DIM)
            y += 20
            d.text(
                (x0 + 6, y),
                "base  " + vecs(a[0:3], "%+.4f") + "   grip %+.3f" % a[10],
                font=F_M,
                fill=FG,
            )
            y += 19
            d.text((x0 + 6, y), "arm   " + vecs(a[3:10], "%+.4f"), font=F_M, fill=FG)
            y += 19
            if a.size == 18:
                d.text((x0 + 6, y), "qvel  " + vecs(a[11:18], "%+.4f"), font=F_M, fill=FG)
            else:
                d.text((x0 + 6, y), "qvel  (none -- 11-wide action)", font=F_M, fill=DIM)
            y += 22

        # --- full qpos dump
        d.text((x0, y), "full qpos[0:39]", font=F_TINY, fill=DIM)
        y += 15
        for i in range(0, 39, 10):
            d.text(
                (x0 + 6, y),
                "%2d " % i + " ".join("%+.3f" % v for v in q[i : i + 10]),
                font=F_TINY,
                fill=(180, 186, 196),
            )
            y += 14

    def scene(
        self,
        *,
        title: str,
        sub: str,
        left_frame: str | None,
        right_frame: str | None,
        left_rec: dict | None,
        right_rec: dict | None,
        cursor: int | None,
        note: str | None = None,
    ) -> None:
        im, d = self.new_canvas()
        d.text((20, 14), title, font=F_TITLE, fill=FG)
        d.text((20, 50), sub, font=F_SUB, fill=DIM)

        for x0, fr, lab, col in (
            (LX, left_frame, "REFINEMENT  --  sim.set_state(x); sim.step(u)", BLUE),
            (RX, right_frame, "EXECUTION  --  the real environment", ORANGE),
        ):
            d.text((x0, PANEL_Y - 26), lab, font=F_HEAD, fill=col)
            if fr is None:
                d.rectangle(
                    (x0, PANEL_Y, x0 + PANEL_W, PANEL_Y + PANEL_H), fill=(24, 26, 32)
                )
                d.text(
                    (x0 + 150, PANEL_Y + PANEL_H // 2),
                    "not yet executed",
                    font=F_HEAD,
                    fill=DIM,
                )
            else:
                im.paste(self.load_frame(fr), (x0, PANEL_Y))
            d.rectangle(
                (x0 - 2, PANEL_Y - 2, x0 + PANEL_W + 2, PANEL_Y + PANEL_H + 2),
                outline=col,
                width=2,
            )

        self.annotate(d, LX, left_rec, side="ref", colour=BLUE, peer=right_rec)
        self.annotate(d, RX, right_rec, side="exec", colour=ORANGE, peer=left_rec)

        # divergence strip
        im.paste(self.divplot, (20, DIV_Y))
        d.text(
            (20, DIV_Y - 18),
            "|refinement cube - execution cube|  (m, log scale) over the 111 aligned steps",
            font=F_M,
            fill=DIM,
        )
        if cursor is not None:
            px = 20 + int(0.045 * 1600) + int(
                (cursor / (len(self.div) - 1)) * (0.945 * 1600)
            )
            d.line((px, DIV_Y + 4, px, DIV_Y + 62), fill=YELLOW, width=2)
            d.text(
                (20 + 1600 - 330, DIV_Y + 62),
                "|d| = %.3e m" % self.div[cursor],
                font=F_MB,
                fill=YELLOW,
            )
        if note:
            d.text((20 + 780, DIV_Y - 18), note, font=F_MB, fill=YELLOW)
        self.frames.append(im)

    # ---------------------------------------------------------------- build
    def build(self) -> None:
        s = self.summary
        simf = s["sim_final_cube"]
        exf = s["exec_final_cube"]
        dx = simf[0] - exf[0]

        self.card(
            "Tossing3D, seed 101: bilevel refinement validates a throw the real "
            "environment cannot reproduce",
            [
                ("t", "Bilevel planning refines a plan by re-simulating it. Its transition"),
                ("t", "function is one line:"),
                ("gap", ""),
                ("m!", "    def transition_fn(x, u):        # kinder_bilevel_planning"),
                ("m!", "        sim.set_state(x)            #   <- a PERSISTENT sim"),
                ("m!", "        return sim.step(u)"),
                ("gap", ""),
                ("t", "`sim` is one long-lived MuJoCo environment, reused across every"),
                ("t", "sampling attempt. `set_state` restores the cube and the arm, but it"),
                ("td", "does not touch the Robotiq 2F-85's eight finger joints (qpos 31..38)."),
                ("gap", ""),
                ("t", "So a REJECTED attempt leaves the fingers where it ended them, and the"),
                ("t", "NEXT attempt starts from that pose -- a grasp geometry execution never has."),
            ],
            34,
        )
        self.card(
            "What you are about to watch",
            [
                ("mb", "LEFT   refinement's own rollout, inside the planner's sim"),
                ("m", "       set_state fires before every single step"),
                ("gap", ""),
                ("mo", "RIGHT  execution: the same actions, in the real environment"),
                ("m", "       set_state never fires; physics runs forward"),
                ("gap", ""),
                ("t", "The action sequences are identical -- measured this run:"),
                ("m!", "   111 planned actions, 111 executed, max |delta action| = %.2e"
                    % s["max_action_diff"]),
                ("gap", ""),
                ("t", "Act 1  the pick               (both paths agree)"),
                ("t", "Act 2  toss attempt #2        REJECTED by the refiner"),
                ("t", "Act 3  toss attempt #3        ACCEPTED -- and executed"),
            ],
            34,
        )

        # ---- Act 1: the pick, aligned
        for k in range(56):
            lr = self.steps[("sim1", k)]
            rr = self.steps[("sim0", k)]
            self.scene(
                title="Act 1 -- the pick  (refinement attempt #1, accepted)",
                sub="Aligned step %d/55.  Both paths agree here: |d| = %.2e m"
                % (k, self.div[k]),
                left_frame=lr["frame"],
                right_frame=rr["frame"],
                left_rec=lr,
                right_rec=rr,
                cursor=k,
            )

        self.card(
            "Act 2 -- the refiner now samples the toss",
            [
                ("t", "The abstract plan is  pick_cube  ->  move_to_toss_location_and_toss."),
                ("t", "The pick took one sampling attempt. The toss takes three."),
                ("gap", ""),
                ("t", "Attempt #2 (the first toss sample) starts from the fingers the pick"),
                ("t", "left closed around the cube -- the same pose execution has:"),
                ("gap", ""),
                ("mb", "  refinement, sim1 step 56   " + vecs(
                    self.steps[("sim1", 56)]["pre_fingers"])),
                ("mo", "  execution,  sim0 step 56   " + vecs(
                    self.steps[("sim0", 56)]["pre_fingers"])),
                ("gap", ""),
                ("td", "Nothing is wrong yet. Watch where attempt #2 leaves the fingers."),
            ],
            34,
        )

        # ---- Act 2: rejected attempt, refinement only
        last_exec_pick = self.steps[("sim0", 55)]
        for i in range(56, 105):
            lr = self.steps[("sim1", i)]
            self.scene(
                title="Act 2 -- refinement toss attempt #2  (this one gets REJECTED)",
                sub="Refinement sim step %d.  Nothing has been executed yet; the robot has "
                "not moved in the real world." % i,
                left_frame=lr["frame"],
                right_frame=None,
                left_rec=lr,
                right_rec=None,
                cursor=None,
                note="attempt #2 has no execution counterpart",
            )

        a2 = self.steps[("sim1", 104)]
        a3 = self.steps[("sim1", 105)]
        e0 = self.steps[("sim0", 56)]
        fdiff = float(
            np.max(
                np.abs(
                    np.asarray(a3["pre_fingers"]) - np.asarray(e0["pre_fingers"])
                )
            )
        )
        self.card(
            "Attempt #2 REJECTED -- and this is the moment the bug happens",
            [
                ("m", "attempt #2 ends with the cube at  x=%+.4f y=%+.4f z=%+.4f"
                    % tuple(a2["cube"])),
                ("m", "-> not in the goal region, so the refiner backtracks and resamples."),
                ("gap", ""),
                ("t", "But the sim is persistent. Attempt #2 released the gripper, so it"),
                ("t", "ends with the fingers open, and attempt #3 inherits them:"),
                ("gap", ""),
                ("m", "  end of attempt #2   " + vecs(a2["post_fingers"])),
                ("mb", "  start of attempt #3  " + vecs(a3["pre_fingers"])),
                ("mo", "  execution, same step " + vecs(e0["pre_fingers"])),
                ("gap", ""),
                ("m!", "  max componentwise difference:  %.4f rad" % fdiff),
                ("gap", ""),
                ("m!", "  set_state at that step:  max|dqpos| = %.4e   but   max|dfinger| = %.4e"
                    % (a3["set_state"]["max_abs_qpos_change"],
                       a3["set_state"]["max_abs_finger_change"])),
                ("td", "  It moved the whole robot and cube back -- and left the fingers alone."),
            ],
            50,
        )

        # ---- Act 3: accepted attempt vs execution
        for i in range(55):
            p, e = 105 + i, 56 + i
            lr = self.steps[("sim1", p)]
            rr = self.steps[("sim0", e)]
            k = 56 + i
            self.scene(
                title="Act 3 -- refinement toss attempt #3 (ACCEPTED) vs its execution",
                sub="Aligned step %d/110   (refinement sim1 step %d, execution sim0 step %d)"
                "   identical action, |d| = %.4f m" % (k, p, e, self.div[k]),
                left_frame=lr["frame"],
                right_frame=rr["frame"],
                left_rec=lr,
                right_rec=rr,
                cursor=k,
            )

        # ---- end: freeze the last pair, then the numbers
        lr = self.steps[("sim1", 159)]
        rr = self.steps[("sim0", 110)]
        for _ in range(14):
            self.scene(
                title="Act 3 -- final state",
                sub="Same 111 actions. Two different outcomes.",
                left_frame=lr["frame"],
                right_frame=rr["frame"],
                left_rec=lr,
                right_rec=rr,
                cursor=110,
            )

        self.card(
            "The outcome gap",
            [
                ("mb", "refinement said the cube lands at   x=%+.4f  y=%+.4f  z=%+.4f"
                    % tuple(simf)),
                ("mo", "execution put it at                 x=%+.4f  y=%+.4f  z=%+.4f"
                    % tuple(exf)),
                ("gap", ""),
                ("m!", "along x:            %.4f m" % dx),
                ("m!", "3-D offset:         %.4f m" % s["final_offset_norm"]),
                ("gap", ""),
                ("m", "same 111 actions, max |delta action| = %.2e" % s["max_action_diff"]),
                ("m", "set_state calls, refinement path:  %d" % 160),
                ("m", "set_state calls, execution path:   0"),
                ("gap", ""),
                ("m!", "env._check_goals() after execution:  %s"
                    % s["check_goals_at_end"]),
                ("td", "Refinement accepted this plan. Execution reproduces the outcome"),
                ("td", "refinement had already REJECTED as attempt #2."),
            ],
            60,
        )

    def write(self, path: Path, fps: int) -> None:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(path),
            (W, H),
            fps=fps,
            quality=8,
            macro_block_size=8,
            output_params=["-pix_fmt", "yuv420p"],
        )
        writer.send(None)
        for f in self.frames:
            writer.send(np.asarray(f).tobytes())
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stills", nargs="*", type=int, default=[])
    ap.add_argument("--stills-dir", default=None)
    args = ap.parse_args()

    c = Comp(Path(args.cap))
    c.build()
    print("composited frames:", len(c.frames))
    c.write(Path(args.out), args.fps)
    print("wrote", args.out)
    if args.stills:
        sd = Path(args.stills_dir or ".")
        for i in args.stills:
            p = sd / ("still-%04d.png" % i)
            c.frames[i].save(p)
            print("still", p)


if __name__ == "__main__":
    main()
