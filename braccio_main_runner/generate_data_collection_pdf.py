"""Generate Data_Collection_Guide.pdf — step-by-step field guide for Stage 0."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)

OUT = "Data_Collection_Guide.pdf"
NAVY = colors.HexColor('#1a3a6c')
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=20, spaceAfter=10, textColor=NAVY)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=14, textColor=NAVY)
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontSize=11, spaceAfter=5, spaceBefore=6, textColor=colors.HexColor('#333'))
B = ParagraphStyle('B', parent=ss['BodyText'], fontSize=10, spaceAfter=5, leading=13)
STEP = ParagraphStyle('Step', parent=B, leftIndent=16)
CODE = ParagraphStyle('Code', parent=B, fontName='Courier', fontSize=9,
                      backColor=colors.HexColor('#f0f0f0'), leftIndent=10, borderPadding=4)
NOTE = ParagraphStyle('Note', parent=B, fontSize=9, leftIndent=10,
                      textColor=colors.HexColor('#555'), fontName='Helvetica-Oblique')

def P(t, s=B): return Paragraph(t, s)
def tbl(data, widths):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#888')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    return t

s = []

# ── Cover ────────────────────────────────────────────────────────────────────
s += [P("Reactive Robot", H1),
      P("Data Collection Field Guide", H1),
      Spacer(1, 6),
      P("Stage 0 protocol for calibrating the RL simulation to your real hardware.", B),
      P("Total session time: ~75 minutes (5 min setup + 70 min scenarios).", B),
      Spacer(1, 10),
      P("Purpose", H2),
      P("The observations (ToF grids, IR, IMU, obstacle map) you collect here are used to "
        "calibrate the simulator's noise model so the policy trained in PyBullet transfers "
        "cleanly to the real arm. You are <b>not</b> teaching the arm what to do in this "
        "session — you are just giving the sim realistic sensor statistics.", B),
      P("You only have to do this once (unless hardware changes).", B)]

# ── What you need ────────────────────────────────────────────────────────────
s += [P("What you need before starting", H2)]
for x in [
    "Braccio arm powered on, connected to <b>/dev/ttyACM0</b>",
    "Teensy sensor board on <b>/dev/ttyACM1</b>",
    "Cylindrical obstacle, 6–10 cm diameter (water bottle, mug, tin can)",
    "Clear workspace around the arm (nothing above 30 cm within arm radius)",
    "Stopwatch or phone timer",
    "Terminal open in the <b>braccio_main_runner/</b> directory",
]:
    s.append(P("• " + x, B))

# ── Launch controller ────────────────────────────────────────────────────────
s += [P("Launch the controller", H2),
      P("From <b>braccio_main_runner/</b> run:", B),
      P("python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1", CODE),
      P("The curses display shows current theta / r / z / delta and sensor states. "
        "The RL transition recorder starts automatically — no key needed. "
        "Transitions save to <b>logs/rl_transitions_TIMESTAMP.npz</b> when you quit.", B)]

# ── Key reference ────────────────────────────────────────────────────────────
s += [P("Key reference (memorize before starting)", H2)]
keys = [['Key','Action'],
        ['A / D','theta −5° / +5°'],
        ['W / S','r +10 mm / −10 mm'],
        ['Q / E','z +10 mm / −10 mm'],
        ['Z','Toggle autonomous sweep (start/stop)'],
        ['X','Open sequence editor'],
        ['M','Open saved-states menu'],
        ['H','Return to home pose'],
        ['G','Start/stop ToF CSV log'],
        ['ESC','Quit controller (flushes NPZ to disk)']]
s += [tbl(keys, [0.9*inch, 5.5*inch]), PageBreak()]

# ── Step 1 ───────────────────────────────────────────────────────────────────
s += [P("Step 1 — Save 11 workspace waypoints (~5 min)", H2),
      P("These named states are used by Scenario 4 (sequence editor runs) and by "
        "Stage 3 SAC sim training to diversify goal positions. <b>Do this before any "
        "timed scenarios.</b>", B),
      P("For each row:", B),
      P("1. Use A/D/W/S/Q/E to navigate the arm so that the curses display shows "
        "the target theta / r / z.", STEP),
      P("2. Press <b>M</b> → select <b>Save current state</b> → type the state "
        "name exactly as written → Enter.", STEP),
      P("3. Press <b>M</b> again to close the menu.", STEP),
      Spacer(1, 6)]
wp = [['Name','theta','r','z','Purpose'],
      ['seq_start','30°','152 mm','60 mm','Left edge of sweep'],
      ['seq_mid','90°','152 mm','60 mm','Center home'],
      ['seq_end','150°','152 mm','60 mm','Right edge of sweep'],
      ['seq_high','90°','152 mm','95 mm','Upper Z level'],
      ['seq_low','90°','152 mm','0 mm','Lower Z level'],
      ['seq_close','90°','80 mm','60 mm','Short reach'],
      ['seq_far','90°','220 mm','60 mm','Extended reach'],
      ['corner_tl','30°','152 mm','95 mm','Top-left corner'],
      ['corner_tr','150°','152 mm','95 mm','Top-right corner'],
      ['corner_bl','30°','152 mm','0 mm','Bottom-left corner'],
      ['corner_br','150°','152 mm','0 mm','Bottom-right corner']]
s += [tbl(wp, [1.0*inch, 0.6*inch, 0.7*inch, 0.6*inch, 3.1*inch]),
      Spacer(1, 6),
      P("When all 11 are saved, press <b>H</b> to return home. Step 1 complete.", NOTE),
      PageBreak()]

# ── Scenario 1 ───────────────────────────────────────────────────────────────
s += [P("Scenario 1 — Free sweep, no obstacles (15 min)", H2),
      P("<b>Goal:</b> Clean ToF noise floor + servo lag measurement at sweep velocity.", B),
      P("<b>Setup:</b> Remove all obstacles from the arm's workspace. Stand back at least 1 m — "
        "your body is a large ToF return.", B),
      P("Steps:", H3),
      P("1. Press <b>G</b> to start the ToF CSV log.", STEP),
      P("2. Press <b>Z</b> to start autonomous sweep.", STEP),
      P("3. <b>Walk away.</b> Leave the room if possible. Set a 15-minute timer.", STEP),
      P("4. When the timer ends, press <b>Z</b> to stop the sweep.", STEP),
      P("5. Press <b>G</b> to stop the ToF log.", STEP),
      P("<b>What success looks like:</b> The arm sweeps 0°↔180° continuously without "
        "triggering REPLAN or back-away. If you see persistent REPLAN events in the curses "
        "display, a sensor is picking up stray geometry — check that no cables, boxes, or "
        "walls are within 30 cm of the sweep arc.", B)]

# ── Scenario 2 ───────────────────────────────────────────────────────────────
s += [P("Scenario 2 — Hand obstacle, CH0 side (15 min)", H2),
      P("<b>Goal:</b> ToF partial-occlusion patterns and cell dropout rates with a soft obstacle.", B),
      P("<b>Setup:</b> CH0 is the front/side sensor. Stand on the side of the arm where CH0 faces. "
        "Start sweep (<b>Z</b>) and keep the ToF log on (<b>G</b>).", B),
      P("Four sub-scenarios — do all of them in order:", H3)]
sc2 = [['#','What you do','Time'],
       ['2a','Hold your open palm toward the arm at theta≈45°, z≈60 mm. Start at ~25 cm, '
             'slowly move to ~12 cm over 4 minutes. Expect the arm to REPLAN as you close in.','4 min'],
       ['2b','Same position but drop your hand to z≈0 mm (low height). The arm should try a '
             'Z-bypass upward when it hits you.','3 min'],
       ['2c','Raise your hand to z≈90 mm (high height). The arm should try a Z-bypass downward.','3 min'],
       ['2d','Return to z≈60 mm at r≈20 cm. Slowly walk your hand along the theta axis from '
             '45° toward 90° while the arm sweeps.','5 min']]
s += [tbl(sc2, [0.4*inch, 4.9*inch, 0.7*inch]),
      Spacer(1, 6),
      P("<b>Success check:</b> You should see varied REPLAN / back-away events in the display. "
        "If the arm never reacts to your hand, the ToF sensor is either occluded or misaligned — "
        "stop and inspect before continuing.", B),
      PageBreak()]

# ── Scenario 3 ───────────────────────────────────────────────────────────────
s += [P("Scenario 3 — Both-side obstacles (10 min)", H2),
      P("<b>Goal:</b> ToF noise when two simultaneous near-field returns are present.", B),
      P("<b>Setup:</b> Place the cylinder at theta=90°, r≈150 mm, z≈60 mm "
        "(directly in front of the arm, mid-sweep). Start sweep (<b>Z</b>).", B)]
sc3 = [['#','What you do','Time'],
       ['3a','Cylinder only, no hand. Stand back and watch. Arm should attempt a theta bypass '
             'around the cylinder.','3 min'],
       ['3b','Add your hand on the <b>opposite</b> side of the arm from the cylinder (so both '
             'CH0 and CH1 report obstacles). Arm should attempt a Z bypass since theta is blocked.','4 min'],
       ['3c','Remove your hand. Move the cylinder down to z≈30 mm (lower). Arm should go <b>over</b> '
             'the cylinder by lifting z.','3 min']]
s += [tbl(sc3, [0.4*inch, 4.9*inch, 0.7*inch]),
      Spacer(1, 6),
      P("<b>Success check:</b> You should see at least one clean Z bypass event in 3b and 3c.", B)]

# ── Scenario 4 ───────────────────────────────────────────────────────────────
s += [P("Scenario 4 — Sequence editor runs (10 min)", H2),
      P("<b>Goal:</b> Non-zero goal-delta observations (tests that the goal features in the "
        "obs vector populate correctly when a named target is active).", B),
      P("Steps:", H3),
      P("1. Stop sweep if running (<b>Z</b>).", STEP),
      P("2. Press <b>X</b> to open the sequence editor.", STEP),
      P("3. Load the file <b>collect_seq.txt</b> (or paste its contents into the editor buffer).", STEP),
      P("4. Start the sequence. Let it cycle through all 11 waypoints.", STEP),
      P("5. After cycle 1 completes, place the cylinder at theta=60°, r=150 mm, z=60 mm — "
        "this forces obstacle avoidance between <b>seq_start</b> and <b>seq_mid</b>.", STEP),
      P("6. Let it run 3–4 full cycles total (~10 min).", STEP),
      P("7. Stop the sequence, remove the cylinder.", STEP),
      P("<b>Success check:</b> Arm visits every named waypoint. If it times out on any waypoint, "
        "that state is unreachable from the previous one with the cylinder in place — "
        "move the cylinder slightly and retry.", B),
      PageBreak()]

# ── Scenario 5 ───────────────────────────────────────────────────────────────
s += [P("Scenario 5 — Manual joystick exploration (10 min)", H2),
      P("<b>Goal:</b> Observation coverage at workspace edges that autonomous sweep never visits.", B),
      P("<b>Setup:</b> Sweep must be <b>off</b>. No obstacles for the first 7 minutes.", B),
      P("Drive the arm through each area below. Spend ~1.5 min per row:", H3)]
sc5 = [['#','Area','How'],
       ['5a','Theta extremes','Press <b>A</b> repeatedly to reach 5°, then <b>D</b> repeatedly to reach 175°.'],
       ['5b','r extremes','Press <b>S</b> to reach 80 mm, then <b>W</b> to reach 220 mm.'],
       ['5c','z extremes','Press <b>Q</b> to reach 95 mm, then <b>E</b> to reach 0 mm.'],
       ['5d','All 4 workspace corners','Combine theta + z extremes: visit (30°,95mm), (150°,95mm), (150°,0mm), (30°,0mm).'],
       ['5e','With obstacle','Place cylinder at r≈15 cm. Slowly drive arm toward it. Stop when IR fires or ToF shows <150 mm.']]
s += [tbl(sc5, [0.4*inch, 1.7*inch, 3.9*inch]),
      Spacer(1, 6),
      P("<b>Success check:</b> The curses display shows theta/r/z values across the full range "
        "listed. If the arm refuses to reach an extreme, you've hit a kinematic limit — that's "
        "fine, just note which one.", B)]

# ── Scenario 6 ───────────────────────────────────────────────────────────────
s += [P("Scenario 6 — Controlled near-misses (10 min)", H2),
      P("<b>Goal:</b> Close-range ToF distributions for the proximity-shaping reward terms.", B),
      P("<b>Setup:</b> Place the cylinder at theta=90°, r=150 mm, z=60 mm. Start sweep (<b>Z</b>).", B)]
sc6 = [['#','Cylinder distance from arm path','What to expect','Time'],
       ['6a','≈220 mm (just inside the ToF threshold)','Soft REPLAN — arm curves around smoothly.','2 min'],
       ['6b','≈120 mm (well inside threshold)','Hard REPLAN — arm may back away, try Z bypass.','2 min'],
       ['6c','≈80 mm (very close)','IR sensor may fire; estop engages briefly.','2 min'],
       ['6d','Slowly walk cylinder from 300 mm to 80 mm over 4 minutes','Watch the proximity response '
        'ramp up — this is the most valuable data in the whole session.','4 min']]
s += [tbl(sc6, [0.4*inch, 2.2*inch, 3.0*inch, 0.6*inch]),
      Spacer(1, 6),
      P("<b>Success check:</b> You should see at least one IR trigger in 6c or 6d. If IR never "
        "fires even at 80 mm, the IR sensors are misaligned or wired wrong — stop and check.", B),
      PageBreak()]

# ── After collection ─────────────────────────────────────────────────────────
s += [P("After collection — verification and next steps", H2),
      P("Steps:", H3),
      P("1. Press <b>ESC</b> to quit the controller. This flushes all buffered NPZ data to disk.", STEP),
      P("2. Verify the files exist:", STEP),
      P("ls -lh logs/rl_transitions_*.npz", CODE),
      P("Expect at least 3 files totalling ≥ 20 MB.", STEP),
      P("3. Run the noise calibration script:", STEP),
      P("cd braccio_main_runner && python calibrate_noise.py", CODE),
      P("This writes <b>braccio_ctrl/sim/noise_params.json</b> with <b>\"calibrated\": true</b>.", STEP),
      Spacer(1, 4),
      P("Sanity-check the calibration output:", H3)]
cal = [['Parameter','Expected range','If outside range'],
       ['servo_lag_factor','0.75 – 1.25','Arm serial latency is unusual; check USB cable.'],
       ['tof_noise_sigma_mm','5 – 20','ToF sensor unusually noisy or clean; check mounts.'],
       ['cell_dropout_rate','0.05 – 0.30','Sensor mount alignment off; inspect CH0/CH1 orientation.']]
s += [tbl(cal, [1.6*inch, 1.4*inch, 3.4*inch]),
      Spacer(1, 8),
      P("You are now ready for Stage 2 (<b>train_bc.py</b>) and Stage 3 (<b>train_rl.py</b>).", B)]

# ── Troubleshooting ──────────────────────────────────────────────────────────
s += [P("Troubleshooting", H2)]
tr = [['Symptom','Likely cause / fix'],
      ['No NPZ files in logs/','Controller crashed before ESC. Check stderr; re-run and quit with ESC.'],
      ['Arm never triggers REPLAN in scenarios 2/3','ToF sensor misaligned — confirm CH0/CH1 faces sweep direction.'],
      ['IR never fires in 6c','IR sensors wired wrong or unplugged — check Teensy pins 2–5.'],
      ['Curses display shows all zeros for ToF','Teensy not connected or wrong port — check /dev/ttyACM1.'],
      ['Arm stutters or makes jerky moves','DELTA value too high — it will stabilise during sweep; not a data issue.'],
      ['calibrate_noise.py exits with "No valid transitions"','NPZ files too short (<1000 transitions). Run longer scenarios.']]
s += [tbl(tr, [2.4*inch, 4.0*inch])]

# ── Build ────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.75*inch,
                        rightMargin=0.75*inch, topMargin=0.7*inch,
                        bottomMargin=0.7*inch, title="Data Collection Guide")
doc.build(s)
print(f"Wrote {OUT} ({len(s)} flowables)")
