#include <BraccioV2.h>

/**
 * braccio_joint_test.ino
 *
 * Serial command server for the Tinkerkit Braccio arm (BraccioV2 library).
 * Replaces the fixed demo loop with a non-blocking command receiver.
 *
 * Braccio Joint Map (servo indices 0–5):
 *   0 - Base       (M1): rotation around vertical axis,  0–180°
 *   1 - Shoulder   (M2): forward/back tilt,              15–165°
 *   2 - Elbow      (M3): up/down,                        0–180°
 *   3 - Wrist Vert (M4): pitch,                          0–180°
 *   4 - Wrist Rot  (M5): roll,                           0–180°
 *   5 - Gripper    (M6): open/close,                     10–73°
 *
 * Serial Protocol (115200 baud, \n-terminated):
 *
 *   Command         Response
 *   ──────────────────────────────────────────────
 *   PING            PONG
 *   HOME            OK HOME
 *   GET POS         POS=90,90,90,90,90,73
 *   SET B 45        OK J=0,45
 *   SET S 90        OK J=1,90
 *   SET E 90        OK J=2,90
 *   SET WV 90       OK J=3,90
 *   SET WR 90       OK J=4,90
 *   SET G 10        OK J=5,10
 *   SET ALL b s e wv wr g    OK ALL=b,s,e,wv,wr,g
 *   SET DELTA val   OK DELTA=val   (1–5 deg/tick)
 *   Out of range    ERR RANGE <token> <val>
 *   Unknown         ERR CMD
 *
 * IK and demo functions are kept below for reference but are no longer
 * called from loop().
 */

// ── Instantiate the arm ────────────────────────────────────────────────────
Braccio arm;

// ── Named joint indices ────────────────────────────────────────────────────
#define JOINT_BASE        0
#define JOINT_SHOULDER    1
#define JOINT_ELBOW       2
#define JOINT_WRIST_VERT  3
#define JOINT_WRIST_ROT   4
#define JOINT_GRIPPER     5

// ── Safe home position (arm pointing straight up) ─────────────────────────
const int HOME_POS[6] = {90, 90, 90, 90, 90, 73};

// ── Speed: degrees moved per update() tick (lower = smoother) ─────────────
#define DEFAULT_DELTA  1   // ~1°/10 ms  →  full 180° sweep takes ~1.8 s

// ── Position shadow arrays ─────────────────────────────────────────────────
int _pos[6]     = {90, 90, 90, 90, 90, 73};
int _prevPos[6] = {90, 90, 90, 90, 90, 73};

// ── Joint limits and protocol tokens ──────────────────────────────────────
const int  JOINT_MIN[6]    = {  0, 15,  0,  0,  0, 10 };
const int  JOINT_MAX[6]    = {180,165,180,180,180, 73 };
const char* JOINT_TOKEN[6] = {"B","S","E","WV","WR","G"};

// ── Non-blocking tick scheduler ────────────────────────────────────────────
unsigned long _lastTick = 0;
const unsigned long TICK_MS = 10;  // 100 Hz servo update rate

// ── Forward declarations ───────────────────────────────────────────────────
void setPos(int joint, int deg);
void setRelPos(int joint, int delta_deg);
void setAllPos(int b, int s, int e, int w, int wr, int g);
void goHome();
void printJointAngles();
void handleCommand(String cmd);
bool solveIK(float x_mm, float y_mm, float z_mm,
             int &base_deg, int &shoulder_deg,
             int &elbow_deg, int &wristV_deg);

// ══════════════════════════════════════════════════════════════════════════
// ── Position helpers ───────────────────────────────────────────────────────

void setPos(int joint, int deg) {
  _prevPos[joint] = _pos[joint];
  _pos[joint]     = constrain(deg, 0, 180);
  arm.setOneAbsolute(joint, _pos[joint]);
}

void setRelPos(int joint, int delta_deg) {
  _prevPos[joint] = _pos[joint];
  _pos[joint]     = constrain(_pos[joint] + delta_deg, 0, 180);
  arm.setOneRelative(joint, delta_deg);
}

void setAllPos(int b, int s, int e, int w, int wr, int g) {
  int t[6] = {b, s, e, w, wr, g};
  for (int j = 0; j < 6; j++) {
    _prevPos[j] = _pos[j];
    _pos[j]     = t[j];
  }
  arm.setAllAbsolute(b, s, e, w, wr, g);
}

// ══════════════════════════════════════════════════════════════════════════
// ── Setup ──────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  arm.begin();

  for (int j = 0; j < 6; j++) {
    arm.setDelta(j, DEFAULT_DELTA);
  }

  // Blocking wait only at init — acceptable here
  arm.safeDelay(2000);

  // Move to home position
  setAllPos(HOME_POS[0], HOME_POS[1], HOME_POS[2],
            HOME_POS[3], HOME_POS[4], HOME_POS[5]);
  arm.safeDelay(2000);

  Serial.println(F("READY"));
}

// ══════════════════════════════════════════════════════════════════════════
// ── Main loop: non-blocking tick + serial command reader ──────────────────

void loop() {
  // 1. Advance servos at 100 Hz (non-blocking)
  unsigned long now = millis();
  if (now - _lastTick >= TICK_MS) {
    _lastTick = now;
    arm.update();
  }

  // 2. Process one command if available (non-blocking)
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      handleCommand(cmd);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
// ── Command parser ─────────────────────────────────────────────────────────

void handleCommand(String cmd) {

  // ── PING ──────────────────────────────────────────────────────────────
  if (cmd == F("PING")) {
    Serial.println(F("PONG"));
    return;
  }

  // ── HOME ──────────────────────────────────────────────────────────────
  if (cmd == F("HOME")) {
    setAllPos(HOME_POS[0], HOME_POS[1], HOME_POS[2],
              HOME_POS[3], HOME_POS[4], HOME_POS[5]);
    Serial.println(F("OK HOME"));
    return;
  }

  // ── GET POS ───────────────────────────────────────────────────────────
  if (cmd == F("GET POS")) {
    Serial.print(F("POS="));
    for (int j = 0; j < 6; j++) {
      Serial.print(_pos[j]);
      if (j < 5) Serial.print(',');
    }
    Serial.println();
    return;
  }

  // ── SET DELTA <val> ───────────────────────────────────────────────────
  if (cmd.startsWith(F("SET DELTA "))) {
    int val = cmd.substring(10).toInt();
    val = constrain(val, 1, 5);
    for (int j = 0; j < 6; j++) {
      arm.setDelta(j, val);
    }
    Serial.print(F("OK DELTA="));
    Serial.println(val);
    return;
  }

  // ── SET ALL <b> <s> <e> <wv> <wr> <g> ────────────────────────────────
  if (cmd.startsWith(F("SET ALL "))) {
    int vals[6];
    String rest = cmd.substring(8);
    rest.trim();
    for (int j = 0; j < 6; j++) {
      rest.trim();
      int sp = rest.indexOf(' ');
      String tok = (sp >= 0) ? rest.substring(0, sp) : rest;
      vals[j] = constrain(tok.toInt(), JOINT_MIN[j], JOINT_MAX[j]);
      if (sp >= 0) rest = rest.substring(sp + 1);
    }
    setAllPos(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
    Serial.print(F("OK ALL="));
    for (int j = 0; j < 6; j++) {
      Serial.print(vals[j]);
      if (j < 5) Serial.print(',');
    }
    Serial.println();
    return;
  }

  // ── SET <token> <deg> ─────────────────────────────────────────────────
  if (cmd.startsWith(F("SET "))) {
    String rest = cmd.substring(4);
    int sp = rest.indexOf(' ');
    if (sp < 0) { Serial.println(F("ERR CMD")); return; }
    String token = rest.substring(0, sp);
    int deg = rest.substring(sp + 1).toInt();
    for (int j = 0; j < 6; j++) {
      if (token == JOINT_TOKEN[j]) {
        if (deg < JOINT_MIN[j] || deg > JOINT_MAX[j]) {
          Serial.print(F("ERR RANGE "));
          Serial.print(JOINT_TOKEN[j]);
          Serial.print(' ');
          Serial.println(deg);
          return;
        }
        setPos(j, deg);
        Serial.print(F("OK J="));
        Serial.print(j);
        Serial.print(',');
        Serial.println(deg);
        return;
      }
    }
    Serial.println(F("ERR CMD"));
    return;
  }

  // ── Unknown command ───────────────────────────────────────────────────
  Serial.println(F("ERR CMD"));
}

// ══════════════════════════════════════════════════════════════════════════
// ── Utility helpers ────────────────────────────────────────────────────────

void goHome() {
  setAllPos(HOME_POS[0], HOME_POS[1], HOME_POS[2],
            HOME_POS[3], HOME_POS[4], HOME_POS[5]);
  printJointAngles();
}

void printJointAngles() {
  Serial.print(F("  Joints [B S E Wv Wr G]: "));
  for (int j = 0; j < 6; j++) {
    Serial.print(_pos[j]);
    Serial.print(j < 5 ? F("  ") : F("\n"));
  }
}

// ══════════════════════════════════════════════════════════════════════════
// ── IK solver (reference — not called from loop) ───────────────────────────
/**
 * 2-D geometric IK for the Braccio planar arm.
 *
 * Link lengths (measure your build and update):
 *   L1 = shoulder pivot → elbow pivot  ≈ 125 mm
 *   L2 = elbow pivot    → wrist pivot  ≈ 125 mm
 *   L3 = wrist pivot    → gripper tip  ≈  60 mm
 */
bool solveIK(float x_mm, float y_mm, float z_mm,
             int &base_deg, int &shoulder_deg,
             int &elbow_deg, int &wristV_deg) {

  const float L1 = 125.0f;
  const float L2 = 125.0f;
  const float L3 =  60.0f;

  base_deg = (int)degrees(atan2(y_mm, x_mm));
  base_deg = constrain(base_deg, 0, 180);

  float r = sqrt(x_mm*x_mm + y_mm*y_mm) - L3;
  float h = z_mm;
  if (r < 0.0f) r = 0.0f;

  float dist2 = r*r + h*h;
  float dist  = sqrt(dist2);

  if (dist < 1e-3f) return false;
  if (dist > (L1 + L2)) return false;
  if (dist < fabs(L1 - L2)) return false;

  float cos_elbow = (dist2 - L1*L1 - L2*L2) / (2.0f * L1 * L2);
  cos_elbow = constrain(cos_elbow, -1.0f, 1.0f);
  elbow_deg = (int)(180.0f - degrees(acos(cos_elbow)));

  float alpha    = atan2(h, r);
  float cos_beta = constrain((dist2 + L1*L1 - L2*L2) / (2.0f * dist * L1),
                             -1.0f, 1.0f);
  shoulder_deg = (int)degrees(alpha + acos(cos_beta));
  shoulder_deg = constrain(shoulder_deg, 15, 165);

  wristV_deg = constrain(180 - shoulder_deg - elbow_deg, 0, 180);

  return true;
}
