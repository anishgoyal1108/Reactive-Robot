#include <BraccioV2.h>

/*
  braccio_joint_test.ino

  Mega-side actuator interface with:
  - Legacy joint commands
  - v1 CMD packet support
  - MCU-side IK via: SET IKP <theta_deg> <r_mm> <z_mm> <wrist_offset_deg> <wr> <g>
  - Safety watchdog + heartbeat
*/

Braccio arm;

#define JOINT_BASE        0
#define JOINT_SHOULDER    1
#define JOINT_ELBOW       2
#define JOINT_WRIST_VERT  3
#define JOINT_WRIST_ROT   4
#define JOINT_GRIPPER     5

const int HOME_POS[6] = {90, 90, 90, 90, 90, 73};
const int JOINT_MIN[6] = {0, 15, 0, 0, 0, 10};
const int JOINT_MAX[6] = {180, 165, 180, 180, 180, 73};
const char* JOINT_TOKEN[6] = {"B", "S", "E", "WV", "WR", "G"};

int _pos[6] = {90, 90, 90, 90, 90, 73};
int _prevPos[6] = {90, 90, 90, 90, 90, 73};

unsigned long _lastTick = 0;
const unsigned long TICK_MS = 10;

const unsigned long CMD_TIMEOUT_MS = 750;
const unsigned long STAT_INTERVAL_MS = 250;
unsigned long lastCmdMs = 0;
unsigned long lastStatMs = 0;
uint32_t lastSeq = 0;
String currentMode = "nominal_tracking";

void setPos(int joint, int deg) {
  _prevPos[joint] = _pos[joint];
  _pos[joint] = constrain(deg, JOINT_MIN[joint], JOINT_MAX[joint]);
  arm.setOneAbsolute(joint, _pos[joint]);
}

void setAllPos(int b, int s, int e, int w, int wr, int g) {
  int t[6] = {b, s, e, w, wr, g};
  for (int j = 0; j < 6; j++) {
    _prevPos[j] = _pos[j];
    _pos[j] = constrain(t[j], JOINT_MIN[j], JOINT_MAX[j]);
  }
  arm.setAllAbsolute(_pos[0], _pos[1], _pos[2], _pos[3], _pos[4], _pos[5]);
}

bool solveIK(float x_mm, float y_mm, float z_mm,
             int &base_deg, int &shoulder_deg,
             int &elbow_deg, int &wristV_deg) {

  const float L1 = 125.0f;
  const float L2 = 125.0f;
  const float L3 = 60.0f;

  base_deg = (int)degrees(atan2(y_mm, x_mm));
  base_deg = constrain(base_deg, JOINT_MIN[0], JOINT_MAX[0]);

  float r = sqrt(x_mm * x_mm + y_mm * y_mm) - L3;
  float h = z_mm;
  if (r < 0.0f) r = 0.0f;

  float dist2 = r * r + h * h;
  float dist = sqrt(dist2);

  if (dist < 1e-3f) return false;
  if (dist > (L1 + L2)) return false;
  if (dist < fabs(L1 - L2)) return false;

  float cos_elbow = (dist2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
  cos_elbow = constrain(cos_elbow, -1.0f, 1.0f);
  elbow_deg = (int)(180.0f - degrees(acos(cos_elbow)));

  float alpha = atan2(h, r);
  float cos_beta = constrain((dist2 + L1 * L1 - L2 * L2) / (2.0f * dist * L1), -1.0f, 1.0f);
  shoulder_deg = (int)degrees(alpha + acos(cos_beta));
  shoulder_deg = constrain(shoulder_deg, JOINT_MIN[1], JOINT_MAX[1]);

  wristV_deg = constrain(180 - shoulder_deg - elbow_deg, JOINT_MIN[3], JOINT_MAX[3]);
  return true;
}

bool setPoseFromPolar(float theta_deg, float r_mm, float z_mm,
                      float wrist_offset_deg, int wrist_rot_deg, int gripper_deg) {
  float theta_rad = radians(theta_deg);
  float x_mm = r_mm * cos(theta_rad);
  float y_mm = r_mm * sin(theta_rad);

  int b, s, e, wv;
  if (!solveIK(x_mm, y_mm, z_mm, b, s, e, wv)) {
    return false;
  }

  int wv_adj = constrain((int)round((float)wv + wrist_offset_deg), JOINT_MIN[3], JOINT_MAX[3]);
  int wr = constrain(wrist_rot_deg, JOINT_MIN[4], JOINT_MAX[4]);
  int g = constrain(gripper_deg, JOINT_MIN[5], JOINT_MAX[5]);

  setAllPos(b, s, e, wv_adj, wr, g);
  return true;
}

void holdCurrentPosition() {
  setAllPos(_pos[0], _pos[1], _pos[2], _pos[3], _pos[4], _pos[5]);
}

void sendAck(uint32_t seq, const char* status) {
  Serial.print("ACK,");
  Serial.print(seq);
  Serial.print(",");
  Serial.print(status);
  Serial.print(",");
  Serial.println(millis());
}

void sendStat() {
  Serial.print("STAT,");
  Serial.print(currentMode);
  Serial.print(",");
  Serial.print(lastSeq);
  Serial.print(",");
  Serial.println(millis() - lastCmdMs);
}

bool parseCmdV1(const String& cmd) {
  if (!cmd.startsWith("CMD,")) return false;

  String fields[11];
  int count = 0;
  int start = 0;

  for (int i = 0; i <= cmd.length(); i++) {
    bool at_end = (i == cmd.length());
    if (at_end || cmd.charAt(i) == ',') {
      if (count < 11) {
        fields[count++] = cmd.substring(start, i);
      }
      start = i + 1;
    }
  }

  if (count != 11 || fields[0] != "CMD") {
    sendAck(lastSeq, "ERR_FORMAT");
    return true;
  }

  uint32_t seq = (uint32_t)fields[1].toInt();
  String mode = fields[3];

  int vals[6];
  for (int j = 0; j < 6; j++) {
    vals[j] = fields[5 + j].toInt();
    if (vals[j] < JOINT_MIN[j] || vals[j] > JOINT_MAX[j]) {
      sendAck(seq, "ERR_RANGE");
      return true;
    }
  }

  setAllPos(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
  currentMode = mode;
  lastSeq = seq;
  lastCmdMs = millis();
  sendAck(seq, "OK");
  return true;
}

void handleLegacy(String cmd) {
  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  if (cmd == "HOME") {
    setAllPos(HOME_POS[0], HOME_POS[1], HOME_POS[2], HOME_POS[3], HOME_POS[4], HOME_POS[5]);
    currentMode = "nominal_tracking";
    lastCmdMs = millis();
    Serial.println("OK HOME");
    return;
  }

  if (cmd == "GET POS") {
    Serial.print("POS=");
    for (int j = 0; j < 6; j++) {
      Serial.print(_pos[j]);
      if (j < 5) Serial.print(',');
    }
    Serial.println();
    return;
  }

  if (cmd.startsWith("SET DELTA ")) {
    int val = cmd.substring(10).toInt();
    val = constrain(val, 1, 5);
    for (int j = 0; j < 6; j++) arm.setDelta(j, val);
    lastCmdMs = millis();
    Serial.print("OK DELTA=");
    Serial.println(val);
    return;
  }

  if (cmd.startsWith("SET IKP ")) {
    String rest = cmd.substring(8);
    rest.trim();

    float valsF[4] = {0, 0, 0, 0};
    int valsI[2] = {90, 73};

    for (int k = 0; k < 4; k++) {
      rest.trim();
      int sp = rest.indexOf(' ');
      if (sp < 0) {
        Serial.println("ERR IK FORMAT");
        return;
      }
      valsF[k] = rest.substring(0, sp).toFloat();
      rest = rest.substring(sp + 1);
    }

    for (int k = 0; k < 2; k++) {
      rest.trim();
      int sp = rest.indexOf(' ');
      String tok = (sp >= 0) ? rest.substring(0, sp) : rest;
      valsI[k] = tok.toInt();
      if (sp >= 0) rest = rest.substring(sp + 1);
    }

    bool ok = setPoseFromPolar(valsF[0], valsF[1], valsF[2], valsF[3], valsI[0], valsI[1]);
    if (!ok) {
      Serial.println("ERR IK UNREACH");
      return;
    }

    currentMode = "nominal_tracking";
    lastCmdMs = millis();
    Serial.print("OK IK=");
    for (int j = 0; j < 6; j++) {
      Serial.print(_pos[j]);
      if (j < 5) Serial.print(',');
    }
    Serial.println();
    return;
  }

  if (cmd.startsWith("SET ALL ")) {
    int vals[6];
    String rest = cmd.substring(8);
    rest.trim();
    for (int j = 0; j < 6; j++) {
      rest.trim();
      int sp = rest.indexOf(' ');
      String tok = (sp >= 0) ? rest.substring(0, sp) : rest;
      vals[j] = tok.toInt();
      if (sp >= 0) rest = rest.substring(sp + 1);
      if (vals[j] < JOINT_MIN[j] || vals[j] > JOINT_MAX[j]) {
        Serial.println("ERR RANGE");
        return;
      }
    }
    setAllPos(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
    currentMode = "nominal_tracking";
    lastCmdMs = millis();
    Serial.print("OK ALL=");
    for (int j = 0; j < 6; j++) {
      Serial.print(vals[j]);
      if (j < 5) Serial.print(',');
    }
    Serial.println();
    return;
  }

  if (cmd.startsWith("SET ")) {
    String rest = cmd.substring(4);
    int sp = rest.indexOf(' ');
    if (sp < 0) {
      Serial.println("ERR CMD");
      return;
    }
    String token = rest.substring(0, sp);
    int deg = rest.substring(sp + 1).toInt();
    for (int j = 0; j < 6; j++) {
      if (token == JOINT_TOKEN[j]) {
        if (deg < JOINT_MIN[j] || deg > JOINT_MAX[j]) {
          Serial.println("ERR RANGE");
          return;
        }
        setPos(j, deg);
        currentMode = "nominal_tracking";
        lastCmdMs = millis();
        Serial.print("OK J=");
        Serial.print(j);
        Serial.print(',');
        Serial.println(deg);
        return;
      }
    }
  }

  Serial.println("ERR CMD");
}

void setup() {
  Serial.begin(115200);
  arm.begin();

  for (int j = 0; j < 6; j++) arm.setDelta(j, 1);

  arm.safeDelay(2000);
  setAllPos(HOME_POS[0], HOME_POS[1], HOME_POS[2], HOME_POS[3], HOME_POS[4], HOME_POS[5]);
  arm.safeDelay(500);

  lastCmdMs = millis();
  lastStatMs = millis();

  Serial.println("READY");
}

void loop() {
  unsigned long now = millis();
  if (now - _lastTick >= TICK_MS) {
    _lastTick = now;
    arm.update();
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      if (!parseCmdV1(cmd)) {
        handleLegacy(cmd);
      }
    }
  }

  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    holdCurrentPosition();
    currentMode = "comms_fault";
  }

  if (now - lastStatMs >= STAT_INTERVAL_MS) {
    lastStatMs = now;
    sendStat();
  }
}
