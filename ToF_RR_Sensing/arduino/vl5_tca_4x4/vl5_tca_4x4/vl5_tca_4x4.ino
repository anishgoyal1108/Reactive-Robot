/*
  Teensy Hyperion ToF + IMU streamer

  TF,<seq>,<mcu_ms>,<sensor_id>,<mux_ch>,<joint_id>,<status>,<rows>,<cols>,<d...>,<v...>
  IMU,<seq>,<mcu_ms>,<ax_g>,<ay_g>,<az_g>,<gx_dps>,<gy_dps>,<gz_dps>,<temp_c>,<status>
  IR,<bits>
  MODE,<mode>
  CFG,ACT,<count>
  CFG,GRID,<rows>,<cols>
  CFG,TARGET_HZ,<hz>
*/

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

#define TCA_ADDR 0x70
#define VL5_ADDR 0x29
#define MPU_ADDR 0x68

#define TARGET_HZ 60
#define RES_ROWS 4
#define RES_COLS 4
#define RES_ZONES (RES_ROWS * RES_COLS)
#define NUM_TOF_SENSORS 4

#define IMU_SEND_INTERVAL_MS 10

SparkFun_VL53L5CX imager[NUM_TOF_SENSORS];
VL53L5CX_ResultsData results[NUM_TOF_SENSORS];
bool tofPresent[NUM_TOF_SENSORS] = {false, false, false, false};

const uint8_t TOF_MUX_CH[NUM_TOF_SENSORS] = {0, 1, 2, 3};

uint32_t tfSeqCounter = 0;
uint32_t imuSeqCounter = 0;

const int IR_PIN_BIT0 = 2;
const int IR_PIN_BIT1 = 3;
const unsigned long IR_SEND_INTERVAL_MS = 100;
unsigned long lastIRSend = 0;
unsigned long lastIMUSend = 0;

bool imuPresent = false;
bool imuInitOk = false;

enum Mode { MODE_MUX, MODE_CH0, MODE_CH1, MODE_CH2, MODE_CH3 };
volatile Mode mode = MODE_MUX;
String cmdBuf;

// Default to the legacy dual-sensor behavior; host enables Hyperion with ACT 4.
uint8_t activeChannelCount = 2;
uint8_t rrCursor = 0;

static inline void tcaSelect(uint8_t ch) {
  if (ch > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

static inline bool i2cProbe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

static inline bool modeAllowsToFIndex(uint8_t idx) {
  if (idx >= NUM_TOF_SENSORS) return false;
  uint8_t ch = TOF_MUX_CH[idx];

  if (mode == MODE_MUX) {
    return (idx < activeChannelCount);
  }
  if (mode == MODE_CH0) return (ch == 0);
  if (mode == MODE_CH1) return (ch == 1);
  if (mode == MODE_CH2) return (ch == 2);
  if (mode == MODE_CH3) return (ch == 3);
  return false;
}

static inline void announceMode() {
  Serial.print("MODE,");
  if (mode == MODE_CH0) Serial.println("CH0");
  else if (mode == MODE_CH1) Serial.println("CH1");
  else if (mode == MODE_CH2) Serial.println("CH2");
  else if (mode == MODE_CH3) Serial.println("CH3");
  else Serial.println("MUX");
}

static inline void announceActiveCount() {
  Serial.print("CFG,ACT,");
  Serial.println(activeChannelCount);
}

static inline void announceGrid() {
  Serial.print("CFG,GRID,");
  Serial.print(RES_ROWS);
  Serial.print(",");
  Serial.println(RES_COLS);
}

static inline void announceRate() {
  Serial.print("CFG,TARGET_HZ,");
  Serial.println(TARGET_HZ);
}

static inline void printHelp() {
  Serial.println("Commands: CH0 | CH1 | CH2 | CH3 | MUX | ACT <1..4> | HELP");
}

void handleSerialCommands() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      cmdBuf.trim();
      cmdBuf.toUpperCase();

      if (cmdBuf == "CH0") mode = MODE_CH0;
      else if (cmdBuf == "CH1") mode = MODE_CH1;
      else if (cmdBuf == "CH2") mode = MODE_CH2;
      else if (cmdBuf == "CH3") mode = MODE_CH3;
      else if (cmdBuf == "MUX") mode = MODE_MUX;
      else if (cmdBuf.startsWith("ACT ")) {
        int n = cmdBuf.substring(4).toInt();
        if (n < 1) n = 1;
        if (n > NUM_TOF_SENSORS) n = NUM_TOF_SENSORS;
        activeChannelCount = (uint8_t)n;
        announceActiveCount();
      }
      else if (cmdBuf == "HELP" || cmdBuf == "?") printHelp();
      else if (cmdBuf.length() > 0) {
        Serial.print("ERR,UnknownCommand,");
        Serial.println(cmdBuf);
      }

      if (cmdBuf.length() > 0) announceMode();
      cmdBuf = "";
    } else {
      if (cmdBuf.length() < 48) cmdBuf += c;
    }
  }
}

bool initOneToF(uint8_t idx) {
  if (idx >= NUM_TOF_SENSORS) return false;
  uint8_t ch = TOF_MUX_CH[idx];

  tcaSelect(ch);

  if (!i2cProbe(VL5_ADDR)) {
    Serial.print("WARN,CH");
    Serial.print(ch);
    Serial.println(",NoDeviceAt0x29");
    return false;
  }

  if (!imager[idx].begin()) {
    Serial.print("ERR,CH");
    Serial.print(ch);
    Serial.println(",BeginFailed");
    return false;
  }

  imager[idx].setResolution(RES_ZONES);
  imager[idx].setRangingFrequency(TARGET_HZ);
  imager[idx].startRanging();

  Serial.print("INFO,CH");
  Serial.print(ch);
  Serial.println(",Ready");
  return true;
}

void streamToFFrameV1(uint8_t idx) {
  if (idx >= NUM_TOF_SENSORS) return;
  if (!tofPresent[idx]) return;

  uint8_t ch = TOF_MUX_CH[idx];

  tcaSelect(ch);
  if (!imager[idx].isDataReady()) return;
  if (!imager[idx].getRangingData(&results[idx])) {
    Serial.print("ERR,CH");
    Serial.print(ch);
    Serial.println(",GetRangingDataFailed");
    return;
  }

  uint8_t status = 0;
  int d[RES_ZONES];
  int v[RES_ZONES];

  for (int i = 0; i < RES_ZONES; i++) {
    int mm = (int)results[idx].distance_mm[i];
    bool valid = (mm > 40 && mm < 3000);
    d[i] = mm;
    v[i] = valid ? 1 : 0;
    if (!valid) status |= 0x01;
  }

  Serial.print("TF,");
  Serial.print(tfSeqCounter++);
  Serial.print(",");
  Serial.print(millis());
  Serial.print(",S");
  Serial.print(ch);
  Serial.print(",");
  Serial.print(ch);
  Serial.print(",wrist_pitch,");
  Serial.print(status);
  Serial.print(",");
  Serial.print(RES_ROWS);
  Serial.print(",");
  Serial.print(RES_COLS);

  for (int i = 0; i < RES_ZONES; i++) {
    Serial.print(",");
    Serial.print(d[i]);
  }
  for (int i = 0; i < RES_ZONES; i++) {
    Serial.print(",");
    Serial.print(v[i]);
  }
  Serial.println();
}

uint8_t nextRoundRobinToFIndex() {
  for (int n = 0; n < NUM_TOF_SENSORS; n++) {
    rrCursor = (uint8_t)((rrCursor + 1) % NUM_TOF_SENSORS);
    uint8_t idx = rrCursor;
    if (!modeAllowsToFIndex(idx)) continue;
    if (!tofPresent[idx]) continue;
    return idx;
  }
  return 255;
}

bool mpuWriteByte(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

bool mpuReadBytes(uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t got = Wire.requestFrom((int)MPU_ADDR, (int)len, (int)true);
  if (got != len) return false;

  for (uint8_t i = 0; i < len; i++) {
    if (!Wire.available()) return false;
    buf[i] = Wire.read();
  }
  return true;
}

bool initIMU() {
  if (!i2cProbe(MPU_ADDR)) {
    Serial.println("WARN,IMU,MPU6050_NotFound");
    return false;
  }

  if (!mpuWriteByte(0x6B, 0x00)) {
    Serial.println("ERR,IMU,MPU6050_WakeFailed");
    return false;
  }

  Serial.println("INFO,IMU,MPU6050_Ready");
  return true;
}

void streamIMUFrame() {
  unsigned long now = millis();
  if (now - lastIMUSend < IMU_SEND_INTERVAL_MS) return;
  lastIMUSend = now;

  uint8_t status = 0;
  float ax_g = 0.0f, ay_g = 0.0f, az_g = 0.0f;
  float gx_dps = 0.0f, gy_dps = 0.0f, gz_dps = 0.0f;
  float temp_c = 0.0f;

  if (!imuPresent) status |= 0x01;
  if (!imuInitOk) status |= 0x02;

  if (imuPresent && imuInitOk) {
    uint8_t raw[14];
    bool ok = mpuReadBytes(0x3B, raw, 14);
    if (!ok) {
      status |= 0x04;
    } else {
      int16_t ax = (int16_t)((raw[0] << 8) | raw[1]);
      int16_t ay = (int16_t)((raw[2] << 8) | raw[3]);
      int16_t az = (int16_t)((raw[4] << 8) | raw[5]);
      int16_t temp = (int16_t)((raw[6] << 8) | raw[7]);
      int16_t gx = (int16_t)((raw[8] << 8) | raw[9]);
      int16_t gy = (int16_t)((raw[10] << 8) | raw[11]);
      int16_t gz = (int16_t)((raw[12] << 8) | raw[13]);

      ax_g = (float)ax / 16384.0f;
      ay_g = (float)ay / 16384.0f;
      az_g = (float)az / 16384.0f;
      gx_dps = (float)gx / 131.0f;
      gy_dps = (float)gy / 131.0f;
      gz_dps = (float)gz / 131.0f;
      temp_c = ((float)temp / 340.0f) + 36.53f;
    }
  }

  Serial.print("IMU,");
  Serial.print(imuSeqCounter++);
  Serial.print(",");
  Serial.print(now);
  Serial.print(",");
  Serial.print(ax_g, 5);
  Serial.print(",");
  Serial.print(ay_g, 5);
  Serial.print(",");
  Serial.print(az_g, 5);
  Serial.print(",");
  Serial.print(gx_dps, 5);
  Serial.print(",");
  Serial.print(gy_dps, 5);
  Serial.print(",");
  Serial.print(gz_dps, 5);
  Serial.print(",");
  Serial.print(temp_c, 3);
  Serial.print(",");
  Serial.println(status);
}

void readAndSendIR() {
  unsigned long now = millis();
  if (now - lastIRSend < IR_SEND_INTERVAL_MS) return;
  lastIRSend = now;

  uint8_t bit0 = digitalRead(IR_PIN_BIT0) ? 1 : 0;
  uint8_t bit1 = digitalRead(IR_PIN_BIT1) ? 1 : 0;
  uint8_t ir_val = (bit1 << 1) | bit0;

  Serial.print("IR,");
  Serial.println(ir_val);
}

void setup() {
  Serial.begin(115200);
  delay(300);

  Wire.begin();
  Wire.setClock(1000000);

  pinMode(IR_PIN_BIT0, INPUT);
  pinMode(IR_PIN_BIT1, INPUT);

  printHelp();
  announceMode();
  announceActiveCount();
  announceGrid();
  announceRate();

  if (!i2cProbe(TCA_ADDR)) Serial.println("ERR,TCA,NotFoundAt0x70");
  else Serial.println("INFO,TCA,Found");

  for (uint8_t idx = 0; idx < NUM_TOF_SENSORS; idx++) {
    tofPresent[idx] = initOneToF(idx);
  }

  imuPresent = true;
  imuInitOk = initIMU();
  if (!imuInitOk) {
    imuPresent = false;
  }

  Serial.println("INFO,Streaming,Started");
}

void loop() {
  handleSerialCommands();

  uint8_t idx = nextRoundRobinToFIndex();
  if (idx != 255) {
    streamToFFrameV1(idx);
  }

  streamIMUFrame();
  readAndSendIR();
  delay(1);
}
