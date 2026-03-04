/*
VL5 ToF Sensor Low-Level Script + IR (OUT1D) Support

This script must be uploaded to the Teensy-MUX-ToF chain prior to running the Python Script.
Validate outputs by reading the serial monitor. Close serial monitor in order to run Python file as only one program can
access the serial input at a time.

Supports:
  - 4 VL53L5CX sensors on TCA9548A MUX channels 0-3
  - IR proximity sensor on OUT1D (2-bit logical output on pins D2, D3)
  - Serial protocol: FRAME, IR, MODE lines

IR OUT1D 2-bit encoding:
  00 = no detection (CLEAR)
  01 = far detection (FAR)
  10 = close detection (CLOSE)
  11 = very close / danger (DANGER)

Ver 2.0  -- 4 sensors + IR
*/


#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

// START OF HELPER FUNCTIONS ===================================================

// MUX Configuration
#define TCA_ADDR 0x70
static inline void tcaSelect(uint8_t ch) // Channel selection function
{
  if (ch > 7) return; // Max 8 channels
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch); // Bit shifting for different channels
  Wire.endTransmission();
}

static inline bool i2cProbe(uint8_t addr) // Check existence of I2C device at a given address
{
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

// ToF Sensor Configuration
const uint8_t VL5_ADDR = 0x29;  // default address
const int TARGET_HZ = 15;
const int RES_ZONES = 64;       // 8x8
const int NUM_SENSORS = 4;      // 4 ToF sensors on MUX CH0-CH3

SparkFun_VL53L5CX imager[NUM_SENSORS];
VL53L5CX_ResultsData results[NUM_SENSORS];
bool present[NUM_SENSORS] = {false, false, false, false};

// IR Sensor Configuration (OUT1D -- 2-bit logical output)
const int IR_PIN_BIT0 = 2;  // D2 -- LSB of IR 2-bit output
const int IR_PIN_BIT1 = 3;  // D3 -- MSB of IR 2-bit output
const unsigned long IR_SEND_INTERVAL_MS = 100;  // send IR state every 100ms
unsigned long lastIRSend = 0;

// Mode Control
enum Mode { MODE_MUX, MODE_CH0, MODE_CH1, MODE_CH2, MODE_CH3 };
volatile Mode mode = MODE_MUX;
String cmdBuf;

static inline uint8_t isActive(uint8_t ch) // Check mode activity
{
  if (mode == MODE_MUX) return 1;
  if (mode == MODE_CH0) return (ch == 0) ? 1 : 0;
  if (mode == MODE_CH1) return (ch == 1) ? 1 : 0;
  if (mode == MODE_CH2) return (ch == 2) ? 1 : 0;
  if (mode == MODE_CH3) return (ch == 3) ? 1 : 0;
  return 1;
}

static inline void announceMode() // Announce what mode has been selected
{
  Serial.print("MODE,");
  if (mode == MODE_CH0) Serial.println("CH0");
  else if (mode == MODE_CH1) Serial.println("CH1");
  else if (mode == MODE_CH2) Serial.println("CH2");
  else if (mode == MODE_CH3) Serial.println("CH3");
  else Serial.println("MUX");
}

static inline void printHelp() // Help command, add text if needed.
{
  Serial.println("Commands: CH0 | CH1 | CH2 | CH3 | MUX | HELP");
}

void handleSerialCommands() // Serial command handling
{
  while (Serial.available())
  {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r')
    {
      cmdBuf.trim();
      cmdBuf.toUpperCase();

      if (cmdBuf == "CH0") mode = MODE_CH0;
      else if (cmdBuf == "CH1") mode = MODE_CH1;
      else if (cmdBuf == "CH2") mode = MODE_CH2;
      else if (cmdBuf == "CH3") mode = MODE_CH3;
      else if (cmdBuf == "MUX") mode = MODE_MUX;
      else if (cmdBuf == "HELP" || cmdBuf == "?") printHelp();
      else if (cmdBuf.length() > 0)
      {
        Serial.print("ERR,UnknownCommand,");
        Serial.println(cmdBuf);
        printHelp();
      }

      if (cmdBuf.length() > 0) announceMode();
      cmdBuf = "";
    }
    else
    {
      if (cmdBuf.length() < 32) cmdBuf += c;
    }
  }
}

bool initOneSensor(uint8_t ch, uint8_t idx) // Initialize sensor function
{
  tcaSelect(ch);

  if (!i2cProbe(VL5_ADDR))
  {
    Serial.print("WARN,CH");
    Serial.print(ch);
    Serial.println(",NoDeviceAt0x29");
    return false;
  }

  Serial.print("INFO,CH");
  Serial.print(ch);
  Serial.println(",Begin");

  if (!imager[idx].begin())
  {
    Serial.print("ERR,CH");
    Serial.print(ch);
    Serial.println(",BeginFailed");
    return false;
  }

  // Configure
  imager[idx].setResolution(RES_ZONES);
  imager[idx].setRangingFrequency(TARGET_HZ);
  imager[idx].startRanging();

  Serial.print("INFO,CH");
  Serial.print(ch);
  Serial.println(",Ready");
  return true;
}

void streamFrame(uint8_t ch) // Stream data to Serial Monitor helper function
{
  uint8_t idx = ch;
  if (!present[idx]) return;

  tcaSelect(ch);

  if (!imager[idx].isDataReady()) return;

  if (!imager[idx].getRangingData(&results[idx]))
  {
    Serial.print("ERR,CH");
    Serial.print(ch);
    Serial.println(",GetRangingDataFailed");
    return;
  }

  // Protocol:
  // FRAME,ch,activeFlag,hz,res,d0,d1,...,d63
  Serial.print("FRAME,");
  Serial.print(ch);
  Serial.print(",");
  Serial.print(isActive(ch));
  Serial.print(",");
  Serial.print(TARGET_HZ);
  Serial.print(",");
  Serial.print(RES_ZONES);

  for (int i = 0; i < RES_ZONES; i++)
  {
    Serial.print(",");
    Serial.print(results[idx].distance_mm[i]);
  }
  Serial.println();
}

void readAndSendIR() // Read IR OUT1D pins and send 2-bit value
{
  unsigned long now = millis();
  if (now - lastIRSend < IR_SEND_INTERVAL_MS) return;
  lastIRSend = now;

  uint8_t bit0 = digitalRead(IR_PIN_BIT0) ? 1 : 0;
  uint8_t bit1 = digitalRead(IR_PIN_BIT1) ? 1 : 0;
  uint8_t ir_val = (bit1 << 1) | bit0;  // 2-bit value: 0-3

  // Protocol: IR,bits
  Serial.print("IR,");
  Serial.println(ir_val);
}

// END OF HELPER FUNCTIONS ===================================================


void setup()
{
  Serial.begin(115200); // Baud rate
  delay(300);

  Wire.begin();
  Wire.setClock(1000000); // 1MHz long live teensy

  // IR sensor pins
  pinMode(IR_PIN_BIT0, INPUT);
  pinMode(IR_PIN_BIT1, INPUT);

  // Testing helper functions
  printHelp();
  announceMode();

  // Probe MUX
  if (!i2cProbe(TCA_ADDR))
    Serial.println("ERR,TCA,NotFoundAt0x70");
  else
    Serial.println("INFO,TCA,Found");

  // Initialize all 4 sensors on MUX channels 0-3
  for (int i = 0; i < NUM_SENSORS; i++)
  {
    present[i] = initOneSensor(i, i);
  }

  Serial.println("INFO,Streaming,Started");
  Serial.print("INFO,Sensors,");
  int count = 0;
  for (int i = 0; i < NUM_SENSORS; i++) if (present[i]) count++;
  Serial.print(count);
  Serial.print("/");
  Serial.println(NUM_SENSORS);
}

void loop()
{
  handleSerialCommands();

  // Stream frames from all 4 sensors
  // Active flag will tell the Python script if it should read it or not,
  // but data is always collected.
  for (int i = 0; i < NUM_SENSORS; i++)
  {
    streamFrame(i);
  }

  // Read and stream IR sensor data
  readAndSendIR();

  delay(5);
}
