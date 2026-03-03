/*
VL5 ToF Sensor Low-Level Script

This script must be uploaded to the Teensy-MUX-ToF chain prior to running the Python Script.
Validate outputs by reading the serial monitor. Close serial monitor in order to run Python file as only one program can
access the serial input at a time.

Ver 1.3
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

SparkFun_VL53L5CX imager[2];
VL53L5CX_ResultsData results[2];
bool present[2] = {false, false};

// Mode Control
enum Mode { MODE_MUX, MODE_CH0, MODE_CH1 };
volatile Mode mode = MODE_MUX;
String cmdBuf;

static inline uint8_t isActive(uint8_t ch) // Check mode activity
{
  if (mode == MODE_MUX) return 1;
  if (mode == MODE_CH0) return (ch == 0) ? 1 : 0;
  if (mode == MODE_CH1) return (ch == 1) ? 1 : 0;
  return 1;
}

static inline void announceMode() // Announce what mode has been selected
{
  Serial.print("MODE,");
  if (mode == MODE_CH0) Serial.println("CH0");
  else if (mode == MODE_CH1) Serial.println("CH1");
  else Serial.println("MUX");
}

static inline void printHelp() // Help command, add text if needed.
{
  Serial.println("Commands: CH0 | CH1 | MUX | HELP");
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

// END OF HELPER FUNCTIONS ===================================================


void setup()
{
  Serial.begin(115200); // Baud rate
  delay(300);

  Wire.begin();
  Wire.setClock(1000000); // 1MHz long live teensy

  // Testing helper functions
  printHelp();
  announceMode();

  // Probe MUX
  if (!i2cProbe(TCA_ADDR))
    Serial.println("ERR,TCA,NotFoundAt0x70");
  else
    Serial.println("INFO,TCA,Found");

  present[0] = initOneSensor(0, 0); // Sensor 1
  present[1] = initOneSensor(1, 1); // Sensor 2, can add more

  Serial.println("INFO,Streaming,Started");
}

void loop()
{
  handleSerialCommands();
  // Active flag will tell the Python script if it should read it or not, but data is always collected. Again, add more if needed.
  streamFrame(0);
  streamFrame(1);
  delay(5);
}