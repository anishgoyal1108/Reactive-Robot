# ToF_RR_Sensing — VL53L5CX Round-Robin via TCA9548A (Teensy 4.1)

This repo contains:
- **Arduino/Teensy sketch**: reads two VL53L5CX sensors through a TCA9548A I2C multiplexer (CH0 + CH1) and streams frames over Serial.
- **Python console viewers**: Two scripts, one for visual impact (live_view), another simply displaying console readings (GridView_live). Im terrible at naming things

## Hardware

- **Teensy 4.1** connected to **TCA9548A** via pin 19 and 18 for I2C comms.
- VL53L5CX sensor connected to:
  - **TCA Channel 0** (CH0)
  - **TCA Channel 1** (CH1)
- Both sensors can remain at the default address **0x29** because the mux isolates channels.
- Power + logic level: **3.3V I2C** for ALL hardware (excluding the Teensy since its a beast that generates power through dark magic (USB)).

### Arduino Payload:

The Teensy streams frames like:

```
FRAME,0,1,15,64,<64 distance values...>
FRAME,1,1,15,64,<64 distance values...>
```

Where:
- `FRAME` = frame marker
- `0/1` = channel index
- `activeFlag` = kept for compatibility (viewer ignores it; we run in MUX-style)
- `15` = ranging frequency (Hz)
- `64` = resolution (8×8)
- followed by **64 integers** (distance in mm)

## Python Setup
- Install dependencies in the requirements.txt file (I think I was running **Python 3.9?**)
- Ensure that the Arduino sketch is uploaded to the Teensy
- Run either program through as PS console or PyCharm if you're weak.

