# Message Protocols (ASCII v1)

## Teensy -> Host

### ToF frame
`TF,<seq>,<mcu_ms>,<sensor_id>,<mux_ch>,<joint_id>,<status>,<d0..d63>,<v0..v63>`

- `seq`: uint32 incrementing frame id
- `mcu_ms`: uint32 `millis()` at acquisition/send
- `sensor_id`: string (for example `S0`, `S1`)
- `mux_ch`: ToF channel (`0` or `1` in current deployment)
- `joint_id`: mount identifier (`wrist_pitch`)
- `status`: bitfield
  - bit0: one or more invalid zones in frame
  - bit1: sensor timeout/fault (reserved)
  - bit2: stale/incomplete frame (reserved)
- `d0..d63`: distance in mm
- `v0..v63`: zone validity mask (0/1)

### IMU frame (MPU6050 direct I2C, logical CH2 stream)
`IMU,<seq>,<mcu_ms>,<ax_g>,<ay_g>,<az_g>,<gx_dps>,<gy_dps>,<gz_dps>,<temp_c>,<status>`

- `ax_g..az_g`: acceleration in g
- `gx_dps..gz_dps`: angular rate in deg/s
- `temp_c`: temperature in C
- `status`: bitfield
  - bit0: IMU not present
  - bit1: IMU init failed
  - bit2: IMU read failure

### Auxiliary
- `IR,<bits>` where bits in `{0,1,2,3}`
- `MODE,<MUX|CH0|CH1>`
- `CFG,ACT,<count>` where `count` is active ToF channel count (`1..2`)

## Host -> Mega
`CMD,<seq>,<host_ms>,<mode>,<speed_scale>,<b>,<s>,<e>,<wv>,<wr>,<g>`

- `seq`: uint32 command id
- `host_ms`: epoch ms at send time
- `mode`: `nominal_tracking|obstacle_aware_tracking|fallback_hold|comms_fault`
- `speed_scale`: float `[0..1]`
- `b,s,e,wv,wr,g`: joint targets in degrees

Host also uses legacy-compatible MCU-IK command path:
`SET IKP <theta_deg> <r_mm> <z_mm> <wrist_offset_deg> <wr> <g>`

## Mega -> Host
- `ACK,<seq>,<status>,<last_applied_ms>`
  - `status`: `OK|ERR_FORMAT|ERR_RANGE|ERR_*`
- `STAT,<mode>,<last_seq>,<stale_ms>` periodic heartbeat
- Legacy responses remain available (`PONG`, `OK IK=...`, `POS=...`, `ERR ...`)

## Units and Rates
- Distances: mm in packets; meters internally for geometric modeling.
- Angles: degrees.
- ToF stream: target ~15 Hz per active ToF sensor (CH0/CH1).
- IMU stream: 100 Hz (`10 ms` interval).
- Mega heartbeat: 4 Hz (`250 ms`).

## Error Handling
- Host rejects malformed frames/packets.
- Mega rejects malformed or out-of-range commands and emits `ACK ERR_*`.
- Mega watchdog timeout (`750 ms`) enters hold/comms-fault behavior.
- Host falls back to hold mode on stale ToF/IMU, solver failure, or link loss.

## Compatibility Notes
- Host still parses legacy `FRAME` lines for ToF fallback.
- Existing legacy Mega command handling remains available while v1 packets coexist.

