# Message Protocols (ASCII v1)

## Teensy -> Host

### ToF frame
Legacy 8x8:
`TF,<seq>,<mcu_ms>,<sensor_id>,<mux_ch>,<joint_id>,<status>,<d0..d63>,<v0..v63>`

Hyperion variable-grid:
`TF,<seq>,<mcu_ms>,<sensor_id>,<mux_ch>,<joint_id>,<status>,<rows>,<cols>,<d...>,<v...>`

- `seq`: uint32 incrementing frame id
- `mcu_ms`: uint32 `millis()` at acquisition/send
- `sensor_id`: string (for example `S0`, `S1`, `S2`, `S3`)
- `mux_ch`: ToF MUX channel (`0..3` in Hyperion deployment)
- `joint_id`: mount identifier (`wrist_pitch`)
- `status`: bitfield
  - bit0: one or more invalid zones in frame
  - bit1: sensor timeout/fault (reserved)
  - bit2: stale/incomplete frame (reserved)
- `rows`,`cols`: explicit grid shape for variable-resolution frames
- `d...`: distance in mm, flattened row-major
- `v...`: zone validity mask (0/1), flattened row-major

### IMU frame (MPU6050 direct I2C)
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
- `MODE,<MUX|CH0|CH1|CH2|CH3>`
- `CFG,ACT,<count>` where `count` is active ToF channel count (`1..4` on Hyperion)
- `CFG,GRID,<rows>,<cols>`
- `CFG,TARGET_HZ,<hz>`

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
- ToF stream:
  - legacy sketch: target ~15 Hz per active ToF sensor, 8x8
  - Hyperion sketch: sensor-side target 60 Hz, 4x4, across CH0..CH3 with host-controlled active count
- IMU stream: 100 Hz (`10 ms` interval).
- Mega heartbeat: 4 Hz (`250 ms`).

## Error Handling
- Host rejects malformed frames/packets.
- Mega rejects malformed or out-of-range commands and emits `ACK ERR_*`.
- Mega watchdog timeout (`750 ms`) enters hold/comms-fault behavior.
- Host falls back to hold mode on stale ToF/IMU, solver failure, or link loss.

## Compatibility Notes
- Host still parses legacy `FRAME` lines for ToF fallback.
- Host parses both legacy 8x8 `TF` packets and Hyperion variable-grid `TF` packets.
- Existing legacy Mega command handling remains available while v1 packets coexist.

