#!/usr/bin/env python3
"""
tof_serial_diagnose.py - Standalone Teensy ToF serial line checker.

By default this does NOT clear the RX buffer after open so you still see
boot-time ERR/WARN/INFO about the TCA9548 mux and VL53L5 sensors.  Use
--flush only if you need a clean capture window.

Usage:
    python tof_serial_diagnose.py /dev/ttyACM0
    python tof_serial_diagnose.py /dev/ttyACM0 --seconds 5 --act 4
    python tof_serial_diagnose.py /dev/ttyACM0 --flush
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

try:
    import serial
except ImportError:
    print('pyserial required: pip install pyserial', file=sys.stderr)
    sys.exit(2)

from braccio_ctrl.serial_bridge import ensure_serial_device_path

_BOOT_PREFIXES = frozenset({
    'ERR', 'WARN', 'INFO', 'CFG', 'MODE',
})


def _classify_prefix(line: str) -> str:
    if not line:
        return 'EMPTY'
    if ',' in line:
        return line.split(',', 1)[0]
    return 'MISC'


def _try_parse_tf(line: str) -> dict | None:
    if not line.startswith('TF,'):
        return None
    parts = line.split(',')
    if len(parts) < 10:
        return {'error': 'too_few_fields', 'n': len(parts)}
    try:
        ch = int(parts[4])
        rows = int(parts[7])
        cols = int(parts[8])
        zones = rows * cols
        d0 = 9
        d1 = d0 + zones
        if len(parts) < d1:
            return {'error': 'truncated_distances', 'need': d1, 'have': len(parts)}
        dist = [int(float(parts[i])) for i in range(d0, d1)]
        vmin, vmax = min(dist), max(dist)
        n_valid = None
        if len(parts) >= d1 + zones:
            flags = [int(parts[i]) for i in range(d1, d1 + zones)]
            n_valid = sum(flags)
        return {
            'ch': ch, 'rows': rows, 'cols': cols,
            'dist_min': vmin, 'dist_max': vmax,
            'zones_valid': n_valid, 'zones_total': zones,
        }
    except Exception as e:
        return {'error': str(e)}


def _try_parse_frame(line: str) -> dict | None:
    if not line.startswith('FRAME,'):
        return None
    parts = line.split(',')
    if len(parts) < 6:
        return {'error': 'too_few_fields'}
    try:
        ch = int(parts[1])
        res = int(parts[4])
        need = 5 + res
        if len(parts) < need:
            return {'error': 'truncated', 'need': need, 'have': len(parts)}
        vals = [int(float(parts[i])) for i in range(5, need)]
        return {
            'ch': ch, 'res': res,
            'dist_min': min(vals), 'dist_max': max(vals),
        }
    except Exception as e:
        return {'error': str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description='Diagnose Teensy ToF serial stream')
    ap.add_argument(
        'port',
        help='USB serial device, e.g. /dev/ttyACM0 (not UDP port 9871)',
    )
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--act', type=int, default=None,
                    help='Send ACT N after open (vl5_tca_4x4 firmware)')
    ap.add_argument(
        '--flush',
        action='store_true',
        help='Clear input buffer after open (hides boot ERR/WARN/INFO)',
    )
    args = ap.parse_args()

    try:
        ensure_serial_device_path(args.port, 'serial device (positional argument)')
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    print('Opening %s @ %d ...' % (args.port, args.baud))
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except Exception as e:
        print('FAILED to open serial:', e, file=sys.stderr)
        sys.exit(1)

    # USB enumerated: MCU may still be in bootloader reset — give setup() time
    time.sleep(1.2)

    if args.flush:
        ser.reset_input_buffer()
        print('(RX buffer flushed — boot messages discarded)')
    if args.act is not None:
        ser.write(('ACT %d\n' % args.act).encode())
        ser.flush()
        print('Sent: ACT', args.act)

    t_end = time.monotonic() + args.seconds
    by_kind: Counter[str] = Counter()
    bytes_read = 0
    sample: dict[str, str] = {}
    first_tf: dict | None = None
    first_frame: dict | None = None
    boot_lines: list[str] = []

    def consume_line(line: str) -> None:
        nonlocal first_tf, first_frame
        kind = _classify_prefix(line)
        by_kind[kind] += 1
        if kind not in sample and kind not in ('EMPTY', 'MISC'):
            sample[kind] = line[:220]
        if kind in _BOOT_PREFIXES or line.startswith('Commands:'):
            if line not in boot_lines and len(boot_lines) < 50:
                boot_lines.append(line[:300])
        if kind == 'TF' and first_tf is None:
            first_tf = _try_parse_tf(line)
        if kind == 'FRAME' and first_frame is None:
            first_frame = _try_parse_frame(line)

    while time.monotonic() < t_end:
        raw = ser.readline()
        if not raw:
            continue
        bytes_read += len(raw)
        line = raw.decode('utf-8', errors='replace').strip()
        consume_line(line)

    ser.close()

    print('\n--- %.1fs summary ---' % args.seconds)
    print('Total bytes read:', bytes_read)
    if bytes_read == 0:
        print('VERDICT: No serial data - wrong port, permissions, or firmware silent.')
        sys.exit(1)

    if boot_lines and not args.flush:
        print('\n--- Boot / status lines (first seen) ---')
        for bl in boot_lines:
            print(' ', bl)

    print('\nLine counts by prefix:')
    for k, v in sorted(by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        print('  %-12s %5d' % (k, v))

    if sample:
        print('\nFirst line of each kind:')
        for k in sorted(sample.keys()):
            print('  [%s] %s' % (k, sample[k]))

    n_tof = by_kind.get('TF', 0) + by_kind.get('FRAME', 0)
    n_ir = by_kind.get('IR', 0)

    print('\n--- Parse checks ---')
    if first_tf:
        if 'error' in first_tf:
            print('  First TF:', first_tf)
        else:
            nz = first_tf.get('zones_valid')
            ch = first_tf['ch']
            rr, cc = first_tf['rows'], first_tf['cols']
            d0, d1 = first_tf['dist_min'], first_tf['dist_max']
            zt = first_tf['zones_total']
            ex = ('  valid_zones=%s/%s' % (nz, zt)) if nz is not None else ''
            print('  First TF: ch=%s grid=%sx%s raw_mm %s..%s%s'
                  % (ch, rr, cc, d0, d1, ex))
            if nz == 0:
                print('  NOTE: firmware marked all zones invalid (vl5_tca_4x4: '
                      'valid only for 40<mm<3000). Host hides those as NaN.')
    else:
        print('  No TF lines.')

    if first_frame:
        if 'error' in first_frame:
            print('  First FRAME:', first_frame)
        else:
            print('  First FRAME: ch=%s res=%s range %s..%s'
                  % (first_frame['ch'], first_frame['res'],
                     first_frame['dist_min'], first_frame['dist_max']))
    else:
        print('  No FRAME lines.')

    print('\n--- Verdict ---')
    if n_tof == 0 and n_ir == 0:
        print('No TF/FRAME/IR - wrong sketch or wiring.')
    elif n_tof == 0 and n_ir > 0:
        print('MCU alive (IR + probably IMU), but no TF/FRAME grid lines.')
        print('  Hardware: vl5_tca_4x4 only emits TF if at least one VL53 '
              'passes init on the TCA9548 mux; if all init fail, loop() never '
              'calls streamToFFrameV1.')
        print('  Check boot messages above for: ERR,TCA | WARN,CH* | ERR,CH* | '
              'BeginFailed | NoDeviceAt0x29.')
        print('  Wrong firmware: legacy sketch uses FRAME, not TF — flash the '
              'sketch that matches your Python parser (vl5_tca_4x4 vs vl5_tca_RR).')
        if args.flush:
            print('  Tip: re-run **without** --flush to capture boot ERR/WARN/INFO.')
    else:
        print('Grid lines OK on serial. If UI is empty, check distance masking '
              '(40–3000 mm valid) or UDP from braccio_ctrl.')
    print('done.')


if __name__ == '__main__':
    main()
