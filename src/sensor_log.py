"""Append GPU/CPU/RAM readings to a CSV every few seconds, fsync'd so they survive a hard reset.

    python src/sensor_log.py logs/sensors.csv 5      (runs until killed)
"""
import os, subprocess, sys, time

out, every = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5
q = 'power.draw,temperature.gpu,utilization.gpu,clocks.sm,memory.used,fan.speed,clocks_event_reasons.active'


def cpu_package_temp():
    for h in os.listdir('/sys/class/hwmon'):
        d = f'/sys/class/hwmon/{h}'
        if open(f'{d}/name').read().strip() == 'coretemp':
            return int(open(f'{d}/temp1_input').read()) / 1000
    return float('nan')


def ram_used_gib():
    m = dict(l.split(':') for l in open('/proc/meminfo'))
    kb = lambda k: int(m[k].split()[0])
    return (kb('MemTotal') - kb('MemAvailable')) / 2**20


with open(out, 'a') as f:
    if f.tell() == 0: f.write('time,gpu_w,gpu_c,gpu_util,gpu_mhz,gpu_mem_mib,fan,throttle,cpu_c,load1,ram_gib\n')
    while True:
        try:
            g = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10).stdout.strip().replace(', ', ',')
        except Exception as e:
            g = f'err {e}'.replace(',', ';')
        f.write(f'{time.strftime("%H:%M:%S")},{g},{cpu_package_temp():.0f},{os.getloadavg()[0]:.1f},{ram_used_gib():.1f}\n')
        f.flush(); os.fsync(f.fileno()); time.sleep(every)
