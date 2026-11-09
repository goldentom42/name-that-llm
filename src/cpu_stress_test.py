"""Progressive all-core CPU load test, to find out why the machine resets under many-thread work.

Stages of N worker processes (default 4, 8, 12, 24 threads for 2 min, then 24 threads for 10 min: the CPU-only
reset of 2026-11-06 came 7-10 min into a 24-thread job). Every 0.5 s one CSV row is written and fsync'd, so the file
shows the last readings even after a hard reset. Stops all load at once when any core reaches --limit (default 95 C;
this CPU throttles at 100 C and cuts out at 110 C). Exit code: 0 finished, 2 stopped by the limit.

Reading the result:
  limit reached within seconds at 12+ threads  -> pump or cooler contact
  limit reached after minutes                  -> cooler too small or the case air too warm
  machine resets while temperatures are normal -> CPU power delivery (motherboard/VRM) or BIOS, not cooling

The workload is single-threaded BLAS matrix products (AVX-512), harsher than the scikit-learn job that crashed.
Refuses to start while a training run is active. --dry replaces the load by sleeping workers, to test the script.

    python src/cpu_stress_test.py                     # about 17 min, logs/cpu_stress.csv
"""
import argparse, glob, multiprocessing as mp, os, subprocess, sys, time
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')  # one thread per worker
import numpy as np

TRAINING = 'train_qwen_lora|qwen_mix_folds|encoder_mix_folds|train_pool|run_qwen3_4b'


def worker(dry):
    os.nice(10)  # the monitor must always get CPU time to read sensors and stop the load
    if dry:
        while True: time.sleep(1)
    a = np.random.rand(400, 400)
    while True: a @ a


def coretemp_files():
    for d in glob.glob('/sys/class/hwmon/hwmon*'):
        if open(f'{d}/name').read().strip() == 'coretemp':
            return sorted(glob.glob(f'{d}/temp*_input'))
    sys.exit('coretemp sensors not found: refusing to run without temperature readings')


def read_int(path):
    try:
        return int(open(path).read())
    except (OSError, ValueError):
        return None


def throttle_events():
    return sum(read_int(p) or 0 for p in glob.glob('/sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count'))


class Power:
    """CPU package watts from RAPL; nan when the counter is root-only."""
    path = '/sys/class/powercap/intel-rapl:0/energy_uj'

    def __init__(self):
        self.e, self.t = read_int(self.path), time.time()

    def read(self):
        e, t = read_int(self.path), time.time()
        w = float('nan') if e is None or self.e is None or e < self.e else (e - self.e) / 1e6 / (t - self.t)
        self.e, self.t = e, t
        return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='4:120,8:120,12:120,24:600', help='threads:seconds,...')
    ap.add_argument('--cooldown', type=int, default=45, help='seconds without load between stages')
    ap.add_argument('--limit', type=float, default=95.0, help='stop all load when any core reaches this temperature')
    ap.add_argument('--out', default='logs/cpu_stress.csv')
    ap.add_argument('--dry', action='store_true', help='sleeping workers, no CPU load (tests the script)')
    a = ap.parse_args()

    if not a.dry and subprocess.run(['pgrep', '-f', TRAINING], capture_output=True).returncode == 0:
        sys.exit('a training run is active: wait for it to finish before loading the CPU')
    files = coretemp_files(); freqs = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq')
    stages = [tuple(int(x) for x in s.split(':')) for s in a.stages.split(',')]
    power, procs = Power(), []
    log = open(a.out, 'a')
    if log.tell() == 0: log.write('time,stage,threads,elapsed_s,max_c,package_c,mhz,cpu_w\n')

    def stop():
        for p in procs: p.kill()
        for p in procs: p.join()
        procs.clear()

    def sample(stage, n, elapsed):
        t = [read_int(f) for f in files]
        if None in t: raise RuntimeError('a temperature sensor stopped answering')
        mhz = np.mean([read_int(f) or 0 for f in freqs]) / 1000; w = power.read()
        row = (max(t) / 1000, t[0] / 1000, mhz, w)
        log.write(f'{time.strftime("%H:%M:%S")},{stage},{n},{elapsed:.1f},{row[0]:.0f},{row[1]:.0f},{row[2]:.0f},{row[3]:.1f}\n')
        log.flush(); os.fsync(log.fileno())
        return row

    print(f'limit {a.limit:.0f} C, stages {stages}, log {a.out}{" (DRY: no load)" if a.dry else ""}', flush=True)
    print(f'idle: {sample("idle", 0, 0)[0]:.0f} C', flush=True)
    code = 0
    try:
        for i, (n, secs) in enumerate(stages):
            th0 = throttle_events(); rows = []; t0 = time.time(); first85 = None
            procs.extend(mp.Process(target=worker, args=(a.dry,), daemon=True) for _ in range(n))
            for p in procs: p.start()
            while (el := time.time() - t0) < secs:
                r = sample(i, n, el); rows.append(r)
                if first85 is None and r[0] >= 85: first85 = el
                if r[0] >= a.limit:
                    stop(); code = 2
                    print(f'STOPPED: {r[0]:.0f} C reached after {el:.0f} s at {n} threads', flush=True)
                    break
                time.sleep(0.5)
            stop()
            last = np.array(rows[-60:]); w = last[:, 3]
            print(f'stage {i}: {n:2d} threads {el:4.0f} s | max {max(r[0] for r in rows):.0f} C, last 30 s mean {last[:, 0].mean():.0f} C, '
                  f'{last[:, 2].mean():.0f} MHz, {"n/a" if np.isnan(w).all() else f"{np.nanmean(w):.0f} W"} | '
                  f'reached 85 C: {"never" if first85 is None else f"after {first85:.0f} s"} | throttle events +{throttle_events() - th0}', flush=True)
            if code: break
            if i < len(stages) - 1:
                t1 = time.time()
                while time.time() - t1 < a.cooldown: sample(f'cool{i}', 0, time.time() - t1); time.sleep(1)
    finally:
        stop()
    t = sample('end', 0, 0)
    print(f'done, load removed, {t[0]:.0f} C now. Exit code {code}.', flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
