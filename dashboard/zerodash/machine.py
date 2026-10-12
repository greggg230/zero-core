"""The machine strip: GPU (nvidia-smi), CPU and memory (psutil), free disk, the Mac's load.

A background thread samples everything (the Mac every ~60 s, the rest every ~10 s); a page
only ever reads the newest sample, so a slow or missing tool never blocks it. Each part
degrades on its own: a missing nvidia-smi / psutil / drive / unreachable Mac becomes
{"error": "..."} for that part only.
"""
import re
import shutil
import subprocess
import threading
import time

NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
GPU_QUERY = ['--query-gpu=memory.used,memory.total,power.draw,utilization.gpu', '--format=csv,noheader,nounits']


def run_cmd(argv, timeout=10):
    """-> (returncode, stdout); raises OSError / TimeoutExpired."""
    p = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=timeout, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return p.returncode, p.stdout


def _num(s):
    try:
        return float(s.strip())
    except (TypeError, ValueError):
        return None                   # '[N/A]' (e.g. power on some boards)


def parse_gpu(out):
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        used, total, power, util = (_num(p) for p in parts[:4])
        gpus.append({'mem_used_mb': used, 'mem_total_mb': total, 'power_w': power, 'util_pct': util})
    return gpus


def parse_uptime(out):
    """`uptime` (macOS 'load averages: 2.31 2.10 1.95', Linux 'load average: 0.1, 0.2, 0.3',
    decimal commas too) -> [1, 5, 15 min load] or None."""
    m = re.search(r'load averages?:\s*(.*)$', out.strip(), re.M)
    if not m:
        return None
    nums = re.findall(r'\d+(?:[.,]\d+)?', m.group(1))
    if len(nums) < 3:
        return None
    return [float(n.replace(',', '.')) for n in nums[:3]]


class Machine:
    def __init__(self, cfg, runner=run_cmd, ssh=None, clock=time.time, interval=10.0, mac_interval=60.0):
        self.cfg = cfg
        self.runner = runner
        self.ssh = ssh
        self.clock = clock
        self.interval = interval
        self.mac_interval = mac_interval
        self.data = {'gpu': None, 'cpu': None, 'disks': [], 'mac': None, 'sampled': None}
        self.lock = threading.Lock()
        self.th = self.mac_th = None
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None

    # -------------------------------------------------------------- parts

    def sample_gpu(self):
        exe = shutil.which('nvidia-smi')
        if not exe:
            return {'error': 'nvidia-smi not found'}
        try:
            code, out = self.runner([exe] + GPU_QUERY)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {'error': f'nvidia-smi: {e}'}
        if code != 0:
            return {'error': f'nvidia-smi exited with status {code}'}
        gpus = parse_gpu(out)
        return {'gpus': gpus} if gpus else {'error': 'nvidia-smi: no GPU listed'}

    def sample_cpu(self):
        if self.psutil is None:
            return {'error': 'psutil not installed'}
        try:
            vm = self.psutil.virtual_memory()
            first, self.cpu_primed = not getattr(self, 'cpu_primed', False), True
            pct = self.psutil.cpu_percent(0.5 if first else None)     # the first call measures half a second
            return {'percent': pct, 'cores': self.psutil.cpu_count(),
                    'mem_percent': vm.percent, 'mem_used_gb': round((vm.total - vm.available) / 2**30, 1),
                    'mem_total_gb': round(vm.total / 2**30, 1)}
        except Exception as e:                     # noqa: BLE001
            return {'error': str(e)}

    def sample_disks(self):
        out = []
        for d in self.cfg.get('disks', []):
            item = {'label': d['label'], 'path': d['path'], 'warn_below_gb': d.get('warn_below_gb')}
            try:
                u = shutil.disk_usage(d['path'])
                item.update(free_gb=round(u.free / 2**30, 1), total_gb=round(u.total / 2**30, 1))
                item['warn'] = d.get('warn_below_gb') is not None and u.free / 2**30 < d['warn_below_gb']
            except OSError as e:
                item['error'] = e.strerror or str(e)
            out.append(item)
        return out

    def sample_mac(self):
        mac = self.cfg.get('mac')
        if not mac:
            return None
        res = {'label': mac['label'], 'host': mac['ssh'], 'time': self.clock()}
        if self.ssh is None:
            res['error'] = 'no ssh'
            return res
        try:
            code, out, err = self.ssh.run(mac['ssh'], 'uptime', timeout=15)
        except Exception as e:                     # noqa: BLE001 - SshError: no ssh, timeout
            res['error'] = str(e)
            return res
        load = parse_uptime(out) if code == 0 else None
        if load is None:
            lines = (err or out or '').strip().splitlines()
            res['error'] = lines[-1][:160] if lines else f'ssh exited with status {code}'
        else:
            res['load'] = load
            res['uptime'] = out.strip()
        return res

    def sample(self):
        """Everything but the Mac, now (the sampler thread; tests)."""
        gpu, cpu, disks = self.sample_gpu(), self.sample_cpu(), self.sample_disks()
        with self.lock:
            self.data.update(gpu=gpu, cpu=cpu, disks=disks, sampled=self.clock())

    def sample_mac_now(self):
        m = self.sample_mac()
        with self.lock:
            prev = self.data.get('mac')
            if m and m.get('error') and prev and prev.get('load'):
                m['last_load'], m['last_time'] = prev['load'], prev['time']
            self.data['mac'] = m

    def snapshot(self):
        with self.lock:
            out = {k: v for k, v in self.data.items()}
        now = self.clock()
        out['age_s'] = round(now - out['sampled'], 1) if out.get('sampled') else None
        if out.get('mac') and out['mac'].get('time'):
            out['mac'] = dict(out['mac'], age_s=round(now - out['mac']['time']))
        return out

    # -------------------------------------------------------------- threads

    def start(self):
        if self.th is None:
            self.th = threading.Thread(target=self._loop, args=(self.sample, self.interval), daemon=True,
                                       name='zerodash-machine')
            self.th.start()
        if self.mac_th is None and self.cfg.get('mac'):
            self.mac_th = threading.Thread(target=self._loop, args=(self.sample_mac_now, self.mac_interval),
                                           daemon=True, name='zerodash-mac')
            self.mac_th.start()

    @staticmethod
    def _loop(fn, every):
        while True:
            try:
                fn()
            except Exception:                      # noqa: BLE001 - keep sampling
                pass
            time.sleep(every)
