# benchmark_v2 28-model formal run package

This package transcribes the uploaded E1-A through E3-C runbooks.

- 26 trainable models use `train`.
- Persistence and MovingAverage use `evaluate-only`.
- One model runs in one Python process at a time.
- The suite stops on the first nonzero exit code.
- Existing formal run directories are never overwritten.

## Recommended gate

Run the all-26 preflight script on the actual deployment machine first. The ten
models already marked hardware-preflight-required are TiDE, SegRNN,
Transformer, PatchTST, WPMixer, MultiPatchFormer, TimeMixer, FreTS, MSGNet,
and TimeFilter.

## Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& .\benchmark_v2_preflight_all_26_windows.ps1
& .\benchmark_v2_run_all_28_windows.ps1
```

## Linux

```bash
chmod +x /path/to/benchmark_v2_*.sh
export PROJECT_ROOT="$(pwd)"
bash /path/to/benchmark_v2_preflight_all_26_linux.sh
bash /path/to/benchmark_v2_run_all_28_linux.sh
```

## Linux nohup + exit code + sync + unconditional shutdown

```bash
mkdir -p logs/benchmark_v2
export PROJECT_ROOT="$(pwd)"
nohup env PROJECT_ROOT="$PROJECT_ROOT"   bash /path/to/benchmark_v2_run_all_28_linux_autoshutdown.sh   > logs/benchmark_v2/all_28_seed2026.log 2>&1 &
```

Inspect:

```bash
tail -f logs/benchmark_v2/all_28_seed2026.log
ps -ef | grep '[r]un_benchmark.py'
watch -n 2 nvidia-smi
```
