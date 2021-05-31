#!/usr/bin/python
import os
import subprocess

DEVICE=os.getenv("DEVICE", "5")
CUDA_PATH=os.getenv("CUDA_PATH", "/usr/local/cuda-11.2")

sizes = [60, 75, 90, 105, 120, 135]
vars = ["MINCCACHE", "NEWCACHE"]

def run(VERIFY, FORWARD, PHIOPT, BRANCHYOPT, MINCCACHE, NEWCACHE, SPECPHI, SELECT, OPTIMIZE, RESTRICT, runs, sizes):
    print(f'VERIFY={VERIFY} RESTRICT={RESTRICT} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} PHIOPT={PHIOPT} BRANCHYOPT={BRANCHYOPT} SPECPHI={SPECPHI} SELECT={SELECT} MINCCACHE={MINCCACHE} NEWCACHE={NEWCACHE} make -B -j', flush=True)
    comp = subprocess.run(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} RESTRICT={RESTRICT} FORWARD={FORWARD} PHIOPT={PHIOPT} SPECPHI={SPECPHI} SELECT={SELECT} BRANCHYOPT={BRANCHYOPT} MINCCACHE={MINCCACHE} NEWCACHE={NEWCACHE} make -B -j', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert (comp.returncode == 0)
    
    for size in sizes:
        res = []
        for i in range(runs):
            if VERIFY == "yes":
                if FORWARD == "yes":
                    res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} ./lulesh -s " + str(size) + "| grep \"der=[0-9\.]*\" -o | grep -e \"[0-9\.]*\" -o").read().strip())
                else:
                    res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} ./lulesh -s " + str(size) + "| grep \"out*\" | grep -e \"[0-9\.]*\" -o").read().strip())
            else:
                res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} {CUDA_PATH}/bin/ncu --nvtx -k ApplyMaterialPropertiesAndUpdateVolume_kernel --target-processes all ./lulesh -s " + str(size) + "| grep \"Duration\" ").read().strip())
        print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} RESTRICT={RESTRICT} FORWARD={FORWARD} PHIOPT={PHIOPT} SPECPHI={SPECPHI} SELECT={SELECT} BRANCHYOPT={BRANCHYOPT} MINCCACHE={MINCCACHE} NEWCACHE={NEWCACHE} size={size}', "\t", "\t".join(res), flush=True)

def do(remain, set):
    if len(remain) == 0:
        if set["FORWARD"] == "yes":
            runb = True
            for k in set:
                if k != "OPTIMIZE":
                    if set[k]=="no":
                        runb = False
                        break
            if not runb:
                return
        run(**set)
    else:
        strue = set.copy()
        strue[remain[0]] = "yes"
        do(remain[1:], strue)
        sfalse = set.copy()
        sfalse[remain[0]] = "no"
        do(remain[1:], sfalse)

def merge(a, b):
    c = {}
    for m in a:
        c[m] = a[m]
    for m in b:
        c[m] = b[m]
    return c

def ablation():
    start = {"runs": 1, "sizes": [90], "VERIFY": "no"}
    run(**(merge(start, {"FORWARD": "yes", "OPTIMIZE": "yes", "PHIOPT": "no", "SPECPHI": "no", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "yes", "SELECT": "no"})))
    run(**(merge(start, {"FORWARD": "no", "OPTIMIZE": "no", "PHIOPT": "no", "SPECPHI": "no", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "yes", "SELECT": "no"})))
    run(**(merge(start, {"FORWARD": "no", "OPTIMIZE": "yes", "PHIOPT": "no", "SPECPHI": "no", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "yes", "SELECT": "no"})))
    run(**(merge(start, {"FORWARD": "no", "OPTIMIZE": "yes", "PHIOPT": "no", "SPECPHI": "yes", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "yes", "SELECT": "no"})))

def scaling():
    start = {"runs": 1, "sizes":list(range(60, 136, 15)), "OPTIMIZE": "yes", "VERIFY": "no"}
    run(**(merge(start, {"FORWARD": "yes", "PHIOPT": "no", "SPECPHI": "no", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "no", "SELECT": "no"})))
    run(**(merge(start, {"FORWARD": "no", "PHIOPT": "no", "SPECPHI": "yes", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "no", "SELECT": "no"})))

def verify():
    start = {"runs": 1, "sizes": [90], "OPTIMIZE": "yes", "VERIFY": "yes"}
    run(**(merge(start, {"FORWARD": "yes", "PHIOPT": "no", "SPECPHI": "yes", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "no", "SELECT": "no"})))
    run(**(merge(start, {"FORWARD": "no", "PHIOPT": "no", "SPECPHI": "yes", "BRANCHYOPT": "no", "MINCCACHE": "no", "NEWCACHE": "no", "RESTRICT": "no", "SELECT": "no"})))


ablation()
scaling()
verify()
