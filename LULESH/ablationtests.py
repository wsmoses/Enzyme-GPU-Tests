#!/usr/bin/python
import os
import subprocess

sizes = [60, 75, 90, 105, 120, 135]
vars = ["MINCCACHE", "NEWCACHE"]

def run(VERIFY, FORWARD, PHIOPT, BRANCHYOPT, MINCCACHE, NEWCACHE, SPECPHI, SELECT, OPTIMIZE, runs, sizes):
    print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} PHIOPT={PHIOPT} BRANCHYOPT={BRANCHYOPT} SPECPHI={SPECPHI} SELECT={SELECT} MINCCACHE={MINCCACHE} NEWCACHE={NEWCACHE} make -B -j', flush=True)
    comp = subprocess.run(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} PHIOPT={PHIOPT} SPECPHI={SPECPHI} SELECT={SELECT} BRANCHYOPT={BRANCHYOPT} MINCCACHE={MINCCACHE} NEWCACHE={NEWCACHE} make -B -j', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert (comp.returncode == 0)
    
    for size in sizes:
        res = []
        for i in range(runs):
            res.append(os.popen("/usr/local/cuda-11.2/bin/ncu --nvtx -k ApplyMaterialPropertiesAndUpdateVolume_kernel --target-processes all ./lulesh -s " + str(size) + "| grep \"Duration\" ").read().strip())
        print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} PHIOPT={PHIOPT} SPECPHI={SPECPHI} SELECT={SELECT} BRANCHYOPT={BRANCHYOPT} MINCCACHE={MINCCACHE} NEWCACHE={NEWCACHE} size={size}', "\t", "\t".join(res), flush=True)

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

def scaling():
    start = {"runs": 1, "sizes":[range(90, 90*13, 90)], "VERIFY": "no"}
    run(**(merge(start, {"FORWARD": "yes", "AGGRPHIOPT": "no", "PHIOPT": "no", "BRANCHYOPT": "no", "MINCCACHE": "yes", "NEWCACHE": yes})))
    run(**(merge(start, {"FORWARD": "no", "AGGRPHIOPT": "no", "PHIOPT": "no", "BRANCHYOPT": "no", "MINCCACHE": "yes", "NEWCACHE": yes})))

# scaling()

do(["OPTIMIZE", "MINCCACHE", "NEWCACHE", "PHIOPT", "SPECPHI", "SELECT"], {"FORWARD":"no", "SPECPHI":"no", "runs":1, "sizes":[90], "VERIFY":"no", "BRANCHYOPT":"no"})