import os
import subprocess 

DEVICE=os.getenv("DEVICE", "1")

def run(OPTIMIZE, FORWARD, INLINE, NEWCACHE, MINCACHE, AA, PHISTRUCT, TEMPLATIZE, DYN, VERIFY, COALESE, CACHELICM, SPECPHI, SELECT, runs, sizes):
    comp = subprocess.run(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} DYN={DYN} INLINE={INLINE} COALESE={COALESE} CACHELICM={CACHELICM} SPECPHI={SPECPHI} SELECT={SELECT} NEWCACHE={NEWCACHE} MINCACHE={MINCACHE} AA={AA} PHISTRUCT={PHISTRUCT} TEMPLATIZE={TEMPLATIZE} make -B -j', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} DYN={DYN} INLINE={INLINE}  COALESE={COALESE} CACHELICM={CACHELICM} SPECPHI={SPECPHI} SELECT={SELECT} NEWCACHE={NEWCACHE} MINCACHE={MINCACHE} AA={AA} PHISTRUCT={PHISTRUCT} TEMPLATIZE={TEMPLATIZE} make -B -j')
    # comp = subprocess.run(f'OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} INLINE={INLINE} NEWCACHE={NEWCACHE} AA={AA} PHISTRUCT={PHISTRUCT} TEMPLATIZE={TEMPLATIZE} VERIFY={VERIFY} make -B -j', shell=True)

    assert (comp.returncode == 0)
    out = {}
    for size in sizes:
        res = []
        for i in range(runs):
            if VERIFY == "yes":
                res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} ./XSBench -m event -k 0 -l " + str(size) + "| grep \"der=[0-9\.]*\" -o | grep -e \"[0-9\.]*\" -o").read().strip())
            else:
                res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} ./XSBench -m event -k 0 -l " + str(size) + "| grep \"Runtime\" | grep -e \"[0-9\.]*\" -o").read().strip())
        out[size] = res
        print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} DYN={DYN} INLINE={INLINE} COALESE={COALESE} CACHELICM={CACHELICM} SPECPHI={SPECPHI} SELECT={SELECT} NEWCACHE={NEWCACHE} MINCACHE={MINCACHE} AA={AA} PHISTRUCT={PHISTRUCT} TEMPLATIZE={TEMPLATIZE} size={size}', "\t", "\t".join(res), flush=True)
    return res

vars = ["OPTIMIZE", "INLINE", "NEWCACHE", "AA", "PHISTRUCT", "TEMPLATIZE" , "FORWARD"]

def do(remain, set):
    if len(remain) == 0:
        print(set)
        run(**set)
    else:
        strue = set.copy()
        strue[remain[0]] = "yes"
        do(remain[1:], strue)
        sfalse = set.copy()
        sfalse[remain[0]] = "no"
        do(remain[1:], sfalse)

vars = ["OPTIMIZE", "INLINE", "NEWCACHE", "AA", "PHISTRUCT", "TEMPLATIZE" , "FORWARD"]

# do(["NEWCACHE", "MINCACHE", "TEMPLATIZE", "PHISTRUCT", "SPECPHI", "SELECT"], {"runs":1, "sizes":[17000000], "OPTIMIZE":"yes", "AA":"yes", "INLINE":"yes", "VERIFY":"no", "FORWARD":"no"})

# do(["AA", "COALESE", "NEWCACHE", "MINCACHE", "CACHELICM"], {"TEMPLATIZE":"no", "INLINE":"no", "MINCACHE":"no", "SELECT":"no", "SPECPHI":"no", "PHISTRUCT":"no", "runs":1, "sizes":[17000000], "OPTIMIZE":"yes", "VERIFY":"no", "FORWARD":"no"})

def merge(a, b):
    c = {}
    for m in a:
        c[m] = a[m]
    for m in b:
        c[m] = b[m]
    return c

def ablation():
    start = {"runs": 5, "sizes": [17000000], "VERIFY":"no", "SELECT":"no", "SPECPHI":"no", "PHISTRUCT":"no", "CACHELICM":"yes", "COALESE":"no"}

    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "yes", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"yes", "DYN":"yes"})))

    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"yes", "DYN":"yes"})))

    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"no", "DYN":"yes"})))

    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "no", "TEMPLATIZE":"no", "DYN":"yes"})))

    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "no", "TEMPLATIZE":"no", "DYN":"no"})))

    # No opt may time out, commented out to ease running script
    # run(**(merge(start,{"OPTIMIZE":"no", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "no", "TEMPLATIZE":"no", "DYN":"no"})))

def scaling():
    start = {"runs": 5, "sizes": list(range(17000000, 17000000*13, 17000000)), "VERIFY":"no", "SELECT":"no", "SPECPHI":"no", "PHISTRUCT":"no", "CACHELICM":"yes", "COALESE":"no"}
    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "yes", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"yes", "DYN":"yes"})))
    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"yes", "DYN":"yes"})))

def verify():
    start = {"runs": 1, "sizes": [17000], "VERIFY":"yes", "SELECT":"no", "SPECPHI":"no", "PHISTRUCT":"no", "CACHELICM":"yes", "COALESE":"no"}

    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "yes", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"yes", "DYN":"yes"})))
    run(**(merge(start,{"OPTIMIZE":"yes", "FORWARD": "no", "INLINE": "yes", "MINCACHE": "yes", "NEWCACHE": "yes", "AA": "yes", "PHISTRUCT": "yes", "TEMPLATIZE":"yes", "DYN":"yes"})))

ablation()
scaling()
verify()
