import os
import subprocess

def run(OPTIMIZE, FORWARD, INLINE, NEWCACHE, AA, PHISTRUCT, TEMPLATIZE, runs):
    comp = subprocess.run(f'OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} INLINE={INLINE} NEWCACHE={NEWCACHE} AA={AA} PHISTRUCT={PHISTRUCT} TEMPLATIZE={TEMPLATIZE} make -B -j', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert (comp.returncode == 0)
    out = {}
    size = "large"
    res = []
    for i in range(runs):
        res.append(os.popen("./XSBench -m event -k " + size + "| grep \"Runtime\" | grep -e \"[0-9\.]*\" -o").read().strip())
    out[size] = res
    print(f'OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} INLINE={INLINE} NEWCACHE={NEWCACHE} AA={AA} PHISTRUCT={PHISTRUCT} TEMPLATIZE={TEMPLATIZE}', "\t", "\t".join(res), flush=True)
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

do(vars[0:], {"runs":5})
