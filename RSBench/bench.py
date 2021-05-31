import os
import subprocess 

DEVICE=os.getenv("DEVICE", "1")

# AA broken here
vars = ["AA", "NEWCACHE", "MINCUT", "OPTIMIZE", "CACHELICM", "PHISTRUCT", "INLINE", "FORWARD"]

def run(CACHELICM, OPTIMIZE, FORWARD, INLINE, NEWCACHE, MINCUT, AA, PHISTRUCT, runs, sizes):
    print(f'CACHELICM={CACHELICM} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} INLINE={INLINE} MINCUT={MINCUT} NEWCACHE={NEWCACHE} AA={AA} PHISTRUCT={PHISTRUCT} make -B -j', flush=True)
    comp = subprocess.run(f'CACHELICM={CACHELICM} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} INLINE={INLINE} MINCUT={MINCUT} NEWCACHE={NEWCACHE} AA={AA} PHISTRUCT={PHISTRUCT} make -B -j', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert (comp.returncode == 0)
    for size in sizes:
        res = []
        for i in range(runs):
            res.append(os.popen("CUDA_VISIBLE_DEVICES=1 ./rsbench -m event -l " + str(size) + "| grep \"Runtime\" | grep -e \"[0-9\.]*\" -o").read().strip())
            # print(res, flush=True)
        print(f'CACHELICM={CACHELICM} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} INLINE={INLINE} MINCUT={MINCUT} NEWCACHE={NEWCACHE} AA={AA} PHISTRUCT={PHISTRUCT} size={size}', "\t", "\t".join(res), flush=True)


def do(remain, set):
    if len(remain) == 0:
        # print(set)
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
    start = {"runs": 5, "sizes": [10200], "AA": "no", "PHISTRUCT": "yes"}

    run(**(merge(start,{"CACHELICM": "yes", "OPTIMIZE": "yes", "FORWARD": "yes", "INLINE": "yes", "NEWCACHE": "no", "MINCUT":"no"})))

    # Fastest version requires newcache off again as cachelicm isn't fully compatible with new cache
    run(**(merge(start,{"CACHELICM": "yes", "OPTIMIZE": "yes", "FORWARD": "no", "INLINE": "yes", "NEWCACHE": "no", "MINCUT":"no"})))

    run(**(merge(start,{"CACHELICM": "no", "OPTIMIZE": "yes", "FORWARD": "no", "INLINE": "yes", "NEWCACHE": "no", "MINCUT":"no"})))

    run(**(merge(start,{"CACHELICM": "no", "OPTIMIZE": "yes", "FORWARD": "no", "INLINE": "no", "NEWCACHE": "no", "MINCUT":"no"})))

    # May run indefinitely, disabled to allow the script to run in reasonable time
    # run(**(merge(start,{"CACHELICM": "no", "OPTIMIZE": "no", "FORWARD": "no", "INLINE": "no", "NEWCACHE": "no", "MINCUT":"no"})))

def scaling():
    start = {"runs": 5, "sizes": list(range(102000, 102000*13, 102000)), "AA": "no", "PHISTRUCT": "yes"}

    run(**(merge(start,{"CACHELICM": "yes", "OPTIMIZE": "yes", "FORWARD": "yes", "INLINE": "yes", "NEWCACHE": "no", "MINCUT":"no"})))

    # Fastest version requires newcache off again as cachelicm isn't fully compatible with new cache
    run(**(merge(start,{"CACHELICM": "yes", "OPTIMIZE": "yes", "FORWARD": "no", "INLINE": "yes", "NEWCACHE": "no", "MINCUT":"no"})))

ablation()
scaling()