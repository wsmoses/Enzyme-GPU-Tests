import os
import subprocess 

DEVICE=os.getenv("DEVICE", "5")

sizes = [102, 1020, 10200, 102000, 1020000, 10200000]

sizes = list(range(100, 37000, 100))
# AA broken here
vars = ["OPTIMIZE", "NEWCACHE", "FORWARD", "ALLOCATOR", "ABI"]

def run(VERIFY, OPTIMIZE, FORWARD, NEWCACHE, ALLOCATOR, ABI, runs, sizes):
    print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} ABI={ABI} ALLOCATOR={ALLOCATOR} NEWCACHE={NEWCACHE} make -B -j', flush=True)
    comp = subprocess.run(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} ABI={ABI} FORWARD={FORWARD} ALLOCATOR={ALLOCATOR} NEWCACHE={NEWCACHE} make -B -j', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert (comp.returncode == 0)
    
    for size in sizes:
        res = []
        for i in range(runs):
            if VERIFY == "yes":
                res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} ./rsbench -i datasets/lbm/short/input/120_120_150_ldc.of -o ref.dat -- " + str(size) + "| grep \"der=[0-9\.]*\" -o | grep -e \"[0-9\.]*\" -o").read().strip())
            else:
                res.append(os.popen(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} ./rsbench -i datasets/lbm/short/input/120_120_150_ldc.of -o ref.dat -- " + str(size) + "| grep \"Kernel   \" | grep -e \"[0-9\.]*\" -o").read().strip())
            # print(res, flush=True)
        print(f'VERIFY={VERIFY} OPTIMIZE={OPTIMIZE} FORWARD={FORWARD} ABI={ABI} ALLOCATOR={ALLOCATOR} NEWCACHE={NEWCACHE} size={size}', "\t", "\t".join(res), flush=True)


def do(remain, set):
    if len(remain) == 0:
        # print(set)
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

# do(vars[2:], {"runs":5, "AA":"no", "NEWCACHE": "no"})
# for s in range(40, 500, 40):
# for s in range(50, 1000, 50):
#    do(vars[1:-1], {"OPTIMIZE":"yes", "ABI":"yes", "runs":5, "size": s})

def merge(a, b):
    c = {}
    for m in a:
        c[m] = a[m]
    for m in b:
        c[m] = b[m]
    return c

def ablation():
    start = {"runs": 5, "sizes": [150], "OPTIMIZE": "yes", "VERIFY": "no"}
    run(**(merge(start,{"FORWARD":"yes", "NEWCACHE":"yes", "ALLOCATOR":"yes", "ABI":"yes"})))
    run(**(merge(start,{"FORWARD":"no", "NEWCACHE":"yes", "ALLOCATOR":"yes", "ABI":"yes"})))
    run(**(merge(start,{"FORWARD":"no", "NEWCACHE":"yes", "ALLOCATOR":"no", "ABI":"yes"})))
    # run(**(merge(start,{"FORWARD":"no", "NEWCACHE":"no", "ALLOCATOR":"no", "ABI":"no"})))

def scaling():
    start = {"runs": 5, "sizes": list(range(50,650,50)), "OPTIMIZE": "yes", "VERIFY":"no"}
    run(**(merge(start,{"FORWARD":"yes", "NEWCACHE":"yes", "ALLOCATOR":"yes", "ABI":"yes"})))
    run(**(merge(start,{"FORWARD":"no", "NEWCACHE":"yes", "ALLOCATOR":"yes", "ABI":"yes"})))

def verify():
    start = {"runs": 1, "sizes": [20], "OPTIMIZE": "yes", "VERIFY":"yes"}
    run(**(merge(start,{"FORWARD":"yes", "NEWCACHE":"yes", "ALLOCATOR":"yes", "ABI":"yes"})))
    run(**(merge(start,{"FORWARD":"no", "NEWCACHE":"yes", "ALLOCATOR":"yes", "ABI":"yes"})))

ablation()
scaling()
verify()

