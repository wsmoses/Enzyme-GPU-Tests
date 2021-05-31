import os
import subprocess 

DEVICE=os.getenv("DEVICE", "1")

os.system("echo \"using Pkg; Pkg.instantiate()\" | julia --project=.")

CUDA_PATH=os.getenv("CUDA_PATH", "/usr/local/cuda-11.2")

p = subprocess.run(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} {CUDA_PATH}/bin/nsys launch julia --project=. ./perf.jl", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert(p.returncode == 0)

print("Scalability tests")
print(open("profile_4.csv").read())

p = subprocess.run(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} UNROLLING=false COALESE=true {CUDA_PATH}/bin/nsys launch julia --project=. ./perf.jl", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert(p.returncode == 0)

print("Unrolling disabled, coalsce enabled")
print(open("profile_4.csv").read())


p = subprocess.run(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} UNROLLING=false COALESE=false {CUDA_PATH}/bin/nsys launch julia --project=. ./perf.jl", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert(p.returncode == 0)

print("Unrolling disabled, coalsce disabled")
print(open("profile_4.csv").read())

print("Verification")
subprocess.run(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={DEVICE} UNROLLING=false COALESE=false {CUDA_PATH}/bin/nsys launch julia --project=. ./verify.jl", shell=True)