#!/usr/bin/env python3
"""Write a hand-rolled NVIDIA Vulkan ICD manifest, then prove it works by enumerating devices.

**Run this on the remote box, not locally** (e.g. `pegasus.py run --script <this file>`) — it
inspects that machine's Vulkan loader and GPUs.

IsaacSim's Omniverse Kit renderer needs a working NVIDIA Vulkan ICD. A fresh box typically has
libvulkan.so.1 and NVIDIA's GL/GLX libraries but *no* nvidia_icd.json under any standard
/usr/share/vulkan/icd.d path — and that directory is root-owned, so you cannot just add one.
NVIDIA ships its Vulkan driver inside libGLX_nvidia.so.0, which *is* present, so a hand-written
ICD JSON plus VK_ICD_FILENAMES / VK_DRIVER_FILES is enough, with no root.

`agents/tools/common/README.md` states that fix in prose; this script is the executable version
plus its verification. It prints device enumeration twice — once with the system ICDs only
(expect zero or llvmpipe) and once with the hand-written manifest (expect the real GPUs) — so a
pass/fail is unambiguous rather than inferred from Kit's own startup log.

Usage:
    python vulkan_icd_probe.py [--icd-dir DIR] [--library PATH]

Defaults target Pegasus (`/data/...`, since almost nothing outside /data has room on that box);
override both on any other machine.
"""
import argparse, ctypes, ctypes.util, json, os, pathlib, sys

VK_SUCCESS = 0


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [("apiVersion", ctypes.c_uint32), ("driverVersion", ctypes.c_uint32),
                ("vendorID", ctypes.c_uint32), ("deviceID", ctypes.c_uint32),
                ("deviceType", ctypes.c_uint32), ("deviceName", ctypes.c_char * 256),
                ("pipelineCacheUUID", ctypes.c_uint8 * 16),
                ("limits", ctypes.c_uint8 * 504), ("sparseProperties", ctypes.c_uint8 * 20)]


DEV_TYPE = {0: "OTHER", 1: "INTEGRATED_GPU", 2: "DISCRETE_GPU", 3: "VIRTUAL_GPU", 4: "CPU"}


def enumerate_devices(label):
    try:
        vk = ctypes.CDLL("libvulkan.so.1")
    except OSError as e:
        print(f"  [{label}] cannot load libvulkan.so.1: {e}")
        return []

    app = VkApplicationInfo(0, None, b"probe", 1, b"none", 1, (1 << 22))  # API 1.0
    ci = VkInstanceCreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = ctypes.c_void_p()
    rc = vk.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    if rc != VK_SUCCESS:
        print(f"  [{label}] vkCreateInstance failed rc={rc} (-9 = incompatible driver)")
        return []

    n = ctypes.c_uint32(0)
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
    if n.value == 0:
        print(f"  [{label}] instance created but 0 physical devices")
        vk.vkDestroyInstance(inst, None)
        return []
    arr = (ctypes.c_void_p * n.value)()
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), arr)

    found = []
    for i in range(n.value):
        props = VkPhysicalDeviceProperties()
        vk.vkGetPhysicalDeviceProperties(arr[i], ctypes.byref(props))
        name = props.deviceName.decode("utf-8", "replace").strip("\x00")
        api = props.apiVersion
        found.append((name, DEV_TYPE.get(props.deviceType, "?"), props.vendorID))
        print(f"  [{label}] device {i}: {name}  type={DEV_TYPE.get(props.deviceType,'?')} "
              f"vendor=0x{props.vendorID:04x} api={api >> 22}.{(api >> 12) & 0x3ff}.{api & 0xfff}")
    vk.vkDestroyInstance(inst, None)
    return found


_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--icd-dir", default="/data/VLA/tingying/vulkan_icd",
                 help="writable dir for the generated manifest (default: %(default)s)")
_ap.add_argument("--library", default="/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
                 help="NVIDIA GLX library carrying the Vulkan driver (default: %(default)s)")
_args = _ap.parse_args()

print("=== 1. baseline: system ICDs only ===")
base = enumerate_devices("system")

print()
print("=== 2. with a hand-written NVIDIA ICD (no root needed) ===")
lib = _args.library
print(f"  ICD library exists: {os.path.exists(lib)}  ({lib})")
if not os.path.exists(lib):
    sys.exit(f"  -> library missing; pass --library with the right path for this box")
icd_dir = pathlib.Path(_args.icd_dir)
icd_dir.mkdir(parents=True, exist_ok=True)
icd = icd_dir / "nvidia_icd.json"
icd.write_text(json.dumps({"file_format_version": "1.0.0",
                           "ICD": {"library_path": lib, "api_version": "1.3.277"}}), encoding="utf-8")
print(f"  wrote {icd}")

# Re-exec so the loader picks the env var up at libvulkan init time.
if os.environ.get("_VK_PROBE_CHILD") != "1":
    env = dict(os.environ)
    env["_VK_PROBE_CHILD"] = "1"
    env["VK_ICD_FILENAMES"] = str(icd)
    env["VK_DRIVER_FILES"] = str(icd)
    env.pop("VK_LOADER_DEBUG", None)
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
