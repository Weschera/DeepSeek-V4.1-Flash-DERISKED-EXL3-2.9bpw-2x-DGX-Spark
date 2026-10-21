#!/usr/bin/env python3
"""Make a Blackfrost EXL3 tree (from our patched exllamav3 convert) loadable by Mia's DSv4.1 EXL3 kit.
Repairs (same five found on the 2.9bpw tree):
  1. config.json -> nested text_config layout (from config.json.orig-nested), exl3 quantization_config with avg_bits
  2. graft Engram (L1/L14) + ratio-1 indexer (L20..36) tensors missing vs the 1017 donor EXL3 (identical upstream weights)
  3. strip unused *.attn.compressor.ape / *.ffn.gate.bias_vl tensors from the shards (kit's loader has no mapping)
  4. replace CSA compressor modules whose wkv/wgate were quantized at the padded 1024 width with the donor's 512-wide modules
  5. verify: no leftovers, index == shard headers, shapes match donor except expert K (bpw allocation)
Usage: fix_blackfrost_tree.py <tree> <donor_tree> <engram_dir_for_config>
"""
import json, struct, os, re, sys, shutil, collections

D, MIA, ENG = sys.argv[1:4]
GRAFT1, GRAFT2 = "model-graft-1017.safetensors", "model-graft-compressor.safetensors"

def hdr(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n)), 8 + n

def rewrite_shard(path, drop):
    h, base = hdr(path); meta = h.pop("__metadata__", None)
    keep = sorted([(k, v) for k, v in h.items() if k not in drop], key=lambda kv: kv[1]["data_offsets"][0])
    nh = {}; off = 0
    for k, v in keep:
        a, b = v["data_offsets"]; nh[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + (b - a)]}; off += b - a
    if meta: nh["__metadata__"] = meta
    hb = json.dumps(nh, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path + ".tmp"
    with open(path, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for k, v in keep:
            a, b = v["data_offsets"]; fi.seek(base + a); rem = b - a
            while rem:
                c = fi.read(min(rem, 64 << 20)); fo.write(c); rem -= len(c)
    os.replace(tmp, path)

def write_graft(path, src_map, keys):
    """copy tensors `keys` (name -> shard file in src tree) into one new safetensors file"""
    blobs = []
    for f in sorted({src_map[k] for k in keys}):
        h, base = hdr(f)
        with open(f, "rb") as fh:
            for k in keys:
                if src_map[k] != f: continue
                a, b = h[k]["data_offsets"]; fh.seek(base + a); blobs.append((k, h[k]["dtype"], h[k]["shape"], fh.read(b - a)))
    hj = {}; off = 0
    for k, dt, sh, bl in blobs: hj[k] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(bl)]}; off += len(bl)
    hb = json.dumps(hj, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for *_, bl in blobs: fo.write(bl)
    return off

idx_p = f"{D}/model.safetensors.index.json"
if not os.path.exists(idx_p + ".orig"): shutil.copy(idx_p, idx_p + ".orig")
idx = json.load(open(idx_p)); wm = idx["weight_map"]
mia = json.load(open(f"{MIA}/model.safetensors.index.json"))["weight_map"]
mia_abs = {k: f"{MIA}/{v}" for k, v in mia.items()}
skip_prefix = ("vision", "aligner", "image", "projector")

# ---- 1. config
if not os.path.exists(f"{D}/config.json.flat-exl3"): shutil.copy(f"{D}/config.json", f"{D}/config.json.flat-exl3")
flat = json.load(open(f"{D}/config.json.flat-exl3")); nested = json.load(open(f"{D}/config.json.orig-nested"))
mcfg = json.load(open(f"{MIA}/config.json"))
qc = dict(flat["quantization_config"]);
if "bits" in qc: qc["avg_bits"] = qc.pop("bits")
c = dict(nested); c["quantization_config"] = qc; c["engram_table_dir"] = mcfg.get("engram_table_dir")
diffs = [k for k in set(c["text_config"]) | set(mcfg["text_config"]) if c["text_config"].get(k) != mcfg["text_config"].get(k)]
json.dump(c, open(f"{D}/config.json", "w"), indent=2)
print("1. config rewritten; text_config diffs vs donor:", diffs)

# ---- 2. graft missing Engram / indexer keys
missing = [k for k in mia if k not in wm and not k.startswith(skip_prefix) and not k.endswith(".bias_vl")]
mods = sorted({re.sub(r"\.(suh|svh|mul1|trellis|weight|bias|scale|fn|base)$", "", k) for k in missing})
print("2. missing modules vs donor:", len(mods), mods[:12])
if missing:
    nbytes = write_graft(f"{D}/{GRAFT1}", mia_abs, missing)
    for k in missing: wm[k] = GRAFT1
    print(f"   grafted {len(missing)} tensors, {nbytes/1e9:.2f} GB")

# ---- 3. strip ape / bias_vl
drop = {k for k in wm if k.endswith(".attn.compressor.ape") or k.endswith(".ffn.gate.bias_vl") or k.endswith(".indexer.compressor.ape")}
by = collections.defaultdict(set)
for k in drop: by[wm[k]].add(k)
print("3. stripping", len(drop), "tensors from", len(by), "shards")
for f, ks in sorted(by.items()):
    if f.startswith("model-graft"): continue
    rewrite_shard(f"{D}/{f}", ks); print("   ", f, "-", len(ks))
for k in drop: wm.pop(k, None)

# ---- 4. padded compressor modules -> donor
bad = set()
for k in list(wm):
    if re.search(r"attn\.(compressor|indexer\.compressor)\.(wkv|wgate)\.svh$", k) and not wm[k].startswith("model-graft"):
        h, _ = hdr(f"{D}/{wm[k]}")
        if h[k]["shape"][0] != 512: bad.add(k.rsplit(".", 1)[0])
need = [k for k in mia if k.rsplit(".", 1)[0] in bad]
print("4. padded compressor modules:", sorted(bad), "->", len(need), "donor tensors")
if need:
    by = collections.defaultdict(set)
    for k in need:
        if k in wm and not wm[k].startswith("model-graft"): by[wm[k]].add(k)
    for f, ks in sorted(by.items()): rewrite_shard(f"{D}/{f}", ks); print("   ", f, "-", len(ks))
    nbytes = write_graft(f"{D}/{GRAFT2}", mia_abs, need)
    for k in need: wm[k] = GRAFT2
    print(f"   grafted {len(need)} tensors, {nbytes/1e6:.1f} MB")

json.dump(idx, open(idx_p, "w"), indent=2)

# ---- 5. verify
shapes = {}
for f in sorted(set(wm.values())):
    h, _ = hdr(f"{D}/{f}")
    for k, v in h.items():
        if k != "__metadata__": shapes[k] = (tuple(v["shape"]), v["dtype"], f)
left = [k for k in shapes if k.endswith((".compressor.ape", ".bias_vl"))]
miss = [k for k in wm if k not in shapes or shapes[k][2] != wm[k]]
extra = [k for k in shapes if k not in wm]
mshapes = {}
for f in sorted(set(mia.values())):
    h, _ = hdr(f"{MIA}/{f}")
    for k, v in h.items():
        if k != "__metadata__": mshapes[k] = (tuple(v["shape"]), v["dtype"])
mism = [k for k in shapes if k in mshapes and shapes[k][:2] != mshapes[k] and ".experts." not in k]
notin = sorted({re.sub(r"\.\d+\.", ".N.", k) for k in mshapes if k not in shapes and not k.startswith(skip_prefix) and not k.endswith(".bias_vl")})
print("5. verify: leftovers", len(left), "| index/shard mismatch", len(miss), "| in shards not index", len(extra),
      "| non-expert shape mismatch vs donor", len(mism), mism[:5], "| donor keys still absent", notin[:8])
print("index keys:", len(wm), "| OK" if not (left or miss or extra or mism or notin) else "| CHECK ABOVE")
