#!/usr/bin/env python3
"""Post-link fix for wasm-ld's command model: every export is wrapped in
`<name>.command_export`, which re-runs C++ static constructors/destructors on
each JS->wasm call. Keep the ctors call only in `main`'s wrapper (it runs once,
first, from the start function); turn every other ctors/dtors call into nops of
the same length so no section size changes."""
import sys

def leb(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7f) << s; s += 7
        if not x & 0x80: return r, i

def enc(n):
    out = bytearray()
    while True:
        x = n & 0x7f; n >>= 7
        out.append(x | (0x80 if n else 0))
        if not n: return bytes(out)

def main(inp, outp, dry=False):
    b = bytearray(open(inp, 'rb').read())
    i = 8; names = {}; code = None
    while i < len(b):
        sid = b[i]; size, j = leb(b, i + 1); end = j + size
        if sid == 10: code = (j, end)
        if sid == 0:
            nl, k = leb(b, j); nm = bytes(b[k:k + nl]).decode(); k += nl
            if nm == 'name':
                while k < end:
                    sub = b[k]; ss, k2 = leb(b, k + 1)
                    if sub == 1:
                        cnt, p = leb(b, k2)
                        for _ in range(cnt):
                            idx, p = leb(b, p); l, p = leb(b, p)
                            names[idx] = bytes(b[p:p + l]).decode(errors='replace'); p += l
                    k = k2 + ss
        i = end
    by = {v: k for k, v in names.items()}
    ctors, dtors = by.get('__wasm_call_ctors'), by.get('__wasm_call_dtors')
    print('ctors idx', ctors, 'dtors idx', dtors, 'names', len(names))
    if ctors is None: sys.exit('no __wasm_call_ctors in name section')
    # number of imported functions = first defined index
    imports_funcs = 0
    i = 8
    while i < len(b):
        sid = b[i]; size, j = leb(b, i + 1)
        if sid == 2:
            n, p = leb(b, j)
            for _ in range(n):
                l, p = leb(b, p); p += l; l, p = leb(b, p); p += l
                kind = b[p]; p += 1
                if kind == 0: _, p = leb(b, p); imports_funcs += 1
                elif kind == 1: p += 1; f = b[p]; p += 1; _, p = leb(b, p); 
                elif kind == 2: f = b[p]; p += 1; _, p = leb(b, p); (_, p) = leb(b, p) if f & 1 else (0, p)
                elif kind == 3: p += 2
                elif kind == 4: p += 1; _, p = leb(b, p)
        i = j + size
    cs, ce = code
    n, p = leb(b, cs)
    pc = b'\x10' + enc(ctors); pd = b'\x10' + enc(dtors) if dtors is not None else None
    patched = 0
    for fi in range(n):
        size, q = leb(b, p); body_end = q + size
        idx = imports_funcs + fi; nm = names.get(idx, '')
        if nm.endswith('.command_export') or nm.endswith('command_export'):
            body = bytes(b[q:body_end])
            keep_ctors = nm.startswith('main.')
            for pat, keep in ((pc, keep_ctors), (pd, False)):
                if pat and pat in body and not keep:
                    off = q + body.index(pat)
                    b[off:off + len(pat)] = b'\x01' * len(pat); patched += 1
            print(f'{nm}: {body.hex()[:90]}')
        p = body_end
    print('patched calls:', patched)
    if not dry: open(outp, 'wb').write(b)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, dry=len(sys.argv) < 3)
