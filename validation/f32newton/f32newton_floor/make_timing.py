#!/usr/bin/env python3
"""Timing copies (C = web/index.html, gated fix; N = same with F32N_NOFIX; A/B = first version app2d_fixA.html) (hook only: window.DIAG_FULL = true renders the frame with the full/readback root path).
  timing_before.html : app2d before this change (app2d_before.html)
  timing_A.html      : web/index.html (fix on every path)
  timing_B.html      : web/index.html with the fix disabled on the display path (g_fast)"""
import os
H = os.path.dirname(os.path.abspath(__file__))
HOOK_OLD = "            if (glProg !== glProgV6 && glProg !== glProgV6K && glProg !== glProgV6F && glProg !== glProgV6FK) return;"
HOOK_NEW = HOOK_OLD + "\n            if (window.DIAG_FULL) readback = true;"
def build(src, out, extra=()):
    s = open(src).read()
    assert s.count(HOOK_OLD) == 1
    s = s.replace(HOOK_OLD, HOOK_NEW)
    for a, b in extra:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    open(os.path.join(H, out), 'w').write(s)
app = os.path.join(H, '..', '..', '..', 'web', 'index.html')
build(os.path.join(H, 'app2d_before.html'), 'timing_before.html')
build(os.path.join(H, 'app2d_fixA.html'), 'timing_A.html')
build(app, 'timing_C.html')
build(app, 'timing_N.html', [("#ifdef F32N_NOFIX", "#if 1")])
build(os.path.join(H, 'app2d_fixA.html'), 'timing_B.html', [("  FixC X = fixConsts(c, dLo, lLo);", "  FixC X; X.on = false;\n  if (!g_fast) X = fixConsts(c, dLo, lLo);")])
print('ok')
