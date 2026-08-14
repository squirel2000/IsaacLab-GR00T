# -*- coding: utf-8 -*-
"""drawio-blocksmith generator (engine copied verbatim from the skill).
DATA below is edited per diagram; run:  python gen_drawio.py --out . --name NAME
Emits NAME.drawio (editable) + NAME.svg (+ NAME.png if cairosvg present), validates layout.
"""
import argparse, html, os, xml.sax.saxutils as sx
import xml.dom.minidom as minidom

# ======================================================================
# ============================  DATA  ==================================
# Selected by --name: "architecture" or "flow".
import sys as _sys
_NAME = "architecture"
for _i, _a in enumerate(_sys.argv):
    if _a == "--name" and _i + 1 < len(_sys.argv):
        _NAME = _sys.argv[_i + 1]

if _NAME == "flow":
    TITLE = "PPO 訓練迴圈（每 epoch）＋ critic warmup 閘門"
    SUBTITLE = "actor 產生動作、critic 估計價值；warmup 期間只更新 critic"
    NODES = [
        dict(id="roll", shape="rounded", x=60, y=110, w=170, h=74, title="1. Rollout",
             label="policy 在 LIBERO\n互動蒐集軌跡", fill="#e8effd", stroke="#2f6fd0", fontsize=13),
        dict(id="adv", shape="rounded", x=280, y=110, w=180, h=74, title="2. 估計 Advantage",
             label="critic 估 V(s)\nGAE 算 A_t", fill="#e8effd", stroke="#2f6fd0", fontsize=13),
        dict(id="upd", shape="rounded", x=510, y=110, w=190, h=74, title="3. PPO 更新",
             label="clip 目標更新 actor\nMSE 更新 critic", fill="#e8effd", stroke="#2f6fd0", fontsize=13),
        dict(id="ev", shape="rounded", x=750, y=110, w=150, h=74, title="4. Eval",
             label="每 10 epochs\nsuccess_once", fill="#e8effd", stroke="#2f6fd0", fontsize=13),
        dict(id="save", shape="cylinder", x=950, y=110, w=150, h=74, title="5. 存檔",
             label="checkpoint", fill="#eaf6ee", stroke="#1f8a4c", fontsize=13),
        dict(id="critic", shape="hexagon", x=300, y=250, w=250, h=86, title="critic / value head",
             label="隨機初始化\n→ 需 warmup", fill="#fde7e7", stroke="#c62828", fontsize=13),
        dict(id="gate", shape="parallelogram", x=610, y=250, w=430, h=86,
             title="critic warmup 閘門 (optimizer_steps < 40)",
             label="warmup 期間凍結 actor、只更新 critic；\n避免用『垃圾 advantage』把 policy 帶壞",
             fill="#fff4e6", stroke="#e8730a", fontsize=13, align="left"),
    ]
    EDGES = [
        ("e1", "roll", "adv", "軌跡", "#2f6fd0", False, True),
        ("e2", "adv", "upd", "A_t", "#2f6fd0", False, True),
        ("e3", "upd", "ev", "", "#2f6fd0", False, True),
        ("e4", "ev", "save", "", "#1f8a4c", False, True),
        ("e5", "save", "roll", "", "#94a3b8", True, True),
        ("e6", "critic", "adv", "V(s) 估計", "#c62828", True, True),
        ("e7", "gate", "upd", "凍結/放行 actor", "#e8730a", True, True),
    ]
    LEGEND = []
    LEGEND_POS = dict(x=0, y=0, w=0, h=0)
else:
    TITLE = "系統架構：本地 ↔ Pegasus 容器 ↔ 單張 H100 (GPU1)"
    SUBTITLE = "RLinf GR00T-N1.7 PPO on LIBERO；GPU0 全程閒置（合作式單卡）"
    NODES = [
        # ---- Local ----
        dict(id="gL", shape="group", x=40, y=90, w=310, h=452, title="本地 (Windows)",
             fill="#eef5ff", stroke="#5b8def", font="#1d4ed8", fontsize=14, align="left"),
        dict(id="launch", shape="rounded", x=60, y=132, w=270, h=54, title="launch.py",
             label="選空閒 GPU→組 Hydra overrides→啟動", fill="#ffffff", stroke="#5b8def", fontsize=12, parent="gL"),
        dict(id="poll", shape="rounded", x=60, y=198, w=270, h=42, title="poll.py",
             label="本地監看", fill="#ffffff", stroke="#5b8def", fontsize=12, parent="gL"),
        dict(id="report", shape="rounded", x=60, y=252, w=270, h=54, title="report.py + extract_metrics",
             label="TB scalars → JSON → HTML", fill="#ffffff", stroke="#5b8def", fontsize=12, parent="gL"),
        dict(id="peg", shape="rect", x=60, y=318, w=270, h=58, title="common/pegasus.py",
             label="Jupyter kernel 遠端執行 / 傳檔", fill="#eef2f7", stroke="#94a3b8", fontsize=12, parent="gL"),
        dict(id="cfg", shape="rect", x=60, y=388, w=270, h=42, title="config/phase1.yaml",
             label="所有旋鈕", fill="#eef2f7", stroke="#94a3b8", fontsize=12, parent="gL"),
        dict(id="patch", shape="rect", x=60, y=442, w=270, h=54, title="patch_rlinf_clone.py",
             label="6 個 RLinf 修補（冪等）", fill="#eef2f7", stroke="#94a3b8", fontsize=12, parent="gL"),
        # ---- Pegasus ----
        dict(id="gP", shape="group", x=430, y=90, w=680, h=452,
             title="Pegasus 容器 pa-jp-v1（cgroup pids.max=2048, EGL, 無 Docker）",
             fill="#eefaf0", stroke="#3fae6a", font="#15803d", fontsize=14, align="left"),
        dict(id="rayhdr", shape="rect", x=450, y=130, w=640, h=26, title="",
             label="Ray cluster（num_cpus=4, dashboard off）", fill="#f0fdf4", stroke="#bbf7d0",
             font="#166534", fontsize=12, parent="gP"),
        dict(id="model", shape="hexagon", x=450, y=162, w=640, h=54, title="GR00T-N1.7 模型 (FSDP, NO_SHARD)",
             label="Cosmos-Reason2-2B backbone ＋ flow-matching action head ＋ value head(隨機初始化)",
             fill="#e6fbff", stroke="#0e9bb5", fontsize=12, parent="gP"),
        dict(id="actor", shape="rounded", x=450, y=228, w=205, h=66, title="ActorGroup @GPU1",
             label="PPO 更新 actor+critic", fill="#fef7db", stroke="#e0a800", fontsize=12, parent="gP"),
        dict(id="rollout", shape="rounded", x=672, y=228, w=200, h=66, title="RolloutGroup @GPU1",
             label="產生 rollout（推論）", fill="#fef7db", stroke="#e0a800", fontsize=12, parent="gP"),
        dict(id="env", shape="rounded", x=888, y=228, w=202, h=66, title="EnvGroup @GPU1",
             label="in-process LIBERO×4", fill="#fef7db", stroke="#e0a800", fontsize=12, parent="gP"),
        dict(id="ckpt", shape="cylinder", x=450, y=316, w=300, h=68, title="checkpoints",
             label="global_step_N", fill="#ffffff", stroke="#3fae6a", fontsize=12, parent="gP"),
        dict(id="tb", shape="cylinder", x=772, y=316, w=318, h=68, title="logs/<ts>/tensorboard",
             label="TB scalars", fill="#ffffff", stroke="#3fae6a", fontsize=12, parent="gP"),
        dict(id="note", shape="rect", x=450, y=404, w=640, h=44, title="",
             label="GPU0 全程閒置；jupyter PID1 無法回收 zombie → 壓低 num_cpus 留 pid 餘裕",
             fill="#f0fdf4", stroke="#bbf7d0", font="#475569", fontsize=12, align="left", parent="gP"),
    ]
    EDGES = [
        ("a1", "peg", "actor", "setsid nohup 啟動", "#5b8def", False, True),
        ("a2", "actor", "rollout", "sync weights", "#e0a800", False, True),
        ("a3", "rollout", "env", "action / obs", "#e0a800", False, True),
        ("a4", "actor", "ckpt", "", "#3fae6a", False, True),
        ("a5", "actor", "tb", "", "#3fae6a", False, True),
        ("a6", "tb", "report", "fetch", "#94a3b8", True, True),
    ]
    LEGEND = []
    LEGEND_POS = dict(x=0, y=0, w=0, h=0)

# ======================================================================
# ==========================  ENGINE  ==================================
INK = "#1b2330"
SHAPE_DEFAULT_FILL = {"group": "#f4f6f9"}
MIN_FONT = 12
MIN_GAP = 8

def _by_id(): return {n["id"]: n for n in NODES}

def _rects_overlap(a, b):
    return not (a["x"]+a["w"] <= b["x"] or b["x"]+b["w"] <= a["x"]
                or a["y"]+a["h"] <= b["y"] or b["y"]+b["h"] <= a["y"])

def _gap(a, b):
    dx = max(0, max(a["x"]-(b["x"]+b["w"]), b["x"]-(a["x"]+a["w"])))
    dy = max(0, max(a["y"]-(b["y"]+b["h"]), b["y"]-(a["y"]+a["h"])))
    return (dx*dx+dy*dy) ** 0.5

def validate():
    errors, warns = [], []
    ids = [n["id"] for n in NODES]
    if len(ids) != len(set(ids)):
        errors.append("duplicate node ids")
    for n in NODES:
        if n.get("shape") == "group":
            continue
        fs = n.get("fontsize", 13)
        if fs < MIN_FONT:
            errors.append(f"node '{n['id']}' fontsize {fs} < {MIN_FONT}pt")
    def nested(a, b):
        return a.get("parent") == b["id"] or b.get("parent") == a["id"]
    for i in range(len(NODES)):
        for j in range(i+1, len(NODES)):
            a, b = NODES[i], NODES[j]
            if nested(a, b):
                continue
            if a.get("shape") == "group" or b.get("shape") == "group":
                grp, oth = (a, b) if a.get("shape") == "group" else (b, a)
                if oth.get("parent") == grp["id"]:
                    continue
            if _rects_overlap(a, b):
                errors.append(f"overlap: '{a['id']}' and '{b['id']}'")
            elif _gap(a, b) < MIN_GAP:
                warns.append(f"tight spacing (<{MIN_GAP}px): '{a['id']}' and '{b['id']}'")
    nid = set(ids)
    for e in EDGES:
        if e[1] not in nid or e[2] not in nid:
            errors.append(f"edge '{e[0]}' references missing node")
    bid = _by_id()
    for e in EDGES:
        if not e[3]:
            continue
        a, b = bid.get(e[1]), bid.get(e[2])
        if not a or not b:
            continue
        lx, ly = (a["x"]+a["w"]/2+b["x"]+b["w"]/2)/2, (a["y"]+a["h"]/2+b["y"]+b["h"]/2)/2
        for n in NODES:
            if n["id"] in (e[1], e[2]) or n.get("shape") == "group":
                continue
            if n["x"] <= lx <= n["x"]+n["w"] and n["y"] <= ly <= n["y"]+n["h"]:
                warns.append(f"edge '{e[0]}' label may overlap node '{n['id']}'")
    return errors, warns

def _xe(t): return sx.escape(t, {'"': "&quot;"})

def _label_html(n):
    parts = []
    if n.get("title"):
        parts.append(f'<b>{html.escape(n["title"])}</b>')
    if n.get("label"):
        parts.append(html.escape(n["label"]).replace("\n", "<br/>"))
    return "<br/>".join(parts)

def _drawio_style(n):
    shape = n.get("shape", "rounded")
    fill = n.get("fill", SHAPE_DEFAULT_FILL.get(shape, "#ffffff"))
    stroke = n.get("stroke", "#5b6573")
    font = n.get("font", INK)
    fs = n.get("fontsize", 13)
    align = n.get("align", "center")
    base = (f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontColor={font};fontSize={fs};align={align};verticalAlign=middle;"
            f"spacingLeft=6;spacingRight=6;")
    smap = {
        "rounded": "rounded=1;arcSize=8;",
        "rect": "rounded=0;",
        "ellipse": "ellipse;",
        "diamond": "rhombus;",
        "hexagon": "shape=hexagon;",
        "cylinder": "shape=cylinder3;boundedLbl=1;",
        "parallelogram": "shape=parallelogram;",
        "group": "rounded=1;arcSize=4;container=1;collapsible=0;verticalAlign=top;dashed=1;",
    }
    return smap.get(shape, "rounded=1;") + base

def build_drawio(pw, ph):
    bid = _by_id()
    cells = []
    for n in NODES:
        parent = n.get("parent", "1")
        x, y = n["x"], n["y"]
        if parent != "1" and parent in bid:
            x -= bid[parent]["x"]; y -= bid[parent]["y"]
        val = _label_html(n)
        cells.append(
            f'<mxCell id="{n["id"]}" value="{_xe(val)}" style="{_drawio_style(n)}" '
            f'vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" '
            f'width="{n["w"]}" height="{n["h"]}" as="geometry"/></mxCell>')
    for eid, s, t, label, color, dashed, arrow in EDGES:
        col = color or "#5b6573"
        d = "dashed=1;" if dashed else ""
        ar = "endArrow=block;" if arrow else "endArrow=none;"
        cells.append(
            f'<mxCell id="{eid}" value="{_xe(label or "")}" style="{ar}rounded=1;html=1;'
            f'strokeColor={col};strokeWidth=2;{d}edgeStyle=orthogonalEdgeStyle;fontSize=12;'
            f'fontColor={col};labelBackgroundColor=#ffffff;" edge="1" parent="1" '
            f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    if TITLE:
        th = f'<font style="font-size:20px"><b>{html.escape(TITLE)}</b></font>'
        if SUBTITLE:
            th += f'<br/><font color="#56616f">{html.escape(SUBTITLE)}</font>'
        cells.append(f'<mxCell id="__title" value="{_xe(th)}" style="text;html=1;align=left;'
                     f'verticalAlign=top;strokeColor=none;fillColor=none;whiteSpace=wrap;" vertex="1" '
                     f'parent="1"><mxGeometry x="40" y="18" width="900" height="46" as="geometry"/></mxCell>')
    body = "\n".join(cells)
    return (f'<mxfile host="app.diagrams.net" version="24.0.0"><diagram name="diagram" id="d1">'
            f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{pw}" pageHeight="{ph}" math="0" shadow="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')

def _se(t): return html.escape(t, quote=True)

def _shape_svg(n):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    fill = n.get("fill", SHAPE_DEFAULT_FILL.get(n.get("shape"), "#ffffff"))
    stroke = n.get("stroke", "#5b6573")
    sh = n.get("shape", "rounded")
    if sh in ("rounded", "group"):
        rx = 10 if sh == "rounded" else 8
        dash = ' stroke-dasharray="6 5"' if sh == "group" else ''
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
    if sh == "rect":
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sh == "ellipse":
        return f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sh == "diamond":
        pts = f"{x+w/2},{y} {x+w},{y+h/2} {x+w/2},{y+h} {x},{y+h/2}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sh == "hexagon":
        i = w*0.06
        pts = f"{x+i},{y} {x+w-i},{y} {x+w},{y+h/2} {x+w-i},{y+h} {x+i},{y+h} {x},{y+h/2}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sh == "parallelogram":
        s = w*0.10
        pts = f"{x+s},{y} {x+w},{y} {x+w-s},{y+h} {x},{y+h}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sh == "cylinder":
        ry = min(12, h*0.16)
        return (f'<path d="M{x} {y+ry} a{w/2} {ry} 0 0 1 {w} 0 v{h-2*ry} a{w/2} {ry} 0 0 1 -{w} 0 z" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
                f'<path d="M{x} {y+ry} a{w/2} {ry} 0 0 0 {w} 0" fill="none" stroke="{stroke}" stroke-width="1.6"/>')
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'

def _text_svg(n):
    if not (n.get("title") or n.get("label")):
        return ""
    font = n.get("font", INK); fs = n.get("fontsize", 13)
    if n.get("shape") == "group":
        t = n.get("title") or n.get("label") or ""
        return (f'<text x="{n["x"]+12}" y="{n["y"]+fs+8}" font-size="{fs}" font-weight="700" '
                f'fill="{font}" text-anchor="start" font-family="Arial">{_se(t)}</text>')
    align = n.get("align", "center")
    anchor = {"left": "start", "center": "middle", "right": "end"}[align]
    if align == "left":
        tx = n["x"]+10
    elif align == "right":
        tx = n["x"]+n["w"]-10
    else:
        tx = n["x"]+n["w"]/2
    lines = []
    if n.get("title"):
        lines.append((n["title"], fs+2, "700"))
    for ln in (n.get("label") or "").split("\n"):
        if ln != "":
            lines.append((ln, fs, "400"))
    total = sum(s for _, s, _ in lines) + (len(lines)-1)*4
    cy = n["y"]+n["h"]/2 - total/2 + lines[0][1] if lines else n["y"]
    out = []
    yy = cy
    for txt, sz, wgt in lines:
        out.append(f'<text x="{tx}" y="{yy}" font-size="{sz}" font-weight="{wgt}" fill="{font}" '
                   f'text-anchor="{anchor}" font-family="Arial">{_se(txt)}</text>')
        yy += sz + 4
    return "\n".join(out)

def _marker(c):
    cid = c.strip("#")
    return (f'<marker id="m_{cid}" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto">'
            f'<path d="M0 0 L9 4 L0 8 z" fill="{c}"/></marker>')

def _edge_svg(e):
    bid = _by_id()
    eid, s, t, label, color, dashed, arrow = e
    a, b = bid.get(s), bid.get(t)
    if not a or not b:
        return ""
    col = color or "#5b6573"
    ax, ay = a["x"]+a["w"]/2, a["y"]+a["h"]/2
    bx, by = b["x"]+b["w"]/2, b["y"]+b["h"]/2
    if abs(by-ay) >= abs(bx-ax):
        y1 = a["y"]+a["h"] if by > ay else a["y"]
        y2 = b["y"] if by > ay else b["y"]+b["h"]
        midy = (y1+y2)/2
        d = f"M{ax} {y1} V{midy} H{bx} V{y2}"
        lx, ly = (ax+bx)/2, midy
    else:
        x1 = a["x"]+a["w"] if bx > ax else a["x"]
        x2 = b["x"] if bx > ax else b["x"]+b["w"]
        midx = (x1+x2)/2
        d = f"M{x1} {ay} H{midx} V{by} H{x2}"
        lx, ly = midx, (ay+by)/2
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    end = f' marker-end="url(#m_{col.strip("#")})"' if arrow else ''
    out = [f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"{dash}{end}/>']
    if label:
        w = len(label)*7.4+14
        out.append(f'<rect x="{lx-w/2}" y="{ly-10}" width="{w}" height="20" rx="9" fill="#fff" stroke="{col}"/>')
        out.append(f'<text x="{lx}" y="{ly+4}" font-size="12" font-weight="600" fill="{col}" '
                   f'text-anchor="middle" font-family="Arial">{_se(label)}</text>')
    return "\n".join(out)

def build_svg(pw, ph):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}" font-family="Arial">']
    cols = set((e[4] or "#5b6573") for e in EDGES)
    o.append("<defs>" + "".join(_marker(c) for c in cols) + "</defs>")
    o.append(f'<rect width="{pw}" height="{ph}" fill="#ffffff"/>')
    if TITLE:
        o.append(f'<text x="40" y="34" font-size="19" font-weight="800" fill="{INK}" font-family="Arial">{_se(TITLE)}</text>')
        if SUBTITLE:
            o.append(f'<text x="40" y="54" font-size="12.5" fill="#56616f" font-family="Arial">{_se(SUBTITLE)}</text>')
    for n in NODES:
        if n.get("shape") == "group":
            o.append(_shape_svg(n)); o.append(_text_svg(n))
    for e in EDGES:
        o.append(_edge_svg(e))
    for n in NODES:
        if n.get("shape") != "group":
            o.append(_shape_svg(n)); o.append(_text_svg(n))
    o.append("</svg>")
    return "\n".join(o)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--name", default="architecture")
    ap.add_argument("--png-width", type=int, default=1700)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    errors, warns = validate()
    for w in warns:
        print("WARN:", w)
    if errors:
        print("VALIDATION: FAIL")
        for e in errors:
            print("  ERROR:", e)
        raise SystemExit(1)
    print("VALIDATION: PASS")
    pad = 40
    pw = max([n["x"]+n["w"] for n in NODES]) + pad
    ph = max([n["y"]+n["h"] for n in NODES]) + pad
    dpath = os.path.join(a.out, a.name + ".drawio")
    xml = build_drawio(pw, ph)
    minidom.parseString(xml)
    open(dpath, "w", encoding="utf-8").write(xml)
    print("wrote", dpath)
    spath = os.path.join(a.out, a.name + ".svg")
    open(spath, "w", encoding="utf-8").write(build_svg(pw, ph))
    print("wrote", spath)
    try:
        import cairosvg
        ppath = os.path.join(a.out, a.name + ".png")
        cairosvg.svg2png(url=spath, write_to=ppath, output_width=a.png_width)
        print("wrote", ppath)
    except Exception as e:
        print("PNG step skipped (no cairosvg):", type(e).__name__)

if __name__ == "__main__":
    main()
