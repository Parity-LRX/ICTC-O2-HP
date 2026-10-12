"""Publication figures from completed 0.1.5 measurements; sbatch only."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import TriMesh
from matplotlib.tri import Triangulation
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from PIL import Image

CHANNELS = (32, 64, 96, 128, 256)
SIZES = (16384, 32768, 65536, 131072)
ORDERS = tuple(range(1, 7))
PARITIES = ("canonical", "full")
FIELDS = {"allocated": "peak_allocated_MiB", "reserved": "peak_reserved_MiB",
          "process": "sampled_process_peak_MiB"}
INK = "#25343C"
TEAL = "#176F82"
CORAL = "#C86C55"
PALETTE = LinearSegmentedColormap.from_list("publication_balance", ["#C86C55", "#FBFCFC", "#176F82"])
SURFACE_PALETTE = LinearSegmentedColormap.from_list("publication_surface", ["#C86C55", "#DDE5E8", "#176F82"], N=1024)
ABS_PALETTE = LinearSegmentedColormap.from_list("publication_absolute", ["#F4F8FA", "#ACCBD3", "#176F82"])
AUDIT = []
SURFACE_AUDIT = []
CONNECTED_AUDIT = []
SURFACE_SUBDIVISIONS = 16
FIGURE_COMPARISON = "ICTC-O2 HP vs eqx"
FIGURE_HEADERS = {}
LEGEND_AUDIT = []
OOM_MARKER_AUDIT = []
PARITY_HEADER_AUDIT = []
SMOOTH_AUDIT = []


def load_json(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#76878F",
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .55,
        "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans"})


def collect_ready(root, strict):
    base = root
    rows, pending, gates = [], [], {}
    for c in CHANNELS:
        folder = base / f"C{c}"
        for m in ORDERS:
            for p in PARITIES:
                qpath = folder / f"qualification-C{c}-m{m}-{p}-N131072.json"
                q = load_json(qpath)
                qualified = (q.get("status") == "ok" and q.get("package_version") == "0.1.5"
                             and q.get("every_row_checked") and q.get("every_weight_checked")
                             and max(q.get("candidate_errors", {"missing": 1}).values()) <= 1e-5)
                if qualified: gates[c, m, p] = str(qpath)
                for n in SIZES:
                    suffix = f"C{c}-m{m}-{p}-N{n}"
                    paths = {"paired": folder/f"paired-candidate-equivariantx-{suffix}.json"}
                    paths.update({name: folder/f"memory-{name}-{suffix}.json" for name in ("candidate", "equivariantx")})
                    raw = {name: load_json(path) for name, path in paths.items()}
                    ready = qualified and all(r.get("status") in ("ok", "oom") and r.get("package_version") == "0.1.5" for r in raw.values())
                    if not ready:
                        pending.append(dict(N=n, C=c, m_max=m, parity=p)); continue
                    assert all(r.get("dtype") == "FP32" and r.get("tf32") is False and r.get("amp") is False for r in raw.values())
                    assert all((r["edges"],r["channels"],r["order"],r["parity"]) == (n,c,m,p) for r in raw.values())
                    assert len({r["gpu_uuid"] for r in raw.values()}) == 1
                    for mode in ("forward", "forward_vjp"):
                        pair = raw["paired"].get("timing", {}).get(mode, {})
                        memories = {name: raw[name].get("timing", {}).get(mode, {}).get(name, {}) for name in ("candidate", "equivariantx")}
                        paired = "candidate" in pair and "equivariantx" in pair
                        a, b = (pair["candidate"], pair["equivariantx"]) if paired else (memories["candidate"], memories["equivariantx"])
                        assert a, (suffix, mode, "candidate failure")
                        row = dict(N=n, C=c, m_max=m, parity=p, mode=mode, source_version="0.1.5",
                                   paired_complete=paired, throughput_ratio=b["median_ms"]/a["median_ms"] if b else None,
                                   paired_IQR_relation=("overlap" if paired and not (a["q75_ms"]<b["q25_ms"] or b["q75_ms"]<a["q25_ms"]) else "separated" if paired else "isolated"),
                                   qualification_source=str(qpath), timing_source=str(paths["paired"]), gpu_uuid=raw["paired"]["gpu_uuid"])
                        for name, timing in (("candidate", a), ("equivariantx", b)):
                            row[name+"_status"] = "ok" if memories[name] else raw[name]["status"]
                            row[name+"_ms"] = timing.get("median_ms")
                            row[name+"_rows_per_second"] = timing.get("rows_per_second")
                            row[name+"_memory_source"] = str(paths[name])
                            for field in FIELDS.values(): row[name+"_"+field] = memories[name].get(field)
                        for quantity, field in FIELDS.items():
                            ours, eqx = row["candidate_"+field], row["equivariantx_"+field]
                            row[quantity+"_memory_ratio"] = eqx/ours if ours is not None and eqx is not None else None
                        rows.append(row)
    if strict:
        assert not pending and len(rows) == 480 and len(gates) == 60
    assert rows
    return rows, pending, gates


def value(row, quantity):
    if row is None: return None
    if quantity == "throughput": return row["throughput_ratio"]
    a, b = row["candidate_"+FIELDS[quantity]], row["equivariantx_"+FIELDS[quantity]]
    return b/a if a is not None and b is not None else None


def scale(rows, mode, quantity):
    data = [value(r, quantity) for r in rows if r["mode"] == mode]
    data = [x for x in data if x is not None]
    return TwoSlopeNorm(vmin=min(.85,min(data)), vcenter=1, vmax=max(1.65,max(data)))


def label(row, quantity, number):
    if number is None:
        return "PENDING" if row is None else "EQX\nOOM"
    text = f"{number:.2f}×"
    if quantity == "throughput":
        text += "*" if not row["paired_complete"] else "†" if row["paired_IQR_relation"] == "overlap" else ""
    elif number < 1 and f"{number:.2f}" == "1.00":
        text += "‡"
    return text


def metric_formatter(quantity):
    return FuncFormatter(lambda v, _: f"{v:g}×")


def sparse_ratio_legend(cb,norm,quantity):
    positive=[1,2,3] if norm.vmax>=3 else [1,2,2.5] if norm.vmax>=2.5 else [1,1.5,2] if norm.vmax>=2 else [1,1.5]
    ticks=([.9] if quantity=="throughput" else [])+positive
    ticks=[v for v in ticks if norm.vmin<=v<=norm.vmax]
    cb.set_ticks(ticks);cb.formatter=metric_formatter(quantity);cb.update_ticks()
    LEGEND_AUDIT.append(dict(quantity=quantity,ticks=ticks))


def grid(rows, levels, p, mode, quantity, *, n=None, c=None):
    lookup = {(r["N"],r["C"],r["m_max"],r["parity"],r["mode"]): r for r in rows}
    z = np.full((len(levels),len(ORDERS)),np.nan)
    labels = np.empty(z.shape,dtype=object)
    for yi, level in enumerate(levels):
        for xi, m in enumerate(ORDERS):
            nn, cc = (n,level) if n is not None else (level,c)
            row = lookup.get((nn,cc,m,p,mode)); number = value(row,quantity)
            if number is not None: z[yi,xi]=number
            labels[yi,xi]=label(row,quantity,number)
            AUDIT.append(dict(N=nn,C=cc,m_max=m,parity=p,mode=mode,quantity=quantity,value=number,label=labels[yi,xi],
                              source=row["timing_source"] if row else None, state="measured" if number is not None else "oom" if row else "pending"))
    return z, labels


def heatmap(ax, z, labels, levels, norm, ylabel, cmap=PALETTE):
    cm=cmap.copy();cm.set_bad("#F2F4F5")
    im=ax.imshow(z,origin="lower",aspect="auto",cmap=cm,norm=norm,interpolation="nearest")
    ax.set_xticks(range(6),ORDERS);ax.set_yticks(range(len(levels)),levels)
    ax.set_xlabel(r"Maximum frequency, $m_{\max}$",labelpad=5);ax.set_ylabel(ylabel,labelpad=6)
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,len(levels),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=1.5);ax.tick_params(which="minor",length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for yi,xi in np.ndindex(z.shape):
        text=labels[yi,xi];num=z[yi,xi]
        rgba=im.cmap(im.norm(num)) if np.isfinite(num) else (1,1,1,1)
        luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
        ax.text(xi,yi,text,ha="center",va="center",fontsize=8.3 if np.isfinite(num) else 6.2,
                color="white" if luminance<.48 else INK if text!="PENDING" else "#9CA8AE")
        if "OOM" in text:
            ax.add_patch(Rectangle((xi-.5,yi-.5),1,1,facecolor="none",edgecolor="#AEB9BD",hatch="///",lw=0))
        elif "‡" in text:
            ax.plot(xi+.35,yi+.33,marker="v",color=CORAL,markersize=3,clip_on=False)
    return im


def journal_header(fig,metric,mode,scope,*,library=None):
    title=f"{FIGURE_COMPARISON}: {metric}"
    operation="Forward" if mode=="forward" else "Forward + backward"
    subject=f"O(2) tensor products · {operation}"
    if library:subject+=f" · {library}"
    subtitle=f"{subject} · {scope}"
    heading=fig.text(.5,.966,title,ha="center",va="center",fontsize=9.5,fontweight="semibold",color=INK)
    detail=fig.text(.5,.930,subtitle,ha="center",va="center",fontsize=7.3,color="#4C5C65")
    fig.text(.975,.014,"v0.1.5",ha="right",va="center",fontsize=6.5,color="#74838B")
    fig._o2hp_header=(heading,detail)


def save(fig,out,stem):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    heading,detail=fig._o2hp_header
    boxes=[t.get_window_extent(renderer) for t in (heading,detail)]
    assert all(box.x0>=fig.bbox.x0+5 and box.x1<=fig.bbox.x1-5 for box in boxes)
    assert boxes[0].y0>boxes[1].y1+2
    assert all(t.get_ha()=="center" and t.get_position()[0]==.5 for t in (heading,detail))
    FIGURE_HEADERS[stem]=dict(title=heading.get_text(),subtitle=detail.get_text(),
        horizontal_alignment="center",x=.5,title_size=heading.get_fontsize(),subtitle_size=detail.get_fontsize())
    for ax in fig.axes:
        title=ax.get_title(loc="center")
        if "parity" in title:
            assert not ax.get_title(loc="left") and ax.title.get_position()[0]==.5
            PARITY_HEADER_AUDIT.append(dict(figure=stem,title=title,alignment="center",x=.5))
    for ext in ("png","svg","pdf"):
        fig.savefig(out/f"{stem}.{ext}",dpi=450,metadata={"Creator":"ICTC-O2 HP 0.1.5 publication batch"})
    plt.close(fig)


def overview(rows,out,mode,*,n=None,c=None):
    fig=plt.figure(figsize=(7.6,5.9))
    levels=CHANNELS if n is not None else SIZES
    scope=f"N = {n:,}" if n is not None else f"C = {c}"
    limits="C = 32–256" if n is not None else "N = 16,384–131,072"
    journal_header(fig,"throughput and peak memory",mode,f"{scope} · {limits} · m_max = 1–6")
    for qi,q in enumerate(("throughput","allocated")):
        norm=scale(rows,mode,q)
        for pi,p in enumerate(PARITIES):
            ax=fig.add_axes((.08+pi*.44,.555-qi*.385,.365,.285))
            z,labels=grid(rows,levels,p,mode,q,n=n,c=c)
            level_labels=list(levels) if n is not None else [str(level//1024) for level in levels]
            im=heatmap(ax,z,labels,level_labels,norm,"Channels, C" if n is not None else "TP rows, N (×1,024)")
            if qi==0: ax.set_xlabel("")
            ax.text(-.13,1.09,"abcd"[qi*2+pi],transform=ax.transAxes,fontsize=10,fontweight="bold")
            ax.set_title(p+" parity",loc="center",pad=13,fontweight="medium")
        cax=fig.add_axes((.91,.555-qi*.385,.014,.285));cb=fig.colorbar(im,cax=cax)
        cb.ax.tick_params(labelsize=7,length=2,pad=3)
        cb.set_label("Throughput (ICTC / eqx)" if q=="throughput" else "Allocated memory (eqx / ICTC)",fontsize=8,labelpad=9)
        sparse_ratio_legend(cb,norm,q)
    fig.text(.07,.052,"† IQR overlap    * isolated timing    ‡ memory increase",fontsize=6.8,color="#64767E")
    stem=f"overview_{mode}_N{n}" if n is not None else f"N_landscape_{mode}_C{c}"
    save(fig,out,stem)


def atlas(rows,out,mode,quantity):
    fig=plt.figure(figsize=(7.6,10.8));norm=scale(rows,mode,quantity)
    metric="Throughput" if quantity=="throughput" else {"allocated":"Allocated memory","reserved":"Reserved memory","process":"NVML process memory"}[quantity]
    journal_header(fig,metric.lower(),mode,"N = 16,384–131,072 · C = 32–256 · m_max = 1–6")
    for ni,n in enumerate(SIZES):
        for pi,p in enumerate(PARITIES):
            ax=fig.add_axes((.08+pi*.44,.721-ni*.205,.365,.153))
            z,labels=grid(rows,CHANNELS,p,mode,quantity,n=n)
            im=heatmap(ax,z,labels,CHANNELS,norm,"Channels, C")
            if ni<3:ax.set_xlabel("")
            ax.set_title(f"{p} parity  ·  N = {n:,}",loc="center",pad=10,fontsize=8.5)
            ax.text(-.13,1.085,"abcdefgh"[ni*2+pi],transform=ax.transAxes,fontsize=9.5,fontweight="bold")
    cb=fig.colorbar(im,cax=fig.add_axes((.91,.106,.014,.768)))
    cb.set_label("ICTC / eqx throughput (×)" if quantity=="throughput" else "eqx / ICTC memory (×)",labelpad=10,fontsize=8)
    cb.ax.tick_params(labelsize=7,length=2,pad=3)
    sparse_ratio_legend(cb,norm,quantity)
    fig.text(.07,.03,"† IQR overlap    * isolated timing    ‡ memory increase",fontsize=6.8,color="#64767E")
    save(fig,out,f"all_N_{quantity}_{mode}")


def gradient_surface(z, *, levels=SIZES, positions=None, axis_name="N", columns=ORDERS,
                     column_positions=None, column_name="m_max", subdivisions=SURFACE_SUBDIVISIONS):
    """Bilinear display facets only inside cells with four measured corners."""
    faces, heights, cells = [], [], []
    assert z.shape == (len(levels),len(columns))
    positions = np.arange(len(levels),dtype=float) if positions is None else np.asarray(positions,dtype=float)
    column_positions = np.asarray(columns,dtype=float) if column_positions is None else np.asarray(column_positions,dtype=float)
    assert positions.shape == (len(levels),) and np.all(np.diff(positions)>0)
    assert column_positions.shape == (len(columns),) and np.all(np.diff(column_positions)>0)
    t = np.linspace(0, 1, subdivisions+1)
    tx, ty = t[None, :], t[:, None]
    for ni in range(len(levels)-1):
        for mi in range(len(columns)-1):
            corners = z[ni:ni+2, mi:mi+2]
            if not np.isfinite(corners).all():
                continue
            zz = (1-ty)*((1-tx)*corners[0,0] + tx*corners[0,1]) + ty*((1-tx)*corners[1,0] + tx*corners[1,1])
            xx = np.broadcast_to(column_positions[mi] + (column_positions[mi+1]-column_positions[mi])*tx, zz.shape)
            yy = np.broadcast_to(positions[ni] + (positions[ni+1]-positions[ni])*ty, zz.shape)
            assert np.array_equal(zz[np.ix_([0,-1], [0,-1])], corners)
            lo, hi = float(corners.min()), float(corners.max())
            assert zz.min() >= lo-1e-12 and zz.max() <= hi+1e-12
            vertices = np.stack((xx,yy,zz), axis=-1)
            patch = np.stack((vertices[:-1,:-1], vertices[:-1,1:],
                              vertices[1:,1:], vertices[1:,:-1]), axis=2).reshape(-1,4,3)
            faces.append(patch); heights.append(patch[:,:,2].mean(axis=1))
            cells.append({axis_name:list(levels[ni:ni+2]), column_name:list(columns[mi:mi+2]),
                          "measured_corners":corners.tolist(), "displayed_bounds":[lo,hi]})
    if not faces:
        return np.empty((0,4,3)), np.empty(0), cells
    return np.concatenate(faces), np.concatenate(heights), cells


def oom_quads(mask,row_positions,column_positions):
    for ri in range(mask.shape[0]-1):
        for ci in range(mask.shape[1]-1):
            if mask[ri:ri+2,ci:ci+2].any():
                yield ri,ci,np.array([[column_positions[ci],row_positions[ri]],
                    [column_positions[ci+1],row_positions[ri]],
                    [column_positions[ci+1],row_positions[ri+1]],
                    [column_positions[ci],row_positions[ri+1]]],dtype=float)


class ProjectedGradientMesh(TriMesh):
    """Continuous vertex colours on an orthographically projected 3D plane."""
    def __init__(self,ax,geometry,ratios,norm,zorder):
        self.world_vertices=geometry.reshape(-1,3).copy()
        self.projection_axes=ax
        px,py,_=proj3d.proj_transform(*self.world_vertices.T,ax.get_proj())
        offsets=4*np.arange(len(geometry))[:,None]
        triangles=np.concatenate((offsets+np.array([0,1,2]),offsets+np.array([0,2,3])))
        self.projected_triangulation=Triangulation(px,py,triangles=triangles)
        super().__init__(self.projected_triangulation,cmap=SURFACE_PALETTE,norm=norm,
            linewidths=0,edgecolors="none",zorder=zorder,rasterized=True)
        self.set_array(ratios.ravel())

    def do_3d_projection(self):
        px,py,pz=proj3d.proj_transform(*self.world_vertices.T,self.projection_axes.get_proj())
        self.projected_triangulation.x=px;self.projected_triangulation.y=py
        return float(pz.min())


def shared_vertex_continuity(geometries,ratios):
    vertices=np.concatenate([g.reshape(-1,3) for g in geometries])
    values=np.concatenate([v.ravel() for v in ratios])
    assert len(vertices)==len(values) and np.isfinite(values).all()
    _,inverse,counts=np.unique(np.round(vertices,12),axis=0,return_inverse=True,return_counts=True)
    lows=np.full(len(counts),np.inf);highs=np.full(len(counts),-np.inf)
    np.minimum.at(lows,inverse,values);np.maximum.at(highs,inverse,values)
    delta=float((highs-lows).max())
    assert delta<=1e-12,delta
    return dict(shared_vertices=int((counts>1).sum()),maximum_seam_ratio_difference=delta)


def smooth_gradient(ax,geometry,ratios,norm,zorder):
    assert geometry.shape[:2]==ratios.shape and geometry.shape[1:]==(4,3)
    continuity=shared_vertex_continuity([geometry],[ratios])
    mesh=ProjectedGradientMesh(ax,geometry,ratios,norm,zorder)
    ax.add_collection(mesh,autolim=False)
    return continuity


def hatch_oom(ax,vertices,zorder):
    ax.add_collection3d(Poly3DCollection([vertices],facecolors="white",edgecolors="#A6B2B8",
        linewidths=.5,hatch="///",zorder=zorder))


def oom_badge(ax,regions,fontsize):
    if not regions:return
    ax.apply_aspect()
    vertices=np.unique(np.concatenate(regions),axis=0)
    px,py,_=proj3d.proj_transform(vertices[:,0],vertices[:,1],vertices[:,2],ax.get_proj())
    projected=ax.transAxes.inverted().transform(ax.transData.transform(np.column_stack((px,py))))
    centre=(projected.min(axis=0)+projected.max(axis=0))/2
    assert np.all(centre>=0) and np.all(centre<=1)
    ax.text2D(*centre,"eqx\nOOM",transform=ax.transAxes,ha="center",va="center",fontsize=fontsize,
        color=INK,bbox=dict(facecolor="white",edgecolor="none",pad=.15),zorder=20)


def surface(rows,out,mode):
    fig=plt.figure(figsize=(9.2,8.1))
    channel_height=np.log2(np.asarray(CHANNELS,dtype=float))
    X,Z=np.meshgrid(ORDERS,channel_height)
    journal_header(fig,"throughput and peak memory",mode,"N = 16,384–131,072 · C = 32–256 · m_max = 1–6")
    for qi,q in enumerate(("throughput","allocated")):
        norm=scale(rows,mode,q)
        for pi,p in enumerate(PARITIES):
            ax=fig.add_axes((.04+pi*.445,.525-qi*.42,.405,.355),projection="3d",computed_zorder=False)
            ax.set_proj_type("ortho");ax.view_init(elev=17,azim=-156);ax.set_box_aspect((1.65,2.3,1.5),zoom=1.08)
            ax.set(xlim=(.8,6.2),ylim=(-.15,3.15),zlim=(channel_height[0]-.12,channel_height[-1]+.12))
            ax.set_xticks(ORDERS);ax.set_yticks(range(4),["16","32","64","128"])
            ax.set_xlabel(r"$m_{\max}$",labelpad=1);ax.set_ylabel("N (×1,024)",labelpad=2)
            ax.set_zticks(channel_height,CHANNELS);ax.set_zlabel("Channels, C (log2)",labelpad=5,fontsize=8)
            ax.tick_params(labelsize=7,pad=0);ax.grid(False)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.pane.fill=False;axis.pane.set_edgecolor((1,1,1,0));axis.line.set_color("#87979E");axis.line.set_linewidth(.5)
            missing,oom=0,0;oom_regions=[]
            for ni,n in enumerate(SIZES):
                z,labels=grid(rows,CHANNELS,p,mode,q,n=n)
                faces, colour_values, cells = gradient_surface(z,levels=CHANNELS,
                    positions=channel_height,axis_name="C")
                draw_order=2*(len(SIZES)-1-ni)
                if len(faces):
                    vertex_ratios=faces[:,:,2].copy()
                    faces[:,:,2]=faces[:,:,1]
                    faces[:,:,1]=ni
                    assert np.all(faces[:,:,1] == ni)
                    assert faces[:,:,2].min()>=channel_height[0] and faces[:,:,2].max()<=channel_height[-1]
                    continuity=smooth_gradient(ax,faces,vertex_ratios,norm,draw_order+1)
                    SMOOTH_AUDIT.append(dict(view="sheets",N=n,mode=mode,parity=p,quantity=q,**continuity))
                ax.plot([1,6,6,1,1],[ni]*5,[channel_height[0],channel_height[0],channel_height[-1],channel_height[-1],channel_height[0]],
                        color="#BCC8CD",lw=.45,zorder=draw_order)
                good=np.isfinite(z)
                ax.scatter(X[good],np.full(int(good.sum()),ni),Z[good],
                           c=z[good],norm=norm,cmap=SURFACE_PALETTE,s=9,
                           edgecolors="white",linewidths=.4,depthshade=False,zorder=draw_order+2)
                SURFACE_AUDIT.append(dict(N=n,mode=mode,parity=p,quantity=q,z_axis="channels",
                    plotted_N_position=ni,plotted_channel_range=channel_height.tolist(),
                    method="bilinear colour interpolation over m_max and log2(C), at constant N; four measured corners required",
                    subdivisions=SURFACE_SUBDIVISIONS,measured_vertices=int(good.sum()),valid_measured_cells=len(cells),
                    rendered_facets=len(faces),complete_cells=cells))
                mask=np.char.find(labels.astype(str),"OOM")>=0
                for ri,ci,quad in oom_quads(mask,channel_height,ORDERS):
                    vertices=np.column_stack((quad[:,0],np.full(4,ni),quad[:,1]))
                    hatch_oom(ax,vertices,draw_order+1.25);oom_regions.append(vertices)
                    failed=[dict(N=n,C=CHANNELS[r],m_max=ORDERS[c])
                            for r in (ri,ri+1) for c in (ci,ci+1) if mask[r,c]]
                    OOM_MARKER_AUDIT.append(dict(view="sheets",mode=mode,parity=p,quantity=q,
                        source_failures=failed,vertices=vertices.tolist(),ratio=None,style="white diagonal hatch"))
                missing+=int(np.sum(labels=="PENDING"));oom+=int(np.sum(np.char.find(labels.astype(str),"OOM")>=0))
            ax.text2D(.06,1.025,"abcd"[qi*2+pi],transform=ax.transAxes,fontsize=10,fontweight="bold")
            ax.text2D(.20,1.025,p+" parity",transform=ax.transAxes,fontsize=9)
            if missing:ax.text2D(.13,-.085,"Pending",transform=ax.transAxes,fontsize=6.8,color="#64767E")
            oom_badge(ax,oom_regions,fontsize=5.8)
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=SURFACE_PALETTE),
                        cax=fig.add_axes((.92,.552-qi*.42,.012,.278)))
        sparse_ratio_legend(cb,norm,q);cb.ax.tick_params(labelsize=7,length=2,pad=3)
        cb.set_label("Throughput (ICTC / eqx)" if q=="throughput" else "Allocated memory (eqx / ICTC)",fontsize=8,labelpad=8)
    save(fig,out,f"N_channel_layers_{mode}")


def connected_channels(rows,out,mode):
    """Three connected boundary faces, with channel gradients on both sides."""
    fig=plt.figure(figsize=(9.2,8.1))
    log_c=np.log2(np.asarray(CHANNELS,dtype=float))
    lookup={(r["N"],r["C"],r["m_max"],r["parity"],r["mode"]):r for r in rows}
    specs=(dict(plane="xy",fixed={"C":256},rows_axis="N",levels=SIZES,positions=np.arange(4),columns_axis="m_max",columns=ORDERS,column_positions=ORDERS),
           dict(plane="xz",fixed={"N":131072},rows_axis="C",levels=CHANNELS,positions=log_c,columns_axis="m_max",columns=ORDERS,column_positions=ORDERS),
           dict(plane="yz",fixed={"m_max":6},rows_axis="C",levels=CHANNELS,positions=log_c,columns_axis="N",columns=SIZES,column_positions=np.arange(4)))
    journal_header(fig,"throughput and peak memory",mode,"N = 16,384–131,072 · C = 32–256 · m_max = 1–6")
    for qi,q in enumerate(("throughput","allocated")):
        norm=scale(rows,mode,q)
        for pi,p in enumerate(PARITIES):
            ax=fig.add_axes((.04+pi*.445,.525-qi*.42,.405,.355),projection="3d",computed_zorder=False)
            ax.set_proj_type("ortho");ax.view_init(elev=21,azim=52);ax.set_box_aspect((1.55,1.25,1.4),zoom=1.04)
            ax.set(xlim=(.85,6.15),ylim=(-.1,3.1),zlim=(log_c[0]-.1,log_c[-1]+.1))
            ax.set_xticks(ORDERS);ax.set_yticks(range(4),["16","32","64","128"]);ax.set_zticks(log_c,CHANNELS)
            ax.set_xlabel(r"$m_{\max}$",labelpad=1);ax.set_ylabel("N (×1,024)",labelpad=2);ax.set_zlabel("Channels, C (log2)",labelpad=5,fontsize=8)
            ax.tick_params(labelsize=7,pad=0);ax.grid(False)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.pane.fill=False;axis.pane.set_edgecolor((1,1,1,0));axis.line.set_color("#87979E");axis.line.set_linewidth(.5)
            pending,oom=set(),set();oom_regions=[];panel_geometries=[];panel_ratios=[]
            for spec in specs:
                colours=np.full((len(spec["levels"]),len(spec["columns"])),np.nan)
                oom_mask=np.zeros(colours.shape,dtype=bool)
                for ri,rlevel in enumerate(spec["levels"]):
                    for ci,clevel in enumerate(spec["columns"]):
                        point={**spec["fixed"],spec["rows_axis"]:rlevel,spec["columns_axis"]:clevel}
                        row=lookup.get((point["N"],point["C"],point["m_max"],p,mode));v=value(row,q)
                        key=(point["N"],point["C"],point["m_max"])
                        if v is not None:colours[ri,ci]=v
                        elif row is None:pending.add(key)
                        else:oom.add(key);oom_mask[ri,ci]=True
                        AUDIT.append(dict(**point,parity=p,mode=mode,quantity=q,value=v,label=label(row,q,v),
                            source=row["timing_source"] if row else None,state="measured" if v is not None else "oom" if row else "pending"))
                faces,face_values,cells=gradient_surface(colours,levels=spec["levels"],positions=spec["positions"],axis_name=spec["rows_axis"],
                    columns=spec["columns"],column_positions=spec["column_positions"],column_name=spec["columns_axis"])
                def map_coordinates(a,b):
                    if spec["plane"]=="xy":return a,b,np.full_like(a,log_c[-1])
                    if spec["plane"]=="xz":return a,np.full_like(a,3),b
                    return np.full_like(a,6),a,b
                if len(faces):
                    geometry=np.stack(map_coordinates(faces[:,:,0],faces[:,:,1]),axis=-1)
                    assert geometry[:,:,0].min()>=1 and geometry[:,:,0].max()<=6
                    assert geometry[:,:,1].min()>=0 and geometry[:,:,1].max()<=3
                    assert geometry[:,:,2].min()>=log_c[0] and geometry[:,:,2].max()<=log_c[-1]
                    vertex_ratios=faces[:,:,2].copy()
                    smooth_gradient(ax,geometry,vertex_ratios,norm,1)
                    panel_geometries.append(geometry);panel_ratios.append(vertex_ratios)
                a,b=np.meshgrid(spec["column_positions"],spec["positions"]);good=np.isfinite(colours)
                px,py,pz=map_coordinates(a[good],b[good])
                ax.scatter(px,py,pz,c=colours[good],norm=norm,cmap=SURFACE_PALETTE,s=8,edgecolors="white",linewidths=.35,depthshade=False,zorder=2)
                for ri,ci,quad in oom_quads(oom_mask,spec["positions"],spec["column_positions"]):
                    vertices=np.stack(map_coordinates(quad[:,0],quad[:,1]),axis=-1)
                    hatch_oom(ax,vertices,1.25);oom_regions.append(vertices)
                    failed=[{**spec["fixed"],spec["rows_axis"]:spec["levels"][r],spec["columns_axis"]:spec["columns"][c]}
                            for r in (ri,ri+1) for c in (ci,ci+1) if oom_mask[r,c]]
                    OOM_MARKER_AUDIT.append(dict(view="connected",plane=spec["plane"],mode=mode,parity=p,quantity=q,
                        source_failures=failed,vertices=vertices.tolist(),ratio=None,style="white diagonal hatch"))
                CONNECTED_AUDIT.append(dict(plane=spec["plane"],fixed=spec["fixed"],rows_axis=spec["rows_axis"],levels=list(spec["levels"]),
                    columns_axis=spec["columns_axis"],columns=list(spec["columns"]),parity=p,mode=mode,quantity=q,
                    valid_measured_cells=len(cells),rendered_facets=len(faces),complete_cells=cells))
            continuity=shared_vertex_continuity(panel_geometries,panel_ratios)
            SMOOTH_AUDIT.append(dict(view="connected",mode=mode,parity=p,quantity=q,**continuity))
            ax.text2D(.06,1.025,"abcd"[qi*2+pi],transform=ax.transAxes,fontsize=10,fontweight="bold")
            ax.text2D(.20,1.025,p+" parity",transform=ax.transAxes,fontsize=9)
            if pending:ax.text2D(.13,-.085,"Pending",transform=ax.transAxes,fontsize=6.8,color="#64767E")
            oom_badge(ax,oom_regions,fontsize=7)
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=SURFACE_PALETTE),cax=fig.add_axes((.92,.552-qi*.42,.012,.278)))
        sparse_ratio_legend(cb,norm,q)
        cb.ax.tick_params(labelsize=7,length=2,pad=3);cb.set_label("Throughput (ICTC / eqx)" if q=="throughput" else "Allocated memory (eqx / ICTC)",fontsize=8,labelpad=8)
    save(fig,out,f"N_channel_connected_{mode}")


def absolute_memory_atlas(rows,out,mode,quantity,prefix):
    field=FIELDS[quantity]
    values=[r[name+"_"+field]/1024 for r in rows if r["mode"]==mode
            for name in ("candidate","equivariantx") if r[name+"_"+field] is not None]
    norm=Normalize(vmin=0,vmax=max(values))
    lookup={(r["N"],r["C"],r["m_max"],r["parity"],r["mode"]):r for r in rows}
    fig=plt.figure(figsize=(7.6,10.8))
    library="ICTC-O2 HP" if prefix=="candidate" else "eqx"
    metric="Peak allocated" if quantity=="allocated" else "Peak reserved"
    journal_header(fig,metric.lower()+" memory",mode,"N = 16,384–131,072 · C = 32–256 · m_max = 1–6",library=library)
    for ni,n in enumerate(SIZES):
        for pi,p in enumerate(PARITIES):
            z=np.full((5,6),np.nan);labels=np.empty(z.shape,dtype=object)
            for yi,c in enumerate(CHANNELS):
                for xi,m in enumerate(ORDERS):
                    row=lookup.get((n,c,m,p,mode));v=row[prefix+"_"+field]/1024 if row and row[prefix+"_"+field] is not None else None
                    labels[yi,xi]=f"{v:.2f}" if v is not None else "PENDING" if row is None else "EQX\nOOM"
                    if v is not None:z[yi,xi]=v
                    AUDIT.append(dict(N=n,C=c,m_max=m,parity=p,mode=mode,quantity=quantity,prefix=prefix,
                        value=v,label=labels[yi,xi],source=row[prefix+"_memory_source"] if row else None,
                        state="measured" if v is not None else "pending" if row is None else "oom"))
            ax=fig.add_axes((.08+pi*.44,.721-ni*.205,.365,.153))
            im=heatmap(ax,z,labels,CHANNELS,norm,"Channels, C",cmap=ABS_PALETTE)
            if ni<3:ax.set_xlabel("")
            ax.set_title(f"{p} parity  ·  N = {n:,}",loc="center",pad=10,fontsize=8.5)
            ax.text(-.13,1.085,"abcdefgh"[ni*2+pi],transform=ax.transAxes,fontsize=9.5,fontweight="bold")
    cb=fig.colorbar(im,cax=fig.add_axes((.92,.106,.014,.768)));cb.set_label(metric+" memory (GiB)",labelpad=10,fontsize=8)
    cb.ax.tick_params(labelsize=7,length=2,pad=3)
    cb.locator=MaxNLocator(nbins=3,min_n_ticks=2);cb.update_ticks()
    LEGEND_AUDIT.append(dict(quantity="absolute_"+quantity,ticks=cb.get_ticks().tolist()))
    save(fig,out,f"all_N_absolute_{quantity}_{mode}_{prefix}")


def verify(rows,out,pending,gates,raw_root):
    lookup={(r["N"],r["C"],r["m_max"],r["parity"],r["mode"]):r for r in rows}
    # Independently compare rendered values with the current eqx-only report.
    import report_eqx as reference
    original=reference.collect(raw_root,preview=bool(pending))[0]
    for r in original:
        actual=lookup[(r["N"],r["C"],r["m_max"],r["parity"],r["mode"])]
        for field in ("throughput_ratio","candidate_ms","equivariantx_ms","paired_complete",
                      "candidate_peak_allocated_MiB","equivariantx_peak_allocated_MiB",
                      "candidate_peak_reserved_MiB","equivariantx_peak_reserved_MiB",
                      "candidate_sampled_process_peak_MiB","equivariantx_sampled_process_peak_MiB"):
            assert actual[field] == r[field], (field,actual,r)
    for item in AUDIT:
        row=lookup.get((item["N"],item["C"],item["m_max"],item["parity"],item["mode"]))
        if item.get("prefix"):
            raw=row[item["prefix"]+"_"+FIELDS[item["quantity"]]] if row else None
            expected=raw/1024 if raw is not None else None
            expected_label=f"{expected:.2f}" if expected is not None else "PENDING" if row is None else "EQX\nOOM"
        else:
            expected=value(row,item["quantity"]);expected_label=label(row,item["quantity"],expected)
            if row is not None and item["quantity"] != "throughput":
                assert expected == row[item["quantity"]+"_memory_ratio"]
        assert expected==item["value"] and expected_label==item["label"]
        if item["state"] == "measured" and not item.get("prefix"):
            assert "×" in item["label"] and "%" not in item["label"]
        assert item["m_max"] in ORDERS
    for sheet in SURFACE_AUDIT:
        assert sheet["plotted_N_position"] == SIZES.index(sheet["N"])
        assert sheet["plotted_channel_range"] == np.log2(np.asarray(CHANNELS,dtype=float)).tolist()
        measured_grid=np.array([[value(lookup.get((sheet["N"],c,m,sheet["parity"],sheet["mode"])),sheet["quantity"])
                                 for m in ORDERS] for c in CHANNELS],dtype=float)
        complete=sum(np.isfinite(measured_grid[ci:ci+2,mi:mi+2]).all()
                     for ci in range(len(CHANNELS)-1) for mi in range(len(ORDERS)-1))
        assert sheet["valid_measured_cells"] == complete
        assert sheet["rendered_facets"] == complete*SURFACE_SUBDIVISIONS**2
        for cell in sheet["complete_cells"]:
            raw=np.array([[value(lookup[(sheet["N"],c,m,sheet["parity"],sheet["mode"])],sheet["quantity"])
                           for m in cell["m_max"]] for c in cell["C"]])
            assert np.array_equal(raw,np.array(cell["measured_corners"]))
            assert cell["displayed_bounds"] == [float(raw.min()),float(raw.max())]
    for face in CONNECTED_AUDIT:
        def raw_value(rlevel,clevel):
            point={**face["fixed"],face["rows_axis"]:rlevel,face["columns_axis"]:clevel}
            return value(lookup.get((point["N"],point["C"],point["m_max"],face["parity"],face["mode"])),face["quantity"])
        raw_grid=np.array([[raw_value(r,c) for c in face["columns"]] for r in face["levels"]],dtype=float)
        complete=sum(np.isfinite(raw_grid[ri:ri+2,ci:ci+2]).all()
                     for ri in range(len(face["levels"])-1) for ci in range(len(face["columns"])-1))
        assert face["valid_measured_cells"] == complete and face["rendered_facets"] == complete*SURFACE_SUBDIVISIONS**2
        for cell in face["complete_cells"]:
            raw=np.array([[raw_value(r,c) for c in cell[face["columns_axis"]]] for r in cell[face["rows_axis"]]])
            assert np.array_equal(raw,np.array(cell["measured_corners"]))
            assert cell["displayed_bounds"] == [float(raw.min()),float(raw.max())]
    images,svgs,fonts={},{},{}
    for marker in OOM_MARKER_AUDIT:
        assert marker["ratio"] is None and marker["source_failures"]
        for point in marker["source_failures"]:
            raw=lookup[(point["N"],point["C"],point["m_max"],marker["parity"],marker["mode"])]
            assert raw["equivariantx_status"]=="oom" and value(raw,marker["quantity"]) is None
    for path in out.glob("*.png"):
        with Image.open(path) as im:images[path.name]=list(im.size)
    for path in out.glob("*.svg"):
        tree=ET.parse(path);count=len(tree.findall(".//{http://www.w3.org/2000/svg}text"));assert count>10
        labels=["".join(t.itertext()) for t in tree.findall(".//{http://www.w3.org/2000/svg}text")]
        assert FIGURE_HEADERS[path.stem]["title"] in labels
        assert not any("boundary slices" in t.lower() or "measured shapes" in t.lower() for t in labels)
        if path.stem.startswith("N_channel_"):
            assert tree.findall(".//{http://www.w3.org/2000/svg}image"),path
        svgs[path.name]=count
    for path in out.glob("*.pdf"):
        report=subprocess.run(["pdffonts",str(path)],check=True,capture_output=True,text=True).stdout
        assert "Type 3" not in report and len(report.splitlines())>2
        assert all(re.search(r"\byes\s+yes\s+yes\b",line) for line in report.splitlines()[2:])
        fonts[path.name]=report
    (out/"figure_data.json").write_text(json.dumps(AUDIT,indent=2)+"\n")
    (out/"surface_geometry.json").write_text(json.dumps(SURFACE_AUDIT,indent=2)+"\n")
    (out/"connected_geometry.json").write_text(json.dumps(CONNECTED_AUDIT,indent=2)+"\n")
    (out/"oom_markers.json").write_text(json.dumps(OOM_MARKER_AUDIT,indent=2)+"\n")
    (out/"smooth_surface_qa.json").write_text(json.dumps(SMOOTH_AUDIT,indent=2)+"\n")
    assert all(len(entry["ticks"])<=4 for entry in LEGEND_AUDIT)
    (out/"figure_qa.json").write_text(json.dumps(dict(status="ok",displayed_cells=len(AUDIT),measured_rows=len(rows),
        source_version="0.1.5",m_max=list(ORDERS),N=list(SIZES),C=list(CHANNELS),pending_shapes=pending,
        qualification_families=len(gates),display_units={"throughput":"×","allocated":"×","reserved":"×","process":"×","absolute_memory":"GiB"},
        ratio_definitions={"throughput":"ICTC rows/s / EQX rows/s", "memory":"EQX peak / ICTC peak", "neutral":1, "favours_ICTC":">1"},
        surface_layers=len(SURFACE_AUDIT),surface_method="bounded bilinear colour interpolation over m_max and log2(C), at constant N; four measured corners required",
        surface_axes={"x":"m_max","y":"log2(N/16384), ticks in 1024 rows","z":"log2(C), ticks show actual C","colour":"throughput ICTC/EQX; memory EQX/ICTC"},
        connected_faces=len(CONNECTED_AUDIT),connected_slices={"C":256,"N":131072,"m_max":6},
        figure_comparison=FIGURE_COMPARISON,figure_headers=FIGURE_HEADERS,legend_ticks=LEGEND_AUDIT,
        parity_headers=PARITY_HEADER_AUDIT,oom_markers=len(OOM_MARKER_AUDIT),oom_marker_style="white diagonal hatch, label inside missing region",
        smooth_surface_method="continuous Gouraud colour between dense bilinear source vertices; 450-dpi raster fill in PDF/SVG, vector labels/axes/markers",
        smooth_surface_seams=SMOOTH_AUDIT,
        png_dimensions=images,svg_text_counts=svgs,pdf_fonts=fonts),indent=2)+"\n")
    print("PUBLICATION_QA_OK",len(rows),"metric rows;",len(AUDIT),"displayed cells;",len(pending),"pending shapes",flush=True)


def main():
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("render/data analysis requires sbatch")
    parser=argparse.ArgumentParser();parser.add_argument("--preview",action="store_true")
    parser.add_argument("--raw",type=Path,default=Path(__file__).resolve().parents[1]/"results/raw")
    parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    root=args.raw.resolve()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    rows,pending,gates=collect_ready(root,strict=not args.preview);style()
    for mode in ("forward","forward_vjp"):
        overview(rows,out,mode,n=131072)
        for c in ((128,256) if args.preview else CHANNELS):
            overview(rows,out,mode,c=c)
        surface(rows,out,mode)
        connected_channels(rows,out,mode)
        if not args.preview:
            for q in ("throughput","allocated","reserved","process"):atlas(rows,out,mode,q)
            for q in ("allocated","reserved"):
                for prefix in ("candidate","equivariantx"):absolute_memory_atlas(rows,out,mode,q,prefix)
    with (out/"complete_available_matrix.csv").open("w") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"FIGURE_CAPTIONS.md").write_text("# ICTC-O2 HP 0.1.5 publication figures\n\n"
        "Only completed 0.1.5 measured shapes are used. N counts independent tensor-product rows; m_max covers1–6. "
        "Figures use centred descriptive comparison titles and concise operation/range subtitles, sparse colourbar ticks and concise plotting labels. Coverage, numerical method and interpolation details are recorded here and in the data/geometry audits. "
        "Preview contains explicit PENDING cells where scanning is unfinished. Final figures require all240 shapes/480 metrics and60 reference families.\n\n"
        "Throughput ratio is ICTC/EQX rows/s. Memory ratio is EQX/ICTC peak memory. Both use1x as neutral and>1x favours ICTC. "
        "Every relative heatmap cell and scale includes ×; no memory percentages are mixed with throughput ratios. "
        "For example,2.19x memory ratio means EQX uses2.19 times the ICTC footprint. Separate absolute-memory figures use GiB. "
        "Colour/height scales are shared across C/parity/N within each metric/operation. A dagger marks overlapping paired IQRs (descriptive, not a significance test). "
        "An asterisk marks isolated timing after joint-process OOM. A double dagger marks a small memory increase whose ratio rounds to1.00x; exact values remain in CSV/JSON. "
        "Hatched OOM cells are labelled EQX and have no finite comparison ratio. "
        "3D OOM regions use white diagonally hatched frames with labels inside the projected missing region, with no assigned ratio. 2D parity headers are centred per panel. "
        "A 3D white frame covers an interpolation cell touching the failed measured corner, rather than declaring all adjoining measured points OOM. "
        "3D axes are m_max, N and channel count C, with N/C at log2 spacing; colour alone represents ratios. Each vertical sheet is one measured N. "
        "Filled 3D surfaces use bilinear colour interpolation only within cells with four measured corners, without overshoot or filling missing/OOM cells. "
        "Dense bilinear vertices receive continuous Gouraud colour interpolation, including the C direction and joined boundary faces. Shared vertex ratios are audited for continuity. "
        "PDF/SVG use 450-dpi raster surface fills to avoid polygon stitching seams; text, axes, measured markers and OOM hatches remain vector. "
        "The lower 3D row uses peak allocated memory. A light-grey1x colour keeps neutral regions visible; the neutral threshold is unchanged. "
        "N ticks16/32/64/128 are in units of1,024 TP rows. "
        "Small outlined dots mark actual measured settings; interpolated colour points are not additional measurements. Geometry/corner audits are in surface_geometry.json.\n\n"
        "The connected 3D view joins the C256, N131072 and m_max=6 boundary faces. Channel gradients on the two vertical faces connect the measured C levels. "
        "Interior values are shown in the companion N sheets and full heatmaps; the connected exterior does not expose every interior case. All source corners/masks are audited in connected_geometry.json.\n\n"
        "All numerical gates cover every output/input-gradient row and every path weight at maximum N, FP32 gate1e-5. "
        "Full VJP includes both input gradients and all packed weights. Allocator memory includes inputs, parameters, outputs/adjoints, workspace and the precreated cotangent "
        "retained in forward-only measurement as well. NVML includes context/cache and samples may miss transients. PNG450dpi, editable SVG, embedded PDF fonts.\n")
    verify(rows,out,pending,gates,root)


if __name__=="__main__":main()
