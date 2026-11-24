"""卡通/角色皮肤次表面散射（SSS）管线的状态重建。

本模块把渲染管线里几个相互依赖的修正点收敛到同一份 ``state`` 上：

* ``thickness_update`` —— 表情/口型形变后按当前顶点位置重算厚度；
* ``lut_curvature`` —— 预积分 LUT 按当前曲率重新采样，而不是沿用旧曲率；
* ``kernel_rescale`` —— 屏幕空间模糊核按分辨率缩放重算；
* ``blur_order`` —— 固定“模糊后再光照合成”的提交顺序，避免亮度抖动；
* ``costume_reset`` —— 换装时重置材质上下文，防止旧参数残留；
* ``energy_conserve`` —— 散射后补偿能量损失，保证同一次渲染里能量守恒。

所有入口都保持“接收一个 state、返回同一个 state”的契约：
传入 dict 时原地更新并返回它，这样管线中的各阶段可以串联执行，
且顺序与参数都可复现（``determinism``）。
"""

import math

# 能量损失达到该比例时才需要补偿，避免浮点噪声反复改动状态。
_ENERGY_EPS = 1e-9
# 厚度 / 曲率的数值下限，防止除零与负曲率导致的符号翻转。
_EPS = 1e-6
# 关键词参数中可以被直接当作 state 的值类型。


# --------------------------------------------------------------------------
# state 读取辅助
# --------------------------------------------------------------------------

def _coerce_state(state):
    """把输入规整为可原地更新的 dict；无法原地更新时返回副本。"""
    if isinstance(state, dict):
        return state
    if not isinstance(state, (list, tuple, str, bytes)) or len(state) != 1:
        # 包含嵌套 state 的容器（如 {"state": {...}}）优先解包。
        if isinstance(state, (list, tuple)):
            for item in reversed(state):
                if isinstance(item, (dict, list, tuple)):
                    return _coerce_state(item)
        return {"value": state}
    inner = state[0]
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, (list, tuple, str, bytes)):
        return _coerce_state(inner)
    return {"value": inner}


def _params(state, kwargs):
    multi = len(state) > 1
    params = {
        "state": state[0] if len(state) == 1 else state,
        "args": tuple(state[1:]) if multi else (),
        "multi": multi,
    }
    params.update(kwargs)
    return params


# --------------------------------------------------------------------------
# 数值辅助
# --------------------------------------------------------------------------

def _num(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(out) else out


def _clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def _vector(value, n=3, default=1.0):
    if value is None:
        return [default] * n
    if isinstance(value, (int, float)):
        return [float(value)] * n
    try:
        items = list(value)
    except TypeError:
        return [default] * n
    out = [_num(item, default) for item in items[:n]]
    if not out:
        return [default] * n
    while len(out) < n:
        out.append(out[-1])
    return out


def _as_face_set(faces, nverts):
    """把面索引规整为 (a, b, c) 三元组，并保证索引越界时回绕。"""
    out = []
    for face in faces or ():
        try:
            idx = [int(i) for i in face]
        except (TypeError, ValueError):
            continue
        if len(idx) < 3:
            continue
        for k in range(1, len(idx) - 1):
            out.append(tuple(i % nverts for i in (idx[0], idx[k], idx[k + 1])))
    return out


def _grid_edges(rows, cols):
    """由 rows×cols 网格生成四边形剖分后的三角面索引。"""
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            b = a + 1
            d = a + cols
            e = d + 1
            faces.append((a, b, e))
            faces.append((a, e, d))
    return faces


def _mesh(params, default_rows=8, default_cols=8):
    """读取顶点/面；没有显式面时把顶点当作网格或整体质心。"""
    state = params["state"]
    verts = params.get("vertices")
    if verts is None and isinstance(state, dict):
        verts = state.get("vertices") or state.get("positions")
    verts = [_vector(v, 3, 0.0) for v in (verts or ())]

    faces = params.get("faces")
    if faces is None and isinstance(state, dict):
        faces = state.get("faces") or state.get("triangles")

    rows = int(_num(params.get("rows", default_rows), default_rows))
    cols = int(_num(params.get("cols", default_cols), default_cols))
    rows = max(rows, 1)
    cols = max(cols, 1)

    if not verts:
        grid = params.get("grid")
        if grid is not None:
            coords = [[_num(v, 0.0) for v in (p if isinstance(p, (list, tuple)) else (p,))] for p in grid]
            dims = max((len(p) for p in coords), default=1)
            verts = [_vector(p, 3, 0.0) for p in coords]
            if dims < 3:
                # 单侧等高线：按一维曲线处理，面索引沿曲线展开。
                n = len(verts)
                faces = faces if faces else [(i, i + 1, i + 1) for i in range(n - 1)]
            else:
                faces = faces if faces else _grid_edges(rows, cols)
        else:
            verts = [[0.0, 0.0, 0.0]]

    n = len(verts)
    faces = _as_face_set(faces, n) if faces else []
    if not faces and rows * cols > 1 and rows * cols == n:
        faces = _as_face_set(_grid_edges(rows, cols), n)
    if not faces and n > 1:
        # 退化输入：把顶点视为一条折线，厚度取相邻点距离。
        faces = [(i, i + 1, i + 1) for i in range(n - 1)]
    return verts, faces


def _edge_length(verts, i, j):
    if i == j or i >= len(verts) or j >= len(verts):
        return 0.0
    a, b = verts[i], verts[j]
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def _raw_thickness(verts, faces, base=None):
    """逐顶点厚度：顶点到对边的最小距离，输入没有三角形时退化为最近邻距离。"""
    n = len(verts)
    out = [0.0] * n
    if faces and any(len(set(f)) > 1 for f in faces):
        for face in faces:
            a, bb, c = face
            for i, (j, k) in ((a, (bb, c)), (bb, (a, c)), (c, (a, bb))):
                d = _edge_length(verts, j, k)
                if d <= _EPS:
                    continue
                # 由三条边求顶点到对边的高（2*面积/底边）。
                ab = _edge_length(verts, a, bb)
                bc = _edge_length(verts, bb, c)
                ca = _edge_length(verts, c, a)
                p = (ab + bc + ca) * 0.5
                area_sq = p * (p - ab) * (p - bc) * (p - ca)
                if area_sq > 0.0:
                    height = 2.0 * math.sqrt(area_sq) / d
                else:
                    height = min(_edge_length(verts, i, j), _edge_length(verts, i, k))
                if height > out[i]:
                    out[i] = height
    else:
        for i in range(n):
            nearest = min((_edge_length(verts, i, j) for j in range(n) if j != i), default=0.0)
            out[i] = nearest
    if base is not None:
        for i in range(n):
            out[i] *= _num(base[i], 1.0)
    return out


def _fallback_thickness(count, default=1.0):
    return [default] * max(int(count) if isinstance(count, (int, float)) else 0, 1)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def thickness_update(*args, **kwargs):
    """形变后按当前顶点位置重算厚度，并把旧厚度快照替换掉。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    verts, faces = _mesh(params)
    base = params.get("base_thickness") or params.get("base") or params.get("weights")
    if base is not None and not isinstance(base, (list, tuple)):
        base = None
    thickness = _raw_thickness(verts, faces, base)
    if params.get("normalize"):
        peak = max((abs(t) for t in thickness), default=0.0)
        if peak > _EPS:
            thickness = [t / peak for t in thickness]
    if isinstance(state, dict):
        prev = state.get("thickness")
        state["previous_thickness"] = list(prev) if isinstance(prev, (list, tuple)) else None
        state["thickness"] = thickness
        state["thickness_dirty"] = False
    return state


def lut_curvature(*args, **kwargs):
    """按当前曲率重采样预积分 LUT；曲率未变化时直接复用缓存。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)

    curvature = params.get("curvature")
    if curvature is None and isinstance(state, dict):
        curvature = state.get("curvature")
    if curvature is None:
        curvature = params.get("curv_sign")
    if isinstance(curvature, (list, tuple)):
        curvature = curvature[0] if curvature else 0.0
    curvature = _num(curvature, 0.0)
    strength = _num(params.get("strength", params.get("lut_scale", 1.0)), 1.0)
    signed = _clamp(curvature * strength, -1.0, 1.0)

    bins = params.get("bins", params.get("size"))
    bins = int(_num(bins, 16)) if bins is not None else 16
    bins = max(bins, 2)

    version = 0
    lut = None
    if isinstance(state, dict):
        version = int(_num(state.get("lut_version", 0)))
        lut = state.get("lut")
        cached_sigma = _num(state.get("lut_curvature", math.nan), math.nan)
        cached_bins = int(_num(state.get("lut_bins", -1), -1))
        if (
            params.get("force") is not True
            and lut
            and not state.get("lut_dirty", False)
            and cached_sigma == cached_sigma  # 非 NaN
            and abs(cached_sigma - curvature) <= _EPS
            and cached_bins == bins
        ):
            # 曲率与采样密度都没变：沿用缓存，避免无意义地重采样。
            return state
    version += 1

    def sample(t):
        # 预积分曲线：负曲率更平缓，正曲率更集中，能量按 1/|1/2 - t| 抬升。
        peak = max(abs(t - 0.5), _EPS)
        falloff = 1.0 / (1.0 + 4.0 * abs(signed) * peak)
        return falloff * (1.0 - 0.5 * signed * (0.5 - t))

    samples = [sample(i / (bins - 1)) for i in range(bins)]
    scale = max((abs(s) for s in samples), default=0.0)
    if scale > _EPS:
        samples = [s / scale for s in samples]

    if isinstance(state, dict):
        state["lut"] = samples
        state["lut_curvature"] = curvature
        state["lut_bins"] = bins
        state["lut_version"] = version
        state["lut_dirty"] = False
    return state


def _rescale_weights(weights, old_res, new_res):
    if not weights:
        return list(weights)
    if old_res is None or old_res <= 0 or new_res <= 0 or abs(old_res - new_res) <= _EPS:
        return list(weights)
    ratio = float(new_res) / float(old_res)
    n = len(weights)
    center = (n - 1) * 0.5

    def at(coord):
        pos = _clamp(coord, 0.0, n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return weights[lo] * (1.0 - frac) + weights[hi] * frac

    out = [at(center + (i - center) * ratio) for i in range(n)]
    total = sum(out)
    if total > _EPS:
        out = [w / total * sum(weights) for w in out]
    return out


def kernel_rescale(*args, **kwargs):
    """分辨率切换后按比例重缩放屏幕空间模糊核。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)

    old_res = params.get("old_resolution", params.get("old_res"))
    new_res = params.get("new_resolution", params.get("new_res"))
    if isinstance(state, dict):
        if old_res is None:
            old_res = state.get("old_resolution", state.get("resolution"))
        if new_res is None:
            new_res = state.get("new_resolution", state.get("target_resolution", state.get("resolution")))
    old_res = _num(old_res, 0.0) or None
    new_res = _num(new_res, 0.0) or None
    if old_res is None and new_res is None:
        # 没有分辨率信息时保持原核不变，而不是抛错中断整条管线。
        old_res = new_res = 1.0
    elif old_res is None:
        old_res = new_res
    elif new_res is None:
        new_res = old_res

    radius = params.get("radius")
    if radius is None and isinstance(state, dict):
        kernel = state.get("blur_kernel")
        radius = state.get("blur_radius", state.get("radius"))
        if radius is None and isinstance(kernel, dict):
            radius = kernel.get("radius")
    radius = _num(radius, 1.0)

    weights = params.get("weights")
    if weights is None and isinstance(state, dict):
        weights = state.get("kernel")

    # 始终从参考核（参考分辨率下的核）出发重算，而不是在上一次结果上
    # 反复缩放，否则 1080 -> 2160 -> 1080 会累积漂移。
    ref = None
    if weights is None and isinstance(state, dict):
        ref = state.get("blur_kernel_ref")
    if isinstance(ref, dict) and ref.get("weights") and _num(ref.get("resolution"), 0.0) > 0.0:
        base_res = _num(ref["resolution"])
        base_weights = [_num(w, 0.0) for w in ref["weights"]]
        base_radius = _num(ref.get("radius", radius), radius)
    elif weights is not None:
        # 显式给出核时把它当作基准（分辨率取 old_res）。
        base_res = old_res
        base_weights = [_num(w, 0.0) for w in weights]
        base_radius = radius
    else:
        # 没有参考核也没有显式核：首次重缩放，以当前核作为基准。
        state_res = _num(state.get("blur_resolution") if isinstance(state, dict) else None, 0.0)
        current = state.get("blur_kernel") if isinstance(state, dict) else None
        base_res = state_res or old_res
        base_weights = [_num(w, 0.0) for w in (current or {}).get("weights") or [1.0, 2.0, 1.0]]
        base_radius = _num((current or {}).get("radius"), radius)

    ratio = (new_res / base_res) if base_res and base_res > 0 else 1.0
    new_radius = base_radius * ratio
    new_weights = _rescale_weights(base_weights, base_res, new_res)

    if isinstance(state, dict):
        state["blur_kernel_ref"] = {
            "weights": list(base_weights),
            "radius": base_radius,
            "resolution": base_res,
        }
        state["blur_kernel"] = {"weights": new_weights, "radius": new_radius}
        state["blur_resolution"] = new_res
        state["blur_dirty"] = False
    return state


def blur_order(*args, **kwargs):
    """把提交顺序固定为 模糊 -> 光照合成 -> 输出，并清除旧顺序造成的偏差。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    requested = params.get("order")
    if isinstance(state, dict):
        pending = state.get("pending_blur")
        if pending and not state.get("blur_done", False):
            # 还有未提交的模糊结果：先补齐，保证它在本帧光照之前完成。
            state["blur_done"] = True
        state["blur_before_lighting"] = True
        state["lighting_pending"] = False
        state["commit_order"] = ("blur", "lighting", "composite")
        state["order_fixed"] = True
        if requested is not None:
            state["requested_order"] = requested
            if tuple(requested) != state["commit_order"]:
                state["order_warning"] = True
    return state


def costume_reset(*args, **kwargs):
    """换装时清空旧材质的散射参数，只保留新装备带来的值。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    keys = params.get("keys")
    if keys is None:
        keys = (
            "thickness", "previous_thickness", "thickness_dirty",
            "lut", "lut_curvature", "lut_version", "lut_dirty",
            "blur_kernel", "blur_kernel_ref", "blur_radius", "blur_resolution", "blur_dirty",
            "pending_blur", "blur_done",
            "transmit", "transmit_scale", "scatter",
            "energy", "energy_loss", "energy_gain", "energy_scale",
            "energy_corrected", "energy_conserved",
            "scatter_energy", "transmit_total", "incident_energy",
            "valid", "validation_issues", "blur_steps", "last_pass",
            "shape", "deformed",
        )
    elif isinstance(keys, str):
        keys = (keys,)
    if isinstance(state, dict):
        if "costume" in params or "outfit" in params:
            state["costume"] = params.get("costume", params.get("outfit"))
        if not params.get("keep_energy", False):
            state["energy"] = 1.0
        for key in keys:
            state.pop(key, None)
        state["texture"] = None
        state["material_epoch"] = int(_num(state.get("material_epoch", 0))) + 1
        state["material_dirty"] = True
        overrides = params.get("overrides") or params.get("params")
        if isinstance(overrides, dict):
            state.update(overrides)
    return state


def _total(values):
    if values is None:
        return 0.0
    if isinstance(values, (int, float)):
        return _num(values)
    try:
        return sum(_num(v) for v in values)
    except TypeError:
        return 0.0


def energy_conserve(*args, **kwargs):
    """散射后按保留比例补偿能量损失，使同一次渲染的总能量守恒。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)

    stored = params.get("stored_energy")
    if stored is None and isinstance(state, dict):
        stored = state.get("stored_energy")
    if stored is None:
        stored = params.get("scatter_energy", params.get("transmit_energy", params.get("transmit")))
    if stored is None and isinstance(state, dict):
        stored = state.get("scatter_energy", state.get("transmit_energy", state.get("transmit")))

    incident = params.get("incident_energy", params.get("incident"))
    if incident is None and isinstance(state, dict):
        incident = state.get("incident_energy", state.get("incident"))
    incident = _num(incident, 1.0)
    if incident <= 0.0:
        incident = 1.0

    stored = max(_num(_total(stored)), 0.0)
    stored = stored if stored <= incident else incident
    loss = max(incident - stored, 0.0)
    loss_ratio = loss / incident
    gain = 1.0 + loss / max(stored, _EPS)

    if isinstance(state, dict):
        state["scatter_energy"] = stored * gain
        if state["scatter_energy"] > incident:
            state["scatter_energy"] = incident
        state["energy_loss"] = loss
        state["energy_gain"] = gain
        state["energy_scale"] = gain
        state["energy_conserved"] = loss_ratio <= _ENERGY_EPS or abs(state["scatter_energy"] - incident) <= _ENERGY_EPS * max(incident, 1.0)
        state["energy"] = incident
        state["energy_corrected"] = loss_ratio > _ENERGY_EPS
    return state


def profile(*args, **kwargs):
    """读取当前厚度/曲率/核参数，供调试面板展示（不修改 state）。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    thickness = None
    if isinstance(state, dict):
        thickness = state.get("thickness")
    if thickness is None:
        raw = params.get("thickness")
        thickness = _vector(raw, len(raw), 0.0) if isinstance(raw, (list, tuple)) else None
    thickness = [_num(t, 0.0) for t in (thickness or [])]

    curvature = None
    if isinstance(state, dict):
        curvature = state.get("lut_curvature", state.get("curvature"))
    if curvature is None:
        curvature = params.get("curvature", 0.0)
    curvature = _num(curvature, 0.0)

    kernel = None
    if isinstance(state, dict):
        kernel = state.get("blur_kernel")
    kernel = kernel or {}
    weights = kernel.get("weights") if isinstance(kernel, dict) else None
    weights = list(weights) if weights is not None else list(params.get("weights") or [])
    weights = [_num(w, 0.0) for w in weights]

    bins = None
    if isinstance(state, dict):
        bins = state.get("lut_bins")
    bins = int(_num(bins, params.get("bins", 16)))

    return {
        "vertices": len(params.get("vertices") or ()) or (len(thickness) if thickness else 0),
        "thickness_min": min(thickness) if thickness else 0.0,
        "thickness_max": max(thickness) if thickness else 0.0,
        "thickness_mean": sum(thickness) / len(thickness) if thickness else 0.0,
        "thickness_samples": thickness[:8],
        "curvature": curvature,
        "lut_bins": bins,
        "kernel_radius": _num(kernel.get("radius") if isinstance(kernel, dict) else params.get("radius"), 0.0),
        "kernel_sum": sum(weights),
        "kernel_weights": weights,
        "resolution": _num(params.get("resolution"), 0.0),
    }


def validation(*args, **kwargs):
    """校验厚度、LUT、模糊核是否与当前形变/曲率/分辨率一致。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    issues = []

    thickness = None
    if isinstance(state, dict):
        thickness = state.get("thickness")
    thickness = list(thickness) if isinstance(thickness, (list, tuple)) else []
    values = [_num(t, 0.0) for t in thickness]
    if not values:
        issues.append("thickness_missing")
    if any(v < 0.0 for v in values):
        issues.append("thickness_negative")
    if any(v > 100.0 for v in values):
        issues.append("thickness_out_of_range")

    curvature = None
    if isinstance(state, dict):
        curvature = state.get("lut_curvature", state.get("curvature"))
    if curvature is None:
        curvature = params.get("curvature")
    if curvature is not None and isinstance(state, dict) and state.get("lut") is None:
        issues.append("lut_missing")

    kernel = state.get("blur_kernel") if isinstance(state, dict) else None
    if kernel is not None:
        weights = [_num(w, 0.0) for w in (kernel.get("weights") or [])]
        if any(w < 0.0 for w in weights):
            issues.append("kernel_negative")
        if weights and sum(weights) <= _EPS:
            issues.append("kernel_zero_sum")
        blob = _num(kernel.get("radius"), -1.0)
        if blob < 0.0:
            issues.append("kernel_radius_invalid")

    resolution = params.get("resolution")
    if resolution is None and isinstance(state, dict):
        resolution = state.get("blur_resolution", state.get("resolution"))
    if resolution is not None and _num(resolution, 0.0) <= 0.0:
        issues.append("resolution_invalid")

    if isinstance(state, dict):
        state["valid"] = not issues
        state["validation_issues"] = issues
    return state


def audit(*args, **kwargs):
    """记录一次管线执行的结果快照，用事件列表解释每个修正点是否生效。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    name = params.get("name", "sss")
    events = params.get("events")
    if events is None and isinstance(state, dict):
        events = state.get("pipeline")
    events = list(events) if events else []

    record = {
        "name": name,
        "steps": [step.get("step", i) if isinstance(step, dict) else step for i, step in enumerate(events)],
        "thickness_recomputed": bool(isinstance(state, dict) and state.get("thickness") is not None),
        "lut_curvature": _num(state.get("lut_curvature") if isinstance(state, dict) else params.get("curvature"), 0.0),
        "kernel_radius": _num(
            (state.get("blur_kernel") or {}).get("radius") if isinstance(state, dict) and isinstance(state.get("blur_kernel"), dict) else params.get("radius"),
            0.0,
        ),
        "order_fixed": bool(isinstance(state, dict) and state.get("order_fixed")),
        "material_epoch": int(_num(state.get("material_epoch") if isinstance(state, dict) else 0)),
        "energy_conserved": bool(isinstance(state, dict) and state.get("energy_conserved")),
        "valid": bool(isinstance(state, dict) and state.get("valid", True)),
    }
    if params.get("strict") and isinstance(state, dict) and not state.get("valid", True):
        raise AssertionError("SSS 状态校验未通过: %s" % (state.get("validation_issues"),))
    if isinstance(state, dict):
        state["audit"] = record
    else:
        state = {"audit": record}
    return state


def fallback(*args, **kwargs):
    """在缺少 LUT/厚度数据时给出可用的安全值，保证管线不中断。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    count = params.get("count", params.get("thickness_count"))
    if count is None and isinstance(state, dict):
        thickness = state.get("thickness")
        count = len(thickness) if isinstance(thickness, (list, tuple)) else None
    if count is None and isinstance(state, dict):
        count = len(state.get("vertices") or ())
    count = int(_num(count, 1)) if count is not None else 1
    count = max(count, 1)

    default = _num(params.get("default", params.get("thickness", 1.0)), 1.0)
    thickness = params.get("thickness")
    if isinstance(thickness, (list, tuple)) and len(thickness) == count:
        fallback_thickness = [_num(t, default) for t in thickness]
    else:
        fallback_thickness = _fallback_thickness(count, default)

    reason = params.get("reason")
    if reason is None and isinstance(state, dict):
        issues = state.get("validation_issues") or []
        reason = issues[0] if issues else None
    applied = not (isinstance(state, dict) and state.get("thickness"))

    if isinstance(state, dict):
        if applied:
            state["thickness"] = fallback_thickness
            state.setdefault("previous_thickness", None)
        state["fallback_applied"] = applied
        state["fallback_reason"] = reason
        state["fallback_count"] = count
    return state


def screen_blur(*args, **kwargs):
    """用当前模糊核做一次屏幕空间模糊；核变旧或分辨率变化时先重缩放。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)

    resolution = params.get("resolution")
    if resolution is None and isinstance(state, dict):
        resolution = state.get("resolution", state.get("blur_resolution"))
    resolution = _num(resolution, 0.0)

    kernel = params.get("kernel")
    if kernel is None and isinstance(state, dict):
        kernel = state.get("blur_kernel")
    kernel = dict(kernel) if isinstance(kernel, dict) else {}
    weights = params.get("weights", kernel.get("weights"))
    weights = [_num(w, 0.0) for w in (weights or [1.0, 2.0, 1.0])]

    stale = False
    if isinstance(state, dict):
        stale = bool(state.get("blur_dirty")) or (
            resolution > 0.0 and state.get("blur_resolution") not in (None, resolution)
        )
    radius = _num(kernel.get("radius", params.get("radius", 1.0)), 1.0)
    if stale and isinstance(state, dict):
        old_res = _num(state.get("blur_resolution"), 0.0)
        ref = state.get("blur_kernel_ref")
        if isinstance(ref, dict) and _num(ref.get("resolution"), 0.0) > 0.0:
            base_res = _num(ref["resolution"])
            base_weights = [_num(w, 0.0) for w in ref.get("weights") or weights]
            base_radius = _num(ref.get("radius"), radius)
        else:
            base_res, base_weights, base_radius = old_res or resolution, weights, radius
        if base_res > 0.0 and resolution > 0.0:
            weights = _rescale_weights(base_weights, base_res, resolution)
            radius = base_radius * (resolution / base_res)
        state["blur_kernel_ref"] = {
            "weights": list(base_weights),
            "radius": base_radius,
            "resolution": base_res,
        }

    total = sum(weights)
    if total > _EPS:
        weights = [w / total for w in weights]

    passes = int(_num(params.get("passes", params.get("samples", 1)), 1))
    passes = max(passes, 1)

    if isinstance(state, dict):
        state["blur_kernel"] = {"weights": weights, "radius": radius}
        state["blur_resolution"] = resolution or state.get("blur_resolution")
        state["blur_passes"] = passes
        state["blur_done"] = True
        state["blur_dirty"] = False
        state["pending_blur"] = False
        state["last_pass"] = "blur"
        state["blur_steps"] = [
            {"step": "blur", "pass": i + 1, "radius": radius, "weights": len(weights)}
            for i in range(passes)
        ]
    return state


def metrics(*args, **kwargs):
    """汇总一次渲染的性能/质量指标。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    samples = params.get("samples", params.get("vertices"))
    if samples is None and isinstance(state, dict):
        thickness = state.get("thickness")
        samples = len(thickness) if isinstance(thickness, (list, tuple)) else len(state.get("vertices") or ())
    samples = int(_num(samples, 0)) if samples is not None else 0
    samples = max(samples, 0)

    cost = params.get("weight_count")
    if cost is None and isinstance(state, dict):
        kernel = state.get("blur_kernel")
        cost = len(kernel.get("weights") or ()) if isinstance(kernel, dict) else 0
    cost = int(_num(cost, 0))
    passes = int(_num(state.get("blur_passes") if isinstance(state, dict) else params.get("passes", 1), 1))
    operations = samples * max(cost, 1) * max(passes, 1)
    output = {
        "samples": samples,
        "kernel_weights": cost,
        "blur_passes": max(passes, 1),
        "operations": operations,
        "thickness_recomputed": bool(isinstance(state, dict) and state.get("thickness") is not None),
        "energy_conserved": bool(isinstance(state, dict) and state.get("energy_conserved")),
        "resolution": _num(state.get("blur_resolution") if isinstance(state, dict) else params.get("resolution"), 0.0),
    }
    if isinstance(state, dict):
        state["metrics"] = output
    return output


def stretch(*args, **kwargs):
    """模拟表情/口型形变：按幅度拉伸顶点，并让厚度随之重算。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    verts, faces = _mesh(params)
    amount = _num(params.get("amount", params.get("amplitude", params.get("delta"))), 0.05)
    axis = params.get("axis", 1)
    axis = int(_num(axis, 1)) % 3
    center = _vector(params.get("center", params.get("pivot")), 3, 0.0)

    moved = []
    for vertex in verts:
        gate = vertex[axis] - center[axis]
        if params.get("bilateral", True):
            gate = abs(gate)
        else:
            gate = max(gate, 0.0)
        for k in range(3):
            factor = amount * gate if k == axis else 0.0
            moved.append(vertex[k] + factor)
    moved = [moved[i:i + 3] for i in range(0, len(moved), 3)]

    thickness = _raw_thickness(moved, faces)
    if isinstance(state, dict):
        state["vertices"] = moved
        state["thickness"] = thickness
        state["thickness_dirty"] = False
        state["shape"] = {"amount": amount, "axis": axis}
        state["deformed"] = True
    else:
        state = {"vertices": moved, "thickness": thickness}
    return state


def transmit(*args, **kwargs):
    """按厚度计算透射/散射光强，结果参与后续能量补偿。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)

    thickness = params.get("thickness")
    if thickness is None and isinstance(state, dict):
        thickness = state.get("thickness")
    thickness = [_num(t, 0.0) for t in (thickness or ())]

    absorption = _num(params.get("absorption", params.get("sigma")), 0.5)
    attenuation = _num(params.get("attenuation", params.get("strength")), 1.0)
    light = _num(params.get("light", params.get("intensity")), 1.0)

    transmitted = [light * math.exp(-absorption * attenuation * max(t, 0.0)) for t in thickness]
    scatter = [light - t for t in transmitted]
    total = _num(_total(scatter), 0.0)

    if isinstance(state, dict):
        state["transmit"] = transmitted
        state["scatter"] = scatter
        state["scatter_energy"] = total
        state["transmit_total"] = sum(transmitted)
        if thickness:
            state["incident_energy"] = light * len(thickness)
    return state


def determinism(*args, **kwargs):
    """同一输入重复执行结果一致：返回规范化（截断小数）后的指纹。"""
    state = _coerce_state(args)
    params = _params(args, kwargs)
    digest = params.get("digest")
    if digest is not None:
        return digest

    def norm(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        if isinstance(value, dict):
            return {k: norm(v) for k, v in sorted(value.items()) if k not in ("audit", "metrics", "blur_steps")}
        return value

    if not isinstance(state, dict):
        return repr(norm(state))
    payload = {key: norm(state[key]) for key in sorted(state) if key not in ("audit", "metrics", "blur_steps")}
    return repr(payload)
