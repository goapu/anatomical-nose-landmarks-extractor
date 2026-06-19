#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

noseLandmarksExtraction.py
==========================
Author: Dilip Goswami, MSc in Geodesy and Geoinformation Sciennce, TU Berlin. 

3D Anatomical Nose Landmark and Surface Template Extractor.

Execution Summary
-----------------
This script extracts a reproducible nasal region of interest (ROI), proposing named anatomical landmarks and dense surface guide templates from 3D human face meshes.

Methodology
-----------
The tool relies exclusively on local 3D mesh geometry (OBJ, PLY, STL) rather than 2D face detectors. It utilizes surface curvature and anterior envelope filtering to identify stable landmarks (such as the nasion, pronasale, and alare) while generating dense structural guides like surface grids, alar crease curves, and nostril rim loops.

Outputs
-------
* Data Files: anatomical_nose_landmarks_3d.csv, dense_nose_template_points.csv, anatomical_nose_landmarks_review.json, and measurements_review_required.json.
* Point Clouds (.ply): Isolated 3D visual extractions for landmarks, dense templates, and the nose surface patch.
* Review Imagery: nose_landmarks_overlay.png for visual auditing.
"""

import os
import csv
import json
import copy
import argparse
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageDraw

try:
    import open3d as o3d
except ImportError:
    raise SystemExit(
        "Missing open3d. Install with:\n"
        "python -m pip install open3d numpy pillow\n"
    )

# ============================================================
# Vector helpers
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def axis_vector(axis_name):
    axis_name = axis_name.strip().lower()
    axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    if axis_name not in axes:
        raise ValueError("Axis must be one of: +x, -x, +y, -y, +z, -z")
    return axes[axis_name]


def get_camera_basis(front_axis_name="+z", up_axis_name="-y"):
    front = normalize(axis_vector(front_axis_name))
    up = normalize(axis_vector(up_axis_name))
    right = normalize(np.cross(front, up))
    if np.linalg.norm(right) < 1e-9:
        raise RuntimeError("front and up axes cannot be parallel")
    up = normalize(np.cross(right, front))
    return right, up, front


def project_points(points, right, up, front):
    return points @ right, points @ up, points @ front


def percentile_range(a, lo=1, hi=99):
    return float(np.percentile(a, hi) - np.percentile(a, lo))

# ============================================================
# Mesh I/O and neighborhood helpers
# ============================================================

def load_mesh(mesh_path, subdivision=0):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {mesh_path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if subdivision > 0:
        mesh = mesh.subdivide_loop(number_of_iterations=subdivision)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh, np.asarray(mesh.vertices, dtype=np.float64)


def clean_detection_points(vertices, nb_neighbors=35, std_ratio=1.8):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    clean_points = np.asarray(clean_pcd.points, dtype=np.float64)
    return vertices.copy() if len(clean_points) < 500 else clean_points


def build_kdtree(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, o3d.geometry.KDTreeFlann(pcd)


def radius_neighbors(tree, point, radius):
    _, idx, _ = tree.search_radius_vector_3d(np.asarray(point, dtype=np.float64), float(radius))
    return np.asarray(idx, dtype=np.int64)


def local_pca_curvature_score(points):
    if len(points) < 10:
        return 0.0
    c = points - np.mean(points, axis=0)
    cov = (c.T @ c) / max(len(points) - 1, 1)
    vals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
    return float(vals[0] / (np.sum(vals) + 1e-12))


def vertex_adjacency_from_triangles(n_vertices, triangles):
    adj = [set() for _ in range(n_vertices)]
    edge_count = defaultdict(int)
    for tri in triangles:
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            adj[u].add(v); adj[v].add(u)
            edge_count[tuple(sorted((u, v)))] += 1
    return [np.array(sorted(s), dtype=np.int64) for s in adj], edge_count

# ============================================================
# Optional symmetry-based pose normalization
# ============================================================

def rotation_matrix_to_axis_angle(R):
    trace = np.trace(R)
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if abs(angle) < 1e-10:
        return np.array([0.0, 1.0, 0.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = normalize(axis / (2.0 * np.sin(angle) + 1e-12))
    return axis, angle


def axis_angle_to_rotation_matrix(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def normalize_pose_by_icp_symmetry(mesh, detection_points, args):
    # For medical work this is disabled by default because it can reduce true asymmetry.
    points = np.asarray(detection_points, dtype=np.float64)
    if len(points) < 800:
        return mesh, detection_points, np.eye(4)

    right, up, front = get_camera_basis(args.front, args.up)
    basis = np.stack([right, up, front], axis=1)
    centroid = np.mean(points, axis=0)
    target_canonical = (points - centroid) @ basis
    source_canonical = target_canonical.copy()
    source_canonical[:, 0] *= -1.0

    source_pcd = o3d.geometry.PointCloud(); target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_canonical)
    target_pcd.points = o3d.utility.Vector3dVector(target_canonical)

    diag = np.percentile(np.linalg.norm(target_canonical, axis=1), 90)
    voxel = max(diag * args.icp_voxel_ratio, 1e-6)
    source_down = source_pcd.voxel_down_sample(voxel)
    target_down = target_pcd.voxel_down_sample(voxel)
    if len(source_down.points) < 250 or len(target_down.points) < 250:
        source_down, target_down = source_pcd, target_pcd

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel * args.icp_threshold_factor, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.icp_iterations)
    )
    axis, angle = rotation_matrix_to_axis_angle(result.transformation[:3, :3])
    R_half = axis_angle_to_rotation_matrix(axis, angle * args.icp_half_factor)
    R_correct_canonical = R_half.T

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices_new = ((R_correct_canonical @ ((vertices - centroid) @ basis).T).T @ basis.T) + centroid
    mesh_norm = copy.deepcopy(mesh)
    mesh_norm.vertices = o3d.utility.Vector3dVector(vertices_new)
    mesh_norm.compute_vertex_normals(); mesh_norm.compute_triangle_normals()

    detection_new = ((R_correct_canonical @ target_canonical.T).T @ basis.T) + centroid
    R_world = basis @ R_correct_canonical @ basis.T
    T_world = np.eye(4)
    T_world[:3, :3] = R_world
    T_world[:3, 3] = centroid - R_world @ centroid
    return mesh_norm, detection_new, T_world

# ============================================================
# Rough ROI detection: MediaPipe-like initialization from mesh
# ============================================================

def validate_tip_candidate(candidate_point, clean_points, tree, radius, front):
    idx = radius_neighbors(tree, candidate_point, radius)
    if len(idx) < 40:
        return -1e18
    neigh = clean_points[idx]
    center = np.mean(neigh, axis=0)
    spread = np.percentile(np.linalg.norm(neigh - center, axis=1), 90)
    if spread < radius * 0.16:
        return -1e18
    tip_d = float(candidate_point @ front)
    behind_ratio = float(np.mean((neigh @ front) < tip_d + radius * 0.12))
    if behind_ratio < 0.62:
        return -1e18
    curvature = local_pca_curvature_score(neigh)
    return tip_d + curvature * radius * 8.0 + behind_ratio * radius


def estimate_nose_region(clean_points, args):
    right, up, front = get_camera_basis(args.front, args.up)
    x, y, z = project_points(clean_points, right, up, front)
    x01, x99 = np.percentile(x, 1), np.percentile(x, 99)
    y01, y99 = np.percentile(y, 1), np.percentile(y, 99)
    global_w, global_h = float(x99 - x01), float(y99 - y01)
    global_cx = float(np.median(x))

    central = np.abs(x - global_cx) < global_w * args.central_band
    vertical = (y > y01 + global_h * args.vertical_low) & (y < y01 + global_h * args.vertical_high)
    candidate_mask = central & vertical
    if candidate_mask.sum() < 250:
        candidate_mask = vertical
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) == 0:
        raise RuntimeError("No candidate points found for rough nose detection")

    _, tree = build_kdtree(clean_points)
    order = candidate_idx[np.argsort(z[candidate_idx])[::-1]]
    validation_radius = global_w * args.tip_validation_radius
    best_idx, best_score = None, -1e18
    for idx in order[:args.tip_candidates]:
        score = validate_tip_candidate(clean_points[int(idx)], clean_points, tree, validation_radius, front)
        if score > best_score:
            best_idx, best_score = int(idx), score
    if best_idx is None:
        raise RuntimeError("Could not validate a nose tip candidate")

    tip_point = clean_points[best_idx]
    tip_x, tip_y, tip_z = float(tip_point @ right), float(tip_point @ up), float(tip_point @ front)

    local_y_band = np.abs(y - tip_y) < global_h * args.local_width_y_band
    if local_y_band.sum() > 100:
        z_cut = np.percentile(z[local_y_band], 55)
        local_face = local_y_band & (z >= z_cut)
        if local_face.sum() > 80:
            local_w = float(np.percentile(x[local_face], 95) - np.percentile(x[local_face], 5))
        else:
            local_w = global_w * 0.24
    else:
        local_w = global_w * 0.24
    local_w = float(np.clip(local_w, global_w * args.local_width_min, global_w * args.local_width_max))

    nose_h = local_w * args.nose_height_ratio
    nose_w = local_w * args.nose_width_ratio
    top_y = tip_y + nose_h * args.top_ratio
    bottom_y = tip_y - nose_h * args.bottom_ratio
    max_h = local_w * args.max_nose_height_factor
    if (top_y - bottom_y) > max_h:
        center_y = tip_y + max_h * 0.05
        top_y = center_y + max_h * 0.55
        bottom_y = center_y - max_h * 0.45
    bottom_y -= nose_h * args.bottom_extension

    return {
        "right": right, "up": up, "front": front,
        "global_w": global_w, "global_h": global_h, "local_face_w": local_w,
        "tip_point": tip_point, "tip_x": tip_x, "tip_y": tip_y, "tip_z": tip_z,
        "top_y": float(top_y), "bottom_y": float(bottom_y),
        "nose_h": float(top_y - bottom_y), "nose_w": float(nose_w),
        "rough_detector": "mesh_geometry_mediapipe_like_roi"
    }

# ============================================================
# Nose patch and surface context
# ============================================================

def create_surface_context(mesh):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    mesh.compute_triangle_normals(); mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    adj, edge_count = vertex_adjacency_from_triangles(len(vertices), triangles)
    return {
        "scene": scene,
        "vertices": vertices,
        "triangles": triangles,
        "triangle_normals": np.asarray(mesh.triangle_normals, dtype=np.float64),
        "vertex_normals": np.asarray(mesh.vertex_normals, dtype=np.float64),
        "adjacency": adj,
        "edge_count": edge_count,
    }


def width_at_y(yv, region, args):
    top_y = region["top_y"]; bottom_y = region["bottom_y"]; tip_y = region["tip_y"]
    base_w = region["nose_w"]
    bridge_w = base_w * args.bridge_width_factor
    mid_w = base_w * args.mid_width_factor
    bottom_w = base_w * args.bottom_width_factor
    if yv >= tip_y:
        t = np.clip((top_y - yv) / max(top_y - tip_y, 1e-9), 0.0, 1.0)
        return bridge_w * (1.0 - t) + mid_w * t
    t = np.clip((tip_y - yv) / max(tip_y - bottom_y, 1e-9), 0.0, 1.0)
    return mid_w * (1.0 - t) + bottom_w * t


def compute_vertex_curvature(vertices, adjacency):
    curv = np.zeros(len(vertices), dtype=np.float64)
    for i, nb in enumerate(adjacency):
        if len(nb) < 3:
            continue
        neigh = vertices[nb]
        lap = np.mean(neigh, axis=0) - vertices[i]
        curv[i] = np.linalg.norm(lap)
    if np.percentile(curv, 99) > 1e-12:
        curv = curv / (np.percentile(curv, 99) + 1e-12)
    return np.clip(curv, 0.0, 1.0)


def largest_connected_component_with_seed(candidate_idx, adjacency, seed_idx):
    """Keep the candidate patch connected to the nose tip, preventing back-of-skull hits."""
    cand = set(map(int, candidate_idx))
    if not cand:
        return np.array([], dtype=np.int64)
    seed_idx = int(seed_idx)
    if seed_idx not in cand:
        # Use the candidate closest to the seed if the exact seed is outside the ROI.
        seed_idx = min(cand, key=lambda i: abs(i - seed_idx))
    visited = set([seed_idx])
    dq = deque([seed_idx])
    while dq:
        u = dq.popleft()
        for v in adjacency[u]:
            if v in cand and v not in visited:
                visited.add(v)
                dq.append(v)
    return np.array(sorted(visited), dtype=np.int64)


def anterior_envelope_filter(idx, x, y, z, region, args):
    """
    For each horizontal Y slice, keep only vertices close to the most anterior surface.
    This removes vertices projected through the same X/Y window on the back of the head.
    """
    if len(idx) == 0:
        return idx
    kept = []
    bins = np.linspace(region["bottom_y"] - region["nose_h"] * 0.10,
                       region["top_y"] + region["nose_h"] * 0.10,
                       max(args.patch_envelope_bins, 4))
    for lo, hi in zip(bins[:-1], bins[1:]):
        row = idx[(y[idx] >= lo) & (y[idx] < hi)]
        if len(row) == 0:
            continue
        z_front = np.percentile(z[row], args.patch_front_percentile)
        # Allow slightly more depth around alar/nostril level, but not enough to include lips/skull.
        yy = 0.5 * (lo + hi)
        local_slack = width_at_y(yy, region, args) * args.patch_front_slack_factor
        kept.append(row[z[row] >= z_front - local_slack])
    if not kept:
        return idx[:0]
    return np.unique(np.concatenate(kept)).astype(np.int64)


def make_nose_patch(surface_ctx, region, args):
    vertices = surface_ctx["vertices"]
    x, y, z = project_points(vertices, region["right"], region["up"], region["front"])
    cx = region["tip_x"]

    # Stop higher than the upper lip by default. The subnasale/nostril refinement still uses lower-nose data.
    y_margin_top = region["nose_h"] * args.patch_y_margin
    y_margin_bottom = region["nose_h"] * args.patch_bottom_margin
    y_low = region["bottom_y"] + region["nose_h"] * args.patch_bottom_raise - y_margin_bottom
    y_high = region["top_y"] + y_margin_top

    # Anatomical tapered asymmetric width. Wider defaults help avoid missing one ala; anterior-envelope + connectivity
    # prevent the wider window from leaking to cheek/back/lip.
    local_half = np.array([width_at_y(yy, region, args) * args.patch_width_factor for yy in y])
    left_half = local_half * args.patch_left_factor
    right_half = local_half * args.patch_right_factor
    dx = x - cx
    in_width = ((dx < 0) & (np.abs(dx) <= left_half)) | ((dx >= 0) & (np.abs(dx) <= right_half))
    in_patch = in_width & (y <= y_high) & (y >= y_low)
    idx = np.where(in_patch)[0].astype(np.int64)

    # Remove back-of-skull / interior projected points using local anterior envelope.
    idx = anterior_envelope_filter(idx, x, y, z, region, args)

    # Remove disconnected islands; retain only the component attached to the nose tip.
    if len(idx) > 0:
        tip_seed = nearest_vertex(vertices, region["tip_point"])
        idx = largest_connected_component_with_seed(idx, surface_ctx["adjacency"], tip_seed)

    # If the connected component becomes too small, fall back to envelope-only but still not global posterior points.
    if len(idx) < args.patch_min_vertices:
        raw = np.where(in_patch)[0].astype(np.int64)
        idx = anterior_envelope_filter(raw, x, y, z, region, args)
    return idx


def nearest_vertex(vertices, point):
    return int(np.argmin(np.linalg.norm(vertices - point, axis=1)))


def choose_point(candidates, score, fallback=None):
    if len(candidates) == 0:
        return fallback, 0.0
    values = score(candidates)
    best_local = int(np.argmax(values))
    conf = float(np.clip((values[best_local] - np.percentile(values, 50)) / (np.ptp(values) + 1e-9), 0.0, 1.0))
    return candidates[best_local], conf

# ============================================================
# Anatomical landmark extraction
# ============================================================

def local_midline_curve(vertices, idx, x, y, z, cx, y_min, y_max, n_samples, half_width, mode="max_z"):
    pts = []
    ys = np.linspace(y_min, y_max, n_samples)
    for yy in ys:
        band = idx[(np.abs(y[idx] - yy) < (y_max - y_min) / max(n_samples, 1) * 0.65) & (np.abs(x[idx] - cx) < half_width)]
        if len(band) == 0:
            continue
        if mode == "min_z":
            chosen = band[np.argmin(z[band])]
        else:
            # prefer central and anterior
            score = z[band] - 0.6 * np.abs(x[band] - cx)
            chosen = band[np.argmax(score)]
        pts.append(vertices[int(chosen)])
    return np.array(pts, dtype=np.float64)


def find_nasion_radix(vertices, idx, x, y, z, region, args):
    cx = region["tip_x"]
    upper = idx[(np.abs(x[idx] - cx) < region["nose_w"] * 0.22) &
                (y[idx] > region["tip_y"] + region["nose_h"] * 0.10) &
                (y[idx] < region["top_y"] + region["nose_h"] * 0.15)]
    if len(upper) == 0:
        return region["tip_point"], 0.0, "fallback_tip"
    # Nasion/radix is a posterior concavity on the upper midline.
    central_penalty = np.abs(x[upper] - cx) / max(region["nose_w"], 1e-9)
    y_pref = 1.0 - np.abs(y[upper] - region["top_y"]) / max(region["nose_h"], 1e-9)
    values = (-z[upper]) - 0.25 * central_penalty + 0.10 * y_pref
    best = int(upper[np.argmax(values)])
    conf = float(np.clip((np.percentile(z[upper], 85) - z[best]) / (np.ptp(z[upper]) + 1e-9), 0.0, 1.0))
    return vertices[best], conf, "upper_midline_posterior_concavity"


def find_pronasale(vertices, idx, x, y, z, region):
    cx = region["tip_x"]
    central = idx[(np.abs(x[idx] - cx) < region["nose_w"] * 0.28) &
                  (y[idx] < region["top_y"]) &
                  (y[idx] > region["bottom_y"])]
    if len(central) == 0:
        return region["tip_point"], 0.0, "fallback_roi_tip"
    score = z[central] - 0.15 * np.abs(x[central] - cx)
    best = int(central[np.argmax(score)])
    conf = float(np.clip((z[best] - np.percentile(z[central], 75)) / (np.ptp(z[central]) + 1e-9), 0.0, 1.0))
    return vertices[best], conf, "max_anterior_central_nose_patch"


def find_subnasale_and_columella(vertices, idx, x, y, z, region, args):
    cx = region["tip_x"]
    lower = idx[(np.abs(x[idx] - cx) < region["nose_w"] * 0.22) &
                (y[idx] < region["tip_y"] - region["nose_h"] * 0.02) &
                (y[idx] > region["bottom_y"] - region["nose_h"] * 0.08)]
    if len(lower) == 0:
        return region["tip_point"], region["tip_point"], 0.0, 0.0
    # Columella: anterior midline point below tip; subnasale: posterior concavity below columella.
    col_band = lower[y[lower] > np.percentile(y[lower], 45)]
    if len(col_band) == 0:
        col_band = lower
    col_score = z[col_band] - 0.20 * np.abs(x[col_band] - cx)
    col_idx = int(col_band[np.argmax(col_score)])

    sub_band = lower[y[lower] < y[col_idx] + region["nose_h"] * 0.10]
    if len(sub_band) == 0:
        sub_band = lower
    sub_score = -z[sub_band] - 0.30 * np.abs(x[sub_band] - cx)
    sub_idx = int(sub_band[np.argmax(sub_score)])

    col_conf = float(np.clip((z[col_idx] - np.percentile(z[col_band], 70)) / (np.ptp(z[col_band]) + 1e-9), 0.0, 1.0))
    sub_conf = float(np.clip((np.percentile(z[sub_band], 80) - z[sub_idx]) / (np.ptp(z[sub_band]) + 1e-9), 0.0, 1.0))
    return vertices[sub_idx], vertices[col_idx], sub_conf, col_conf


def find_alare_subalare(vertices, idx, x, y, z, region, args):
    tip_y = region["tip_y"]; bottom_y = region["bottom_y"]; nose_h = region["nose_h"]; cx = region["tip_x"]
    alar_band = idx[(y[idx] > bottom_y + nose_h * 0.08) &
                    (y[idx] < tip_y + nose_h * 0.18) &
                    (np.abs(x[idx] - cx) > region["nose_w"] * 0.12)]
    left = alar_band[x[alar_band] < cx]
    right = alar_band[x[alar_band] > cx]

    def side_points(side_idx, side):
        if len(side_idx) == 0:
            return region["tip_point"], region["tip_point"], 0.0, 0.0
        if side == "left":
            al_idx = int(side_idx[np.argmin(x[side_idx])])
            lower = side_idx[y[side_idx] < y[al_idx] + nose_h * 0.12]
            sb_idx = int(lower[np.argmin(x[lower] + 0.35 * z[lower])]) if len(lower) else al_idx
        else:
            al_idx = int(side_idx[np.argmax(x[side_idx])])
            lower = side_idx[y[side_idx] < y[al_idx] + nose_h * 0.12]
            sb_idx = int(lower[np.argmax(x[lower] - 0.35 * z[lower])]) if len(lower) else al_idx
        al_conf = float(np.clip(abs(x[al_idx] - cx) / max(region["nose_w"], 1e-9), 0.0, 1.0))
        sb_conf = float(np.clip(abs(x[sb_idx] - cx) / max(region["nose_w"], 1e-9), 0.0, 1.0))
        return vertices[al_idx], vertices[sb_idx], al_conf, sb_conf

    left_alare, left_subalare, lac, lsc = side_points(left, "left")
    right_alare, right_subalare, rac, rsc = side_points(right, "right")
    return left_alare, right_alare, left_subalare, right_subalare, lac, rac, lsc, rsc


def find_alar_crease_curves(vertices, idx, x, y, z, normals, curv, region, args):
    cx = region["tip_x"]; nose_h = region["nose_h"]
    y0 = region["bottom_y"] + nose_h * 0.12
    y1 = region["tip_y"] + nose_h * 0.24
    ys = np.linspace(y0, y1, args.crease_curve_points)
    curves = {"left_alar_crease_curve": [], "right_alar_crease_curve": []}
    points = {}

    for side_name, sign in (("left", -1.0), ("right", 1.0)):
        curve = []
        for yy in ys:
            local = idx[(np.abs(y[idx] - yy) < nose_h * 0.035) &
                        ((x[idx] - cx) * sign > region["nose_w"] * 0.18) &
                        ((x[idx] - cx) * sign < region["nose_w"] * 1.05)]
            if len(local) == 0:
                continue
            n_right = normals[local] @ region["right"]
            # crease: high curvature, lateral-facing normal, and posterior valley.
            z_valley = (np.percentile(z[local], 85) - z[local]) / (np.ptp(z[local]) + 1e-9)
            score = 1.1 * curv[local] + 0.7 * z_valley + 0.45 * np.maximum(0.0, sign * n_right)
            best = int(local[np.argmax(score)])
            curve.append(vertices[best])
        curve = np.array(curve, dtype=np.float64)
        curves[f"{side_name}_alar_crease_curve"] = curve
        if len(curve):
            mid = curve[len(curve)//2]
            points[f"{side_name}_alar_crease"] = mid
        else:
            points[f"{side_name}_alar_crease"] = region["tip_point"]
    return curves, points

# ============================================================
# Nostril rim detection
# ============================================================

def boundary_loops_in_roi(surface_ctx, patch_idx_set):
    edge_count = surface_ctx["edge_count"]
    boundary_edges = [(u, v) for (u, v), c in edge_count.items() if c == 1 and u in patch_idx_set and v in patch_idx_set]
    graph = defaultdict(list)
    for u, v in boundary_edges:
        graph[u].append(v); graph[v].append(u)
    visited = set()
    loops = []
    for start in graph:
        if start in visited:
            continue
        comp = []
        dq = deque([start]); visited.add(start)
        while dq:
            u = dq.popleft(); comp.append(u)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v); dq.append(v)
        if len(comp) >= 8:
            loops.append(np.array(comp, dtype=np.int64))
    return loops


def split_loop_cardinal(vertices, loop, x, y, cx, side):
    lx, ly = x[loop], y[loop]
    if side == "left":
        medial = int(loop[np.argmax(lx)])
        lateral = int(loop[np.argmin(lx)])
    else:
        medial = int(loop[np.argmin(lx)])
        lateral = int(loop[np.argmax(lx)])
    superior = int(loop[np.argmax(ly)])
    inferior = int(loop[np.argmin(ly)])
    return {
        "medial": vertices[medial],
        "lateral": vertices[lateral],
        "superior": vertices[superior],
        "inferior": vertices[inferior],
        "loop_points": vertices[loop],
    }


def fallback_nostril_from_depth(vertices, idx, x, y, z, curv, region, side, args):
    cx = region["tip_x"]; sign = -1.0 if side == "left" else 1.0
    local = idx[(y[idx] > region["bottom_y"] - region["nose_h"] * 0.04) &
                (y[idx] < region["tip_y"] + region["nose_h"] * 0.03) &
                ((x[idx] - cx) * sign > region["nose_w"] * 0.05) &
                ((x[idx] - cx) * sign < region["nose_w"] * 0.70)]
    if len(local) < 8:
        p = region["tip_point"]
        return {"medial": p, "lateral": p, "superior": p, "inferior": p, "loop_points": np.array([p])}, 0.0, "fallback_failed"
    # nostril basin is posterior/deep and often high curvature.
    depth_score = (np.percentile(z[local], 85) - z[local]) / (np.ptp(z[local]) + 1e-9)
    score = 0.85 * depth_score + 0.45 * curv[local]
    center_idx = int(local[np.argmax(score)])
    center = vertices[center_idx]
    dist = np.linalg.norm(vertices[local] - center, axis=1)
    radius = np.percentile(dist, 45)
    rim = local[(dist > radius * 0.65) & (dist < radius * 1.55)]
    if len(rim) < 8:
        rim = local[np.argsort(dist)[:min(len(local), 20)]]
    card = split_loop_cardinal(vertices, rim, x, y, cx, side)
    conf = float(np.clip(np.max(score) - np.median(score), 0.0, 1.0))
    return card, conf, "depth_curvature_basin_fallback"


def find_nostril_rims(surface_ctx, patch_idx, x, y, z, curv, region, args):
    vertices = surface_ctx["vertices"]
    patch_set = set(map(int, patch_idx))
    loops = boundary_loops_in_roi(surface_ctx, patch_set)
    cx = region["tip_x"]
    lower_loops = []
    for loop in loops:
        if np.mean(y[loop]) < region["tip_y"] + region["nose_h"] * 0.05:
            lower_loops.append(loop)

    result = {}
    metadata = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        candidates = [loop for loop in lower_loops if np.mean((x[loop] - cx) * sign) > region["nose_w"] * 0.03]
        if candidates:
            # prefer compact loops in lower nose ROI
            best_loop = max(candidates, key=lambda L: len(L) - 0.25 * abs(np.mean(y[L]) - region["bottom_y"]))
            result[side] = split_loop_cardinal(vertices, best_loop, x, y, cx, side)
            metadata[side] = {"confidence": 0.85, "method": "mesh_boundary_loop"}
        else:
            card, conf, method = fallback_nostril_from_depth(vertices, patch_idx, x, y, z, curv, region, side, args)
            result[side] = card
            metadata[side] = {"confidence": conf, "method": method}
    return result, metadata

# ============================================================
# Dense anatomical surface template
# ============================================================

def sample_curve_by_y(vertices, idx, x, y, z, region, x_center_func, y_values, half_band, prefer_front=True):
    pts = []
    for yy in y_values:
        x_center = x_center_func(yy)
        band = idx[(np.abs(y[idx] - yy) < half_band) & (np.abs(x[idx] - x_center) < half_band * 2.0)]
        if len(band) == 0:
            continue
        score = -np.abs(x[band] - x_center) - np.abs(y[band] - yy)
        if prefer_front:
            score += 0.5 * z[band]
        chosen = int(band[np.argmax(score)])
        pts.append(vertices[chosen])
    return np.array(pts, dtype=np.float64)


def build_dense_anatomical_template(vertices, patch_idx, x, y, z, region, anatomical, curves, nostrils, args):
    dense = {}
    y_values = np.linspace(region["top_y"], region["bottom_y"], args.dense_rows)
    cx = region["tip_x"]

    dense["midline_curve"] = sample_curve_by_y(
        vertices, patch_idx, x, y, z, region,
        lambda yy: cx, y_values, region["nose_h"] * 0.025, True
    )

    for side, sign in (("left", -1.0), ("right", 1.0)):
        dense[f"{side}_sidewall_curve"] = sample_curve_by_y(
            vertices, patch_idx, x, y, z, region,
            lambda yy, sign=sign: cx + sign * width_at_y(yy, region, args) * 0.36,
            y_values, region["nose_h"] * 0.035, True
        )

    dense.update(curves)
    dense["left_nostril_rim_loop"] = nostrils["left"]["loop_points"]
    dense["right_nostril_rim_loop"] = nostrils["right"]["loop_points"]

    # Dense grid only from the anterior connected nose patch. Each row is filtered again against
    # the row's anterior envelope so no sampled point can jump to the back of the skull.
    grid = []
    for r, yy in enumerate(np.linspace(region["top_y"], region["bottom_y"], args.dense_rows)):
        w = width_at_y(yy, region, args) * args.dense_grid_width_factor
        left_w = w * args.dense_left_factor
        right_w = w * args.dense_right_factor
        for c, t in enumerate(np.linspace(-1.0, 1.0, args.dense_cols)):
            xx = cx + (t * left_w if t < 0 else t * right_w)
            band = patch_idx[(np.abs(y[patch_idx] - yy) < region["nose_h"] * args.dense_y_band) &
                             (np.abs(x[patch_idx] - xx) < region["nose_w"] * args.dense_x_band)]
            if len(band) == 0:
                continue
            z_front = np.percentile(z[band], args.dense_front_percentile)
            band2 = band[z[band] >= z_front - width_at_y(yy, region, args) * args.dense_front_slack_factor]
            if len(band2) >= 1:
                band = band2
            score = -1.2 * np.abs(x[band] - xx) - np.abs(y[band] - yy) + 0.25 * z[band]
            best = int(band[np.argmax(score)])
            grid.append({"name": f"surface_grid_r{r:02d}_c{c:02d}", "point": vertices[best]})
    dense["surface_grid"] = grid

    # Controlled columella/lower-nose guide lines. Kept separate from the
    # primary stable nose grid so lower-nose landmarks are not degraded.
    col_lines = build_columella_tip_to_alar_lines(vertices, patch_idx, x, y, z, region, anatomical, args)
    if col_lines:
        dense["columella_tip_to_alar_lines"] = col_lines

    # v29: No vertical columella guide generation.
    #
    # Previous experimental versions attempted to add vertical columns over the
    # columella. In practice those guides were unstable on sparse/occluded lower
    # nose meshes and could be visually misleading. This version intentionally
    # exports only the stable named anatomical landmarks, the dense nasal patch,
    # nostril/alar curves, and optional transverse lower-nose guide lines.
    return dense



# ============================================================
# Controlled columella guide lines
# ============================================================

def _get_anchor_point(anatomical, name, fallback=None):
    rec = anatomical.get(name)
    if rec is None:
        return None if fallback is None else np.asarray(fallback, dtype=np.float64)
    return np.asarray(rec["point"], dtype=np.float64)


def _front_envelope_projection(vertices, patch_idx, x, y, z, target, region, args):
    """Project a designed guide point toward the visible/anterior nose surface.

    Why this function exists
    ------------------------
    The columella guide lines are *template guides*, not core clinical
    landmarks. Earlier versions snapped every guide target to the nearest mesh
    vertex. On low-density meshes this caused many different guide samples to
    collapse onto the same vertex, creating broken or duplicated lines.

    This function uses a safer approach:
      1. Search only inside the already-filtered anterior nose patch.
      2. Keep the local front-envelope points to avoid internal nasal-airway
         vertices and posterior surfaces.
      3. Return a blended surface estimate instead of a single nearest vertex.
      4. Fall back to the designed target if the local surface evidence is weak.

    The result is a smooth, auditable guide line that stays near the external
    mesh surface without damaging the stable anatomical landmark extraction.
    """
    target = np.asarray(target, dtype=np.float64)
    tx = float(target @ region["right"])
    ty = float(target @ region["up"])

    x_band = max(region["nose_w"] * args.columella_x_band, 1e-9)
    y_band = max(region["nose_h"] * args.columella_y_band, 1e-9)

    def local_candidates(scale):
        return patch_idx[
            (np.abs(x[patch_idx] - tx) <= x_band * scale) &
            (np.abs(y[patch_idx] - ty) <= y_band * scale)
        ]

    cand = local_candidates(1.0)
    if len(cand) < args.columella_min_local_candidates:
        cand = local_candidates(1.7)
    if len(cand) == 0:
        return target, False, "designed_target_no_local_surface"

    # Keep only the local anterior envelope. This prevents the guide point from
    # jumping into the nasal airway or onto posterior/internal surfaces.
    z_front = np.percentile(z[cand], args.columella_front_percentile)
    cand_front = cand[z[cand] >= z_front - region["nose_w"] * args.columella_front_slack]
    if len(cand_front) >= 1:
        cand = cand_front

    # Weight by proximity in the projected coordinate system.  A weighted
    # centroid is smoother than snapping to one vertex and greatly reduces
    # duplicate guide samples on sparse meshes.
    dx = (x[cand] - tx) / x_band
    dy = (y[cand] - ty) / y_band
    d2 = dx * dx + dy * dy
    sigma2 = max(args.columella_projection_sigma ** 2, 1e-6)
    weights = np.exp(-0.5 * d2 / sigma2)
    if float(weights.sum()) < 1e-12:
        return target, False, "designed_target_zero_weights"

    surface_estimate = (vertices[cand] * weights[:, None]).sum(axis=0) / weights.sum()

    # Blend rather than hard-snap.  This keeps the designed parallel-line
    # geometry visible while still keeping the guide close to the mesh surface.
    blend = float(np.clip(args.columella_projection_blend, 0.0, 1.0))
    point = target * (1.0 - blend) + surface_estimate * blend
    return point.astype(np.float64), True, "front_envelope_weighted_projection"


def _dedupe_line_points(points, min_spacing):
    """Reduce exact/near-duplicate samples inside one columella guide line.

    If a projected point collapses too close to the previous accepted sample, we
    keep it only when the spacing is above ``min_spacing``. This avoids CSV/PLY
    outputs that contain many distinct IDs but visually occupy the same vertex.
    """
    if min_spacing <= 0 or len(points) <= 1:
        return points
    kept = []
    last = None
    for item in points:
        p = np.asarray(item["point"], dtype=np.float64)
        if last is None or np.linalg.norm(p - last) >= min_spacing:
            kept.append(item)
            last = p
        else:
            # Skip near-duplicate samples. This intentionally makes the exported
            # guide easier to audit: fewer rows, but each visible point carries
            # distinct spatial information.
            continue
    return kept


def build_columella_tip_to_alar_lines(vertices, patch_idx, x, y, z, region, anatomical, args):
    """Build controlled columella guide lines from alar rim toward infratip.

    The generated points are intentionally stored in the dense template group
    ``columella_tip_to_alar_lines``. They are not added to
    ``anatomical_nose_landmarks_3d.csv`` because they are a coverage template
    that should be visually reviewed.

    Geometry model
    --------------
    - The lowest line spans the left/right medial nostril-rim anchors.
    - Additional parallel lines interpolate upward toward the lower tip/infratip.
    - Line width tapers toward the tip.
    - Left/right expansion factors can compensate for asymmetric coverage.
    - A subnasale/lip guard prevents guide lines from falling onto the upper lip.
    - Projection uses front-envelope weighted surface estimates to avoid
      internal nasal-airway vertices and duplicate nearest-vertex snapping.
    """
    if not args.enable_columella_guides:
        return []

    left = _get_anchor_point(anatomical, "left_nostril_medial")
    right = _get_anchor_point(anatomical, "right_nostril_medial")
    tip = _get_anchor_point(anatomical, "pronasale", region["tip_point"])
    sn = _get_anchor_point(anatomical, "subnasale")

    if left is None or right is None or tip is None:
        return []

    base_center = 0.5 * (left + right)
    left_vec = left - base_center
    right_vec = right - base_center

    # Lip guard: keep the lowest transverse line above subnasale/upper lip.
    # Coordinate convention is established by --up. With --up=+y, larger Y is
    # anatomically superior.
    if sn is not None:
        min_y = float(sn @ region["up"]) + region["nose_h"] * args.columella_lip_guard_factor
        base_y = float(base_center @ region["up"])
        if base_y < min_y:
            lift = min_y - base_y
            base_center = base_center + region["up"] * lift
            left = left + region["up"] * lift
            right = right + region["up"] * lift
            left_vec = left - base_center
            right_vec = right - base_center

    # Upper target: blend from medial alar-rim line toward lower tip. Smaller
    # values keep guide lines lower; larger values move them toward pronasale.
    tip_blend = float(np.clip(args.columella_tip_blend, 0.0, 1.0))
    upper_center = base_center * (1.0 - tip_blend) + tip * tip_blend
    upper_center = upper_center - region["up"] * (region["nose_h"] * args.columella_transverse_top_drop)

    n_lines = max(1, int(args.columella_transverse_lines))
    n_samples = max(2, int(args.columella_transverse_samples))
    min_spacing = max(region["nose_w"] * args.columella_duplicate_spacing_factor, 0.0)

    all_points = []
    point_id = 0

    for line_idx, t in enumerate(np.linspace(0.0, 1.0, n_lines)):
        # Lower lines are wider; upper lines taper toward infratip.
        taper = (1.0 - t) ** args.columella_taper_power
        center = base_center * (1.0 - t) + upper_center * t

        width_scale = args.columella_min_width_factor + (1.0 - args.columella_min_width_factor) * taper
        left_half = left_vec * (args.columella_left_expand * width_scale)
        right_half = right_vec * (args.columella_right_expand * width_scale)

        max_half = region["nose_w"] * args.columella_transverse_max_half
        if np.linalg.norm(left_half) > max_half:
            left_half = normalize(left_half) * max_half
        if np.linalg.norm(right_half) > max_half:
            right_half = normalize(right_half) * max_half

        raw_line = []
        for sample_idx, s in enumerate(np.linspace(-1.0, 1.0, n_samples)):
            if s < 0.0:
                target = center + (-s) * left_half
            else:
                target = center + s * right_half

            point, projected, method = _front_envelope_projection(vertices, patch_idx, x, y, z, target, region, args)

            raw_line.append({
                "name": f"columella_tip_to_alar_line_{line_idx:02d}_p{sample_idx:02d}",
                "point": np.asarray(point, dtype=np.float64),
                "projected": bool(projected),
                "method": method,
            })

        line_points = _dedupe_line_points(raw_line, min_spacing)
        for local_idx, item in enumerate(line_points):
            point_id += 1
            # Keep the exported name local to the line.  The CSV id column still
            # gives a unique row number, while the name remains easy to audit.
            item["name"] = f"columella_tip_to_alar_line_{line_idx:02d}_p{local_idx:02d}"
            all_points.append(item)

    return all_points


# ============================================================
# Confidence, export, overlay
# ============================================================

def landmark_record(name, point, confidence, method, region, review_threshold):
    point = np.asarray(point, dtype=np.float64)
    return {
        "name": name,
        "point": point,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "method": method,
        "needs_manual_review": bool(confidence < review_threshold),
        "screen_x": float(point @ region["right"]),
        "screen_y": float(point @ region["up"]),
        "depth_front": float(point @ region["front"]),
    }


def extract_anatomical_nose_landmarks(surface_ctx, region, args):
    vertices = surface_ctx["vertices"]
    x, y, z = project_points(vertices, region["right"], region["up"], region["front"])
    curv = compute_vertex_curvature(vertices, surface_ctx["adjacency"])
    patch_idx = make_nose_patch(surface_ctx, region, args)
    if len(patch_idx) < 80:
        raise RuntimeError("Nose patch is too small. Check --front/--up axes or ROI parameters.")

    nasion, nas_conf, nas_method = find_nasion_radix(vertices, patch_idx, x, y, z, region, args)
    pronasale, prn_conf, prn_method = find_pronasale(vertices, patch_idx, x, y, z, region)
    subnasale, columella, sn_conf, col_conf = find_subnasale_and_columella(vertices, patch_idx, x, y, z, region, args)
    la, ra, lsa, rsa, lac, rac, lsc, rsc = find_alare_subalare(vertices, patch_idx, x, y, z, region, args)
    crease_curves, crease_points = find_alar_crease_curves(vertices, patch_idx, x, y, z, surface_ctx["vertex_normals"], curv, region, args)
    nostrils, nostril_meta = find_nostril_rims(surface_ctx, patch_idx, x, y, z, curv, region, args)

    records = []
    rt = args.review_threshold
    records.append(landmark_record("nasion_radix", nasion, nas_conf, nas_method, region, rt))
    records.append(landmark_record("pronasale", pronasale, prn_conf, prn_method, region, rt))
    records.append(landmark_record("subnasale", subnasale, sn_conf, "lower_midline_posterior_concavity", region, rt))
    records.append(landmark_record("columella", columella, col_conf, "lower_midline_anterior_columella", region, rt))
    records.append(landmark_record("left_alare", la, lac, "most_lateral_left_alar_band", region, rt))
    records.append(landmark_record("right_alare", ra, rac, "most_lateral_right_alar_band", region, rt))
    records.append(landmark_record("left_subalare", lsa, lsc, "inferolateral_left_alar_band", region, rt))
    records.append(landmark_record("right_subalare", rsa, rsc, "inferolateral_right_alar_band", region, rt))
    records.append(landmark_record("left_alar_crease", crease_points["left_alar_crease"], 0.70, "curvature_depth_crease_curve_midpoint", region, rt))
    records.append(landmark_record("right_alar_crease", crease_points["right_alar_crease"], 0.70, "curvature_depth_crease_curve_midpoint", region, rt))

    for side in ("left", "right"):
        conf = nostril_meta[side]["confidence"]
        method = nostril_meta[side]["method"]
        for label in ("medial", "lateral", "superior", "inferior"):
            records.append(landmark_record(f"{side}_nostril_{label}", nostrils[side][label], conf, method, region, rt))

    anatomical = {rec["name"]: rec for rec in records}
    dense = build_dense_anatomical_template(vertices, patch_idx, x, y, z, region, anatomical, crease_curves, nostrils, args)
    return anatomical, dense, patch_idx, curv


def euclidean(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def extract_measurements(anatomical, outdir):
    p = {k: v["point"] for k, v in anatomical.items()}
    measurements = {}
    if all(k in p for k in ("nasion_radix", "subnasale")):
        measurements["Nasal_Height_nasion_to_subnasale"] = euclidean(p["nasion_radix"], p["subnasale"])
    if all(k in p for k in ("left_alare", "right_alare")):
        measurements["Alar_Base_Width"] = euclidean(p["left_alare"], p["right_alare"])
    if all(k in p for k in ("pronasale", "subnasale")):
        measurements["Tip_to_Subnasale_Distance"] = euclidean(p["pronasale"], p["subnasale"])
    if "Nasal_Height_nasion_to_subnasale" in measurements and "Alar_Base_Width" in measurements:
        measurements["Nasal_Index_Width_over_Height"] = measurements["Alar_Base_Width"] / max(measurements["Nasal_Height_nasion_to_subnasale"], 1e-9)
    measurements["manual_review_required"] = any(v["needs_manual_review"] for v in anatomical.values())
    with open(os.path.join(outdir, "measurements_review_required.json"), "w") as f:
        json.dump(measurements, f, indent=4)
    return measurements


def save_point_cloud(path, points, color=(0.0, 1.0, 0.0)):
    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    if len(points):
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(points), 1)))
    o3d.io.write_point_cloud(path, pcd)


def export_outputs(mesh, region, anatomical, dense, patch_idx, pose_transform, outdir, args):
    os.makedirs(outdir, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    # Anatomical named landmarks
    with open(os.path.join(outdir, "anatomical_nose_landmarks_3d.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "method", "confidence", "needs_manual_review", "x", "y", "z", "screen_x", "screen_y", "depth_front"])
        for i, rec in enumerate(anatomical.values(), start=1):
            p = rec["point"]
            writer.writerow([i, rec["name"], rec["method"], rec["confidence"], rec["needs_manual_review"], p[0], p[1], p[2], rec["screen_x"], rec["screen_y"], rec["depth_front"]])

    # JSON with named landmarks and audit fields
    json_ready = {}
    for name, rec in anatomical.items():
        json_ready[name] = {
            "point": [float(v) for v in rec["point"]],
            "confidence": rec["confidence"],
            "method": rec["method"],
            "needs_manual_review": rec["needs_manual_review"],
        }
    with open(os.path.join(outdir, "anatomical_nose_landmarks_review.json"), "w") as f:
        json.dump(json_ready, f, indent=4)

    # Dense curves and grid CSV
    with open(os.path.join(outdir, "dense_nose_template_points.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "id", "name", "x", "y", "z"])
        for group, data in dense.items():
            # Groups like surface_grid and columella_tip_to_alar_lines are
            # stored as dictionaries with explicit point names.
            if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
                for i, item in enumerate(data):
                    p = np.asarray(item["point"], dtype=np.float64)
                    writer.writerow([group, i + 1, item.get("name", f"{group}_{i+1:02d}"), p[0], p[1], p[2]])
            else:
                arr = np.asarray(data, dtype=np.float64)
                for i, p in enumerate(arr):
                    writer.writerow([group, i + 1, f"{group}_{i+1:02d}", p[0], p[1], p[2]])

    # PLY exports
    save_point_cloud(os.path.join(outdir, "anatomical_nose_landmarks_cloud.ply"), [r["point"] for r in anatomical.values()], (0.0, 1.0, 0.0))
    dense_points = []
    for group, data in dense.items():
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
            dense_points.extend([item["point"] for item in data])
        else:
            dense_points.extend(list(np.asarray(data, dtype=np.float64)))
    save_point_cloud(os.path.join(outdir, "dense_nose_template_cloud.ply"), dense_points, (0.0, 0.6, 1.0))
    save_point_cloud(os.path.join(outdir, "nose_surface_patch_cloud.ply"), vertices[patch_idx], (1.0, 0.7, 0.0))

    with open(os.path.join(outdir, "pose_transform_matrix.csv"), "w", newline="") as f:
        csv.writer(f).writerows(pose_transform)

    measurements = extract_measurements(anatomical, outdir)
    make_overlay(vertices, anatomical, dense, patch_idx, region, os.path.join(outdir, "nose_landmarks_medical_review_overlay.png"), args)
    return measurements


def make_overlay(vertices, anatomical, dense, patch_idx, region, out_png, args, image_size=1400):
    x, y, z = project_points(vertices, region["right"], region["up"], region["front"])
    x1, x2 = np.percentile(x, 1), np.percentile(x, 99)
    y1, y2 = np.percentile(y, 1), np.percentile(y, 99)
    pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
    x1, x2, y1, y2 = x1 - pad_x, x2 + pad_x, y1 - pad_y, y2 + pad_y
    img = Image.new("RGB", (image_size, image_size), (18, 18, 18))
    draw = ImageDraw.Draw(img)

    def to_img(xx, yy):
        return int(((xx - x1) / max(x2 - x1, 1e-9)) * (image_size - 1)), int((1.0 - (yy - y1) / max(y2 - y1, 1e-9)) * (image_size - 1))

    sample = np.linspace(0, len(vertices) - 1, min(len(vertices), args.overlay_max_points)).astype(int)
    z_min, z_max = np.percentile(z[sample], 2), np.percentile(z[sample], 98)
    for i in sample:
        px, py = to_img(x[i], y[i])
        shade = int(55 + 170 * np.clip((z[i] - z_min) / max(z_max - z_min, 1e-9), 0, 1))
        draw.point((px, py), fill=(shade, shade, shade))

    # nose patch in orange tint
    for i in patch_idx[::max(1, len(patch_idx)//15000)]:
        px, py = to_img(x[i], y[i])
        draw.point((px, py), fill=(255, 170, 20))

    # Dense surface grid in cyan-ish
    for item in dense.get("surface_grid", []):
        p = item["point"]
        px, py = to_img(float(p @ region["right"]), float(p @ region["up"]))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(0, 210, 255))

    # Columella guide lines in magenta/yellow so they are easy to audit.
    for item in dense.get("columella_tip_to_alar_lines", []):
        p = item["point"]
        px, py = to_img(float(p @ region["right"]), float(p @ region["up"]))
        color = (255, 230, 0) if item.get("projected", False) else (255, 80, 220)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(0, 0, 0), width=1)

    # Named anatomical landmarks
    for i, rec in enumerate(anatomical.values(), start=1):
        p = rec["point"]
        px, py = to_img(float(p @ region["right"]), float(p @ region["up"]))
        fill = (255, 60, 60) if rec["needs_manual_review"] else (0, 255, 80)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=fill, outline=(0, 0, 0), width=1)
        draw.text((px + 8, py - 8), str(i), fill=(255, 255, 255))

    img.save(out_png)

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Mesh-first anatomical nose landmark extraction")
    parser.add_argument("--obj", required=True, help="Input mesh path: OBJ/PLY/STL supported by Open3D")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--front", default="+z", help="Anterior/forward mesh axis, e.g. +z or -z")
    parser.add_argument("--up", default="-y", help="Superior/up mesh axis, e.g. -y or +y")
    parser.add_argument("--subdivision", type=int, default=0)

    parser.add_argument("--sor_neighbors", type=int, default=35)
    parser.add_argument("--sor_std", type=float, default=1.8)

    # For medical use default is OFF.
    parser.add_argument("--enable_pose_normalize", action="store_true", help="Optional symmetry ICP. Off by default to preserve true asymmetry.")
    parser.add_argument("--icp_voxel_ratio", type=float, default=0.012)
    parser.add_argument("--icp_threshold_factor", type=float, default=4.0)
    parser.add_argument("--icp_iterations", type=int, default=80)
    parser.add_argument("--icp_half_factor", type=float, default=0.50)

    # Rough ROI detection parameters
    parser.add_argument("--central_band", type=float, default=0.16)
    parser.add_argument("--vertical_low", type=float, default=0.18)
    parser.add_argument("--vertical_high", type=float, default=0.82)
    parser.add_argument("--tip_candidates", type=int, default=700)
    parser.add_argument("--tip_validation_radius", type=float, default=0.030)
    parser.add_argument("--local_width_y_band", type=float, default=0.045)
    parser.add_argument("--local_width_min", type=float, default=0.14)
    parser.add_argument("--local_width_max", type=float, default=0.32)
    parser.add_argument("--nose_height_ratio", type=float, default=0.56)
    parser.add_argument("--nose_width_ratio", type=float, default=0.34)
    parser.add_argument("--top_ratio", type=float, default=0.60)
    parser.add_argument("--bottom_ratio", type=float, default=0.38)
    parser.add_argument("--bottom_extension", type=float, default=0.00)
    parser.add_argument("--max_nose_height_factor", type=float, default=0.74)

    # Anatomical width model
    parser.add_argument("--bridge_width_factor", type=float, default=0.34)
    parser.add_argument("--mid_width_factor", type=float, default=0.65)
    parser.add_argument("--bottom_width_factor", type=float, default=1.05)

    # Patch and dense template
    parser.add_argument("--patch_width_factor", type=float, default=0.90)
    parser.add_argument("--patch_left_factor", type=float, default=1.30)
    parser.add_argument("--patch_right_factor", type=float, default=1.30)
    parser.add_argument("--patch_y_margin", type=float, default=0.04)
    parser.add_argument("--patch_bottom_margin", type=float, default=0.015)
    parser.add_argument("--patch_bottom_raise", type=float, default=0.06)
    parser.add_argument("--patch_envelope_bins", type=int, default=28)
    parser.add_argument("--patch_front_percentile", type=float, default=97.0)
    parser.add_argument("--patch_front_slack_factor", type=float, default=0.55)
    parser.add_argument("--patch_min_vertices", type=int, default=80)
    # Deprecated, retained for CLI compatibility with older commands.
    parser.add_argument("--patch_depth_percentile", type=float, default=8.0)
    parser.add_argument("--patch_depth_slack", type=float, default=0.25)
    parser.add_argument("--dense_rows", type=int, default=15)
    parser.add_argument("--dense_cols", type=int, default=7)
    parser.add_argument("--dense_grid_width_factor", type=float, default=0.60)
    parser.add_argument("--dense_left_factor", type=float, default=1.20)
    parser.add_argument("--dense_right_factor", type=float, default=1.20)
    parser.add_argument("--dense_y_band", type=float, default=0.030)
    parser.add_argument("--dense_x_band", type=float, default=0.070)
    parser.add_argument("--dense_front_percentile", type=float, default=95.0)
    parser.add_argument("--dense_front_slack_factor", type=float, default=0.42)
    parser.add_argument("--crease_curve_points", type=int, default=9)

    # Controlled columella guide lines. These are dense/auditable guide points,
    # not primary clinical landmarks. They are designed to preserve the stable
    # lower-nose landmarks while adding 2-6 parallel transverse lines from the
    # medial alar-rim level toward the infratip/lower nose tip.
    parser.add_argument("--enable_columella_guides", action="store_true", default=True)
    parser.add_argument("--disable_columella_guides", action="store_false", dest="enable_columella_guides")
    parser.add_argument("--columella_transverse_lines", type=int, default=5)
    parser.add_argument("--columella_transverse_samples", type=int, default=19)
    parser.add_argument("--columella_transverse_top_drop", type=float, default=0.025)
    parser.add_argument("--columella_transverse_max_half", type=float, default=0.22)
    parser.add_argument("--columella_tip_blend", type=float, default=0.72)
    parser.add_argument("--columella_taper_power", type=float, default=0.85)
    parser.add_argument("--columella_lip_guard_factor", type=float, default=0.035)
    parser.add_argument("--columella_left_expand", type=float, default=1.90)
    parser.add_argument("--columella_right_expand", type=float, default=1.05)
    parser.add_argument("--columella_min_width_factor", type=float, default=0.22)
    parser.add_argument("--columella_x_band", type=float, default=0.045)
    parser.add_argument("--columella_y_band", type=float, default=0.045)
    parser.add_argument("--columella_front_percentile", type=float, default=82.0)
    parser.add_argument("--columella_front_slack", type=float, default=0.12)
    parser.add_argument("--columella_min_local_candidates", type=int, default=3,
                        help="Minimum local mesh vertices required before the columella guide search window expands.")
    parser.add_argument("--columella_projection_sigma", type=float, default=0.75,
                        help="Gaussian width for weighted front-envelope columella projection.")
    parser.add_argument("--columella_projection_blend", type=float, default=0.35,
                        help="0 keeps designed guide points; 1 fully projects to mesh front-envelope estimate.")
    parser.add_argument("--columella_duplicate_spacing_factor", type=float, default=0.012,
                        help="Minimum spacing between exported columella guide points as a fraction of nose width.")

    # Deprecated compatibility arguments.
    # v29 intentionally does NOT generate vertical columella guides. These
    # arguments are retained so older shell commands do not fail, but they are ignored.
    parser.add_argument("--enable_columella_vertical_guides", action="store_true", default=False)
    parser.add_argument("--disable_columella_vertical_guides", action="store_false", dest="enable_columella_vertical_guides")
    parser.add_argument("--columella_vertical_cols", type=int, default=0)
    parser.add_argument("--columella_vertical_rows", type=int, default=0)
    parser.add_argument("--columella_vertical_tip_blend", type=float, default=0.0)
    parser.add_argument("--columella_vertical_top_drop", type=float, default=0.0)
    parser.add_argument("--columella_vertical_lip_guard_factor", type=float, default=0.0)
    parser.add_argument("--columella_vertical_left_expand", type=float, default=0.0)
    parser.add_argument("--columella_vertical_right_expand", type=float, default=0.0)
    parser.add_argument("--columella_vertical_max_half", type=float, default=0.0)
    parser.add_argument("--columella_vertical_top_width_scale", type=float, default=0.0)

    # Review / export
    parser.add_argument("--review_threshold", type=float, default=0.45)
    parser.add_argument("--overlay_max_points", type=int, default=220000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mesh, vertices = load_mesh(args.obj, subdivision=args.subdivision)
    clean_points = clean_detection_points(vertices, nb_neighbors=args.sor_neighbors, std_ratio=args.sor_std)

    pose_transform = np.eye(4)
    if args.enable_pose_normalize:
        print("Optional symmetry ICP pose normalization enabled. For medical asymmetry studies, verify this choice.")
        mesh, clean_points, pose_transform = normalize_pose_by_icp_symmetry(mesh, clean_points, args)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)

    region = estimate_nose_region(clean_points, args)
    surface_ctx = create_surface_context(mesh)
    anatomical, dense, patch_idx, _ = extract_anatomical_nose_landmarks(surface_ctx, region, args)
    measurements = export_outputs(mesh, region, anatomical, dense, patch_idx, pose_transform, args.outdir, args)
    # v29: vertical columella guides are intentionally disabled.
    transverse_count = len(dense.get("columella_tip_to_alar_lines", [])) if isinstance(dense.get("columella_tip_to_alar_lines", []), list) else 0

    print("\nSuccess. Mesh-first anatomical nose extraction complete.")
    print(f"Output directory: {args.outdir}")
    print("Key files:")
    print("  anatomical_nose_landmarks_3d.csv")
    print("  anatomical_nose_landmarks_review.json")
    print("  dense_nose_template_points.csv  # includes columella_tip_to_alar_lines when enabled")
    print("  anatomical_nose_landmarks_cloud.ply")
    print("  dense_nose_template_cloud.ply")
    print("  nose_surface_patch_cloud.ply")
    print("  nose_landmarks_medical_review_overlay.png")
    print("  measurements_review_required.json")
    print(f"  columella_tip_to_alar_lines exported: {transverse_count}")

    if measurements.get("manual_review_required", True):
        print("\nManual review required: at least one landmark has low confidence or fallback extraction.")
    else:
        print("\nManual review still recommended before medical measurement use.")


if __name__ == "__main__":
    main()
