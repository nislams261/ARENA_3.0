# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"
# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels, 2, 3))
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0,10,1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%
def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """

    ray = ray[:, :2]
    segment = segment[:, :2]
    origin, direction = ray
    l1, l2 = segment
    mat = t.stack((direction, l1 - l2), dim = 1)
    vec = l1 - origin
    try:
        solution = t.linalg.solve(mat, vec)
    except RuntimeError:
        return False
    u, v = solution.float()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    num_rays = rays.size(0)
    num_segments = segments.size(0)

    rays = rays[..., :2]
    segments = segments[..., :2]

    rays = einops.repeat(rays, 'n o d -> n ns o d', ns=num_segments)
    segments = einops.repeat(segments, 'n l1 l2 -> nr n l1 l2', nr=num_rays)
    origins = rays[:, :, 0]
    directions = rays[:, :, 1] - origins
    l1s = segments[:,:, 0]
    l2s = segments[:,:, 1]

    mats = t.stack((directions, l1s - l2s), dim=-1)
    dets = t.linalg.det(mats)
    is_singular = dets.abs() < 1e-8
    mats[is_singular] = t.eye(2)

    vecs = l1s - origins
    solutions = t.linalg.solve(mats, vecs)
    u = solutions[..., 0]
    v = solutions[..., 1]

    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """

    n_pixels = num_pixels_z * num_pixels_y
    rays = t.zeros((n_pixels, 2, 3))
    y_vals = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_vals = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays[:, 1, 0] = 1
    y , z = t.meshgrid(y_vals, z_vals, indexing='ij')
    
    rays[:, 1, 1] = y.flatten()
    rays[:, 1, 2] = z.flatten()

    return rays


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    B_minus_A = B - A
    C_minus_A = C - A
    mat = t.stack((-D, B_minus_A, C_minus_A), dim = 1)
    vec = O - A
    solutions = t.linalg.solve(mat, vec)
    s, u, v = solutions.float()
    return ((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)).item()



tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    origins = rays[:, 0, :]
    directions = rays[:, 1, :]

    A, B, C = triangle

    nrays = origins.size(0)

    As = einops.repeat(A, 'd -> nr d', nr = nrays)
    Bs = einops.repeat(B, 'd -> nr d', nr = nrays)
    Cs = einops.repeat(C, 'd -> nr d', nr = nrays)

    Bs_minus_As = Bs - As
    Cs_minus_As = Cs - As

    mats = t.stack((-directions, Bs_minus_As, Cs_minus_As), dim=-1)
    vecs = origins - As
    solutions = t.linalg.solve(mats, vecs)
    ss, us, vs = solutions.unbind(dim=1)
    return ((ss >= 0) & (us >= 0) & (vs >= 0) & (us + vs <= 1))

A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(dim=1)

    mat = t.stack([- D, B - A, C - A], dim = -1)
    
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    num_rays = rays.size(0)
    num_triangles = triangles.size(0)

    rays = einops.repeat(rays, 'n p d -> n nt p d', nt=num_triangles)
    triangles = einops.repeat(triangles, 'n p d -> nr n p d', nr=num_rays)

    origins, directions = rays.unbind(dim=-2)
    As, Bs, Cs = triangles.unbind(dim=-2)

    Bs_minus_As = Bs - As
    Cs_minus_As = Cs - As

    mats = t.stack((-directions, Bs_minus_As, Cs_minus_As), dim=-1)
    vecs = origins - As

    dets = t.linalg.det(mats)
    is_singular = dets.abs() <1e-8
    mats[is_singular] = t.eye(3)

    solutions = t.linalg.solve(mats,vecs)
    ss, us, vs = solutions.unbind(dim=-1)

    ss *= directions[..., 0]

    intersects = (us >= 0) & (vs >= 0) & (us + vs <= 1) & ~is_singular
    ss[~intersects] = float("inf")

    return einops.reduce(ss, "NR NT -> NR", "min")

num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %%
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return t.tensor([[t.cos(theta), 0, t.sin(theta)],
                     [0,1,0],
                     [-t.sin(theta),0,t.cos(theta)]])


tests.test_rotation_matrix(rotation_matrix)
# %%
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(
        coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video"
    ).show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)
# %%
def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    num_rays = rays.size(0)
    num_triangles = triangles.size(0)

    rays = einops.repeat(rays, 'n p d -> n nt p d', nt=num_triangles)
    triangles = einops.repeat(triangles, 'n p d -> nr n p d', nr=num_rays)

    rays = rays.to("cuda")
    triangles = triangles.to("cuda")

    origins, directions = rays.unbind(dim=-2)
    As, Bs, Cs = triangles.unbind(dim=-2)

    Bs_minus_As = Bs - As
    Cs_minus_As = Cs - As

    mats = t.stack((-directions, Bs_minus_As, Cs_minus_As), dim=-1)
    vecs = origins - As

    dets = t.linalg.det(mats)
    is_singular = dets.abs() <1e-8
    mats[is_singular] = t.eye(3, device=mats.device)

    solutions = t.linalg.solve(mats,vecs)
    ss, us, vs = solutions.unbind(dim=-1)

    ss *= directions[..., 0]

    intersects = (us >= 0) & (vs >= 0) & (us + vs <= 1) & ~is_singular
    ss[~intersects] = float("inf")

    return (einops.reduce(ss, "NR NT -> NR", "min")).to("cpu")


dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
# %%
