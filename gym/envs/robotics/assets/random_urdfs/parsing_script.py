import trimesh
import os
import numpy as np

def generate_env(obj_index, out_path):
    # step 0: read file
    obj_index = str(obj_index).zfill(3)
    mesh = trimesh.load(os.path.join(obj_index, obj_index+'_coll.obj'))

    # step 1: create convex mesh
    vertices_lst = []
    for m in mesh:
        vertices_lst.append(m.vertices)
    vertices_lst = np.vstack(vertices_lst)
    convex_mesh = trimesh.Trimesh(vertices=vertices_lst).convex_hull

    # step 2: write convex hull as stl file
    convex_mesh.export(os.path.join(obj_index, obj_index+'_coll.stl'))

    # step 3: record center of mass and box size
    convex_com = convex_mesh.center_mass
    half_length = convex_mesh.bounding_box_oriented.primitive.extents
