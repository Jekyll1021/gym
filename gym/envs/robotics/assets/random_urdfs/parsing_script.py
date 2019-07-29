import trimesh
import os

import numpy as np
import xml.etree.ElementTree as ET

def generate_grasp_env(model_path, obj_index, out_path):
    # step 0: read file
    obj_index = str(obj_index).zfill(3)
    mesh = trimesh.load(os.path.join(model_path, os.path.join(obj_index, obj_index+'.obj')))

    # step 2: write as stl file
    new_model_path = os.path.join(model_path, os.path.join(obj_index, obj_index+'.stl'))
    mesh.export(new_model_path)

    # step 3: record center of mass and box size
    convex_com = mesh.center_mass
    half_length = mesh.bounding_box.primitive.extents * 0.5

    scale = 0.035/np.median(half_length)
    convex_com *= scale
    half_length *= scale

    # step 4: read template, change template and write to xml
    tree = ET.parse(os.path.join(out_path, "grasp_template.xml"))
    root = tree.getroot()
    root[3][0].attrib["file"] = os.path.join("..", new_model_path)
    root[3][0].attrib["scale"] = str(scale) + ' ' + str(scale) + ' ' + str(scale)
    # root[3][0].attrib -- {'file': path, 'name': 'obj0', 'scale': scale}
    root[4][4].attrib["pos"] = str(half_length[0]) + ' ' + str(half_length[1]) + ' ' + str(half_length[2])
    root[4][4][2].attrib["pos"] = str(convex_com[0]) + ' ' + str(convex_com[1]) + ' ' + str(convex_com[2])
    root[4][4][2].attrib["size"] = str(half_length[0]/2) + ' ' + str(half_length[1]/2) + ' ' + str(half_length[2]/2)
    # root[4][4][2].attrib["pos"] = str(convex_com[0]) + ' ' + str(convex_com[1]) + ' ' + str(convex_com[2])
    # root[4][4][2].attrib -- {'type': 'box', 'size': bbox size, 'pos': centroid, 'rgba': '1 0 0 0', 'condim': '3', 'material': 'block_mat', 'mass': '2'}

    tree.write(os.path.join(out_path, obj_index+"_grasp.xml"))

def generate_peg_env(model_path, obj_index, out_path):
    # step 0: read file
    obj_index = str(obj_index).zfill(3)
    mesh = trimesh.load(os.path.join(model_path, os.path.join(obj_index, obj_index+'.obj')))

    # step 2: write as stl file
    new_model_path = os.path.join(model_path, os.path.join(obj_index, obj_index+'.stl'))
    mesh.export(new_model_path)

    # step 3: record center of mass and box size
    convex_com = mesh.center_mass
    half_length = mesh.bounding_box.primitive.extents * 0.5

    scale = 0.035/np.median(half_length)
    convex_com *= scale
    half_length *= scale
    zaxis = np.zeros(3)
    zaxis[np.argmin(half_length)] = 1

    # step 4: read template, change template and write to xml
    tree = ET.parse(os.path.join(out_path, "peg_insert_template.xml"))
    root = tree.getroot()
    root[3][0].attrib["file"] = os.path.join("..", new_model_path)
    root[3][0].attrib["scale"] = str(scale) + ' ' + str(scale) + ' ' + str(scale)
    # root[3][0].attrib -- {'file': path, 'name': 'obj0', 'scale': scale}
    root[4][5].attrib["pos"] = str(half_length[0]) + ' ' + str(half_length[1]) + ' ' + str(half_length[2])
    root[4][5].attrib["zaxis"] = str(zaxis[0]) + ' ' + str(zaxis[1]) + ' ' + str(zaxis[2])

    root[4][5][1].attrib["zaxis"] = str(zaxis[0]) + ' ' + str(zaxis[1]) + ' ' + str(zaxis[2])

    max_offset = np.median(half_length)
    handle_size_x, handle_size_y, handle_size_z = np.random.uniform(0.01, 0.03), np.random.uniform(0.01, 0.03), np.random.uniform(0.03, 0.05)
    offset_x, offset_y = np.random.uniform(-max_offset + handle_size_x, max_offset - handle_size_x), np.random.uniform(-max_offset + handle_size_y, max_offset - handle_size_y)
    root[4][5][2].attrib["zaxis"] = str(zaxis[0]) + ' ' + str(zaxis[1]) + ' ' + str(zaxis[2])
    root[4][5][2].attrib["size"] = str(handle_size_x) + ' ' + str(handle_size_y) + ' ' + str(handle_size_z)
    root[4][5][2].attrib["pos"] = str(convex_com[0] + offset_x) + ' ' + str(convex_com[1] + offset_y) + ' ' + str(convex_com[2] + handle_size_z - np.min(half_length))
    # root[4][4][2].attrib -- {'type': 'box', 'size': bbox size, 'pos': centroid, 'rgba': '1 0 0 0', 'condim': '3', 'material': 'block_mat', 'mass': '2'}
    root[4][5][3].attrib["pos"] = str(convex_com[0]) + ' ' + str(convex_com[1]) + ' ' + str(convex_com[2])
    root[4][5][3].attrib["size"] = str(half_length[0]/2) + ' ' + str(half_length[1]/2) + ' ' + str(half_length[2]/2)
    root[4][5][4].attrib["pos"] = str(convex_com[0] + offset_x) + ' ' + str(convex_com[1] + offset_y) + ' ' + str(convex_com[2] + handle_size_z - 0.01 - np.min(half_length))

    tree.write(os.path.join(out_path, obj_index+"_peg.xml"))

def generate_slide_env(model_path, obj_index, out_path):
    # step 0: read file
    obj_index = str(obj_index).zfill(3)
    mesh = trimesh.load(os.path.join(model_path, os.path.join(obj_index, obj_index+'.obj')))

    # step 2: write as stl file
    new_model_path = os.path.join(model_path, os.path.join(obj_index, obj_index+'.stl'))
    mesh.export(new_model_path)

    # step 3: record center of mass and box size
    convex_com = mesh.center_mass
    half_length = mesh.bounding_box.primitive.extents * 0.5

    scale = 0.035/np.median(half_length)
    convex_com *= scale
    half_length *= scale

    # step 4: read template, change template and write to xml
    tree = ET.parse(os.path.join(out_path, "slide_template.xml"))
    root = tree.getroot()
    root[3][0].attrib["file"] = os.path.join("..", new_model_path)
    root[3][0].attrib["scale"] = str(scale) + ' ' + str(scale) + ' ' + str(scale)
    # root[3][0].attrib -- {'file': path, 'name': 'obj0', 'scale': scale}
    root[4][3][2].attrib["pos"] = str(half_length[0]) + ' ' + str(half_length[1]) + ' ' + str(half_length[2] + 0.2)
    root[4][3][2][2].attrib["size"] = str(half_length[0]) + ' ' + str(half_length[1]) + ' ' + str(half_length[2])
    root[4][3][2][2].attrib["pos"] = str(convex_com[0]) + ' ' + str(convex_com[1]) + ' ' + str(convex_com[2])
    # root[4][4][2].attrib -- {'type': 'box', 'size': bbox size, 'pos': centroid, 'rgba': '1 0 0 0', 'condim': '3', 'material': 'block_mat', 'mass': '2'}

    tree.write(os.path.join(out_path, obj_index+"_slide.xml"))

# if __name__ == "__main__":
#     # loop
#     for i in range(1000):
#         generate_grasp_env("../stls/fetch/random_urdfs", i, "../fetch/random_obj_xml")
#         generate_peg_env("../stls/fetch/random_urdfs", i, "../fetch/random_obj_xml")
#         generate_slide_env("../stls/fetch/random_urdfs", i, "../fetch/random_obj_xml")
