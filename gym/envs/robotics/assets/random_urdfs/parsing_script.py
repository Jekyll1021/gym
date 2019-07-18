import trimesh
import os

import numpy as np
import xml.etree.ElementTree as ET

def generate_env(model_path, obj_index, out_path):
    # step 0: read file
    obj_index = str(obj_index).zfill(3)
    mesh = trimesh.load(os.path.join(model_path, os.path.join(obj_index, obj_index+'.obj')))

    # step 2: write as stl file
    new_model_path = os.path.join(model_path, os.path.join(obj_index, obj_index+'.stl'))
    mesh.export(new_model_path)

    # step 3: record center of mass and box size
    convex_com = mesh.center_mass
    half_length = mesh.bounding_box.primitive.extents * 0.5

    scale = 0.035/np.max(half_length)
    convex_com *= scale
    half_length *= scale

    # step 4: read template, change template and write to xml
    tree = ET.parse(os.path.join(out_path, "grasp_template.xml"))
    root = tree.getroot()
    root[3][0].attrib["file"] = os.path.join("..", new_model_path)
    root[3][0].attrib["scale"] = str(scale) + ' ' + str(scale) + ' ' + str(scale)
    # root[3][0].attrib -- {'file': path, 'name': 'obj0', 'scale': scale}
    root[4][4].attrib["pos"] = str(half_length[0]) + ' ' + str(half_length[1]) + ' ' + str(half_length[2])
    root[4][4][2].attrib["size"] = str(half_length[0]) + ' ' + str(half_length[1]) + ' ' + str(half_length[2])
    root[4][4][2].attrib["pos"] = str(convex_com[0]) + ' ' + str(convex_com[1]) + ' ' + str(convex_com[2])
    # root[4][4][2].attrib -- {'type': 'box', 'size': bbox size, 'pos': centroid, 'rgba': '1 0 0 0', 'condim': '3', 'material': 'block_mat', 'mass': '2'}

    tree.write(os.path.join(out_path, obj_index+".xml"))

# loop
# generate_env("../stls/fetch/random_urdfs", 0, "../fetch/random_obj_xml")
