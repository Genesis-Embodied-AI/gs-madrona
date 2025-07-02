import xml.etree.ElementTree as ET
import argparse
import os
import copy
import uuid


def wrap_visual_geoms_with_bodies(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    def wrap_geom(parent):
        removed_elems = []
        append_elems = []
        for i, elem in enumerate(parent):
            if elem.tag == "geom":
                if elem.attrib.get("class") == "visual":
                    # Create a new <body> element
                    body = ET.Element("body")
                    # Give it a name based on mesh if available
                    mesh_name = elem.attrib.get("mesh", f"{i}")
                    body.set("name", f"body_{mesh_name}_{str(uuid.uuid4())[:4]}")
                    # Deep copy of the geom to keep attributes
                    body.append(copy.deepcopy(elem))
                    # Replace <geom> with <body> in parent
                    removed_elems.append(elem)
                    append_elems.append(body)
                elif elem.attrib.get("class") == "collision":
                    removed_elems.append(elem)
            else:
                wrap_geom(elem)  # Recurse into children

        for elem in removed_elems:
            parent.remove(elem)
        for elem in append_elems:
            parent.append(elem)

    wrap_geom(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


# Example usage
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, default="./panda.xml")
args = parser.parse_args()

input_file = args.file
output_file = f"{os.path.splitext(input_file)[0]}_new.xml"
wrap_visual_geoms_with_bodies(input_file, output_file)
