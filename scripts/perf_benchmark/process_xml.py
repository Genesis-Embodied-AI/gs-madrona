import xml.etree.ElementTree as ET
import argparse
import os
import copy
import yaml

# Get asset directory from environment variable
asset_dir = os.getenv("ASSET_DIR")


def process_mjcf_geoms(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    link_geoms = {}

    def wrap_geom(cur, cur_link_name=None):
        replace_elems = []
        remove_elems = []
        for i, elem in enumerate(cur):
            if elem.tag == "geom":
                if cur_link_name is None:
                    continue
                
                if elem.get("class", None) == "collision" or \
                   elem.get("conaffinity", 0) > 0 or \
                   elem.get("contype", 0) > 0:
                    remove_elems.append(elem)
                    continue

                body = ET.Element("body")
                body_idx = link_geoms.get(cur_link_name, 0)
                link_geoms[cur_link_name] = body_idx + 1
                body.set("name", f"{cur_link_name}_geom{body_idx}")
                body.append(copy.deepcopy(elem))
                replace_elems.append((elem, body))
            else:
                nex_link_name = None
                if elem.tag == "body":
                    nex_link_name = elem.get("name", None)
                # Recurse into children
                wrap_geom(elem, nex_link_name if nex_link_name else cur_link_name)

        for elem, body in replace_elems:
            idx = list(cur).index(elem)
            cur.remove(elem)
            cur.insert(idx, body)
        for elem in remove_elems:
            cur.remove(elem)

    wrap_geom(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def process_config_file(config_file):
    """Process all robot files mentioned in a specific benchmark configuration file."""
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        return
    
    print(f"Processing configuration file: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'mjcf_list' not in config:
            print(f"No mjcf_list found in {config_file}")
            return
        
        mjcf_list = config['mjcf_list']
        print(f"Found {len(mjcf_list)} robot files in {config_file}")
        
        processed_count = 0
        for mjcf_path in mjcf_list:
            # Construct full path using asset directory
            full_mjcf_path = os.path.join(asset_dir, mjcf_path)
            
            if not os.path.exists(full_mjcf_path):
                print(f"Warning: Robot file not found: {full_mjcf_path}")
                continue
            
            print(f"Processing robot file: {full_mjcf_path}")
            output_file = f"{os.path.splitext(full_mjcf_path)[0]}_new.xml"
            process_mjcf_geoms(full_mjcf_path, output_file)
            processed_count += 1
            print(f"Created processed file: {output_file}")
        
        print(f"Total files processed: {processed_count}")
            
    except Exception as e:
        print(f"Error processing {config_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process MJCF robot files from a benchmark configuration file")
    parser.add_argument("--file", type=str, help="Path to the benchmark configuration YAML file")
    args = parser.parse_args()
    
    # Process robot files from the specified configuration file
    process_config_file(args.file)


if __name__ == "__main__":
    main()
