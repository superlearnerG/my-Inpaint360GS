import os
import json
import shutil
import argparse
import re

def setup_configs(dataset_name, scene, target_id, surrounding_ids):
    """
    Sets up the configuration files. 
    Format: Each key-value on a new line, but lists stay on a single line.
    """
    
    if target_id is None:
        print("[!] Error: target_id is required.")
        return

    safe_surrounding = surrounding_ids if surrounding_ids is not None else []
    final_obj_ids = list(dict.fromkeys(target_id))

    if surrounding_ids:
        final_obj_ids.extend(surrounding_ids)
    
    dataset_conf_dirs = [
        f"config/object_inpaint/{dataset_name}",
        f"config/object_removal/{dataset_name}"
    ]
    
    for folder in dataset_conf_dirs:
        os.makedirs(folder, exist_ok=True)
        source_file = os.path.join(os.path.dirname(folder), "common.json")
        target_file = os.path.join(folder, f"{scene}.json")
        
        if not os.path.exists(target_file):
            if os.path.exists(source_file):
                shutil.copy(source_file, target_file)
                print(f"[+] Initialized {target_file} from template.")
            else:
                print(f"[!] Warning: {source_file} not found. Skipping {folder}.")
                continue

        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["target_id"] = target_id
            data["surrounding_ids"] = safe_surrounding
            data["select_obj_id"] = final_obj_ids
            
            full_json = json.dumps(data, indent=4, ensure_ascii=False)
            
            compact_json = re.sub(
                r'\[\s+([\d,\s]+)\s+\]', 
                lambda m: "[" + ", ".join(m.group(1).split()).replace(",,", ",") + "]", 
                full_json
            )
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(compact_json)
            
            print(f"[*] Successfully updated {target_file}: select_obj_id -> {final_obj_ids}")
            
        except Exception as e:
            print(f"[!] Error processing {target_file}: {e}")

def list_of_ints(arg):
    if not arg or str(arg).lower() == "none": 
        return []
    return [int(x.strip()) for x in str(arg).replace(',', ' ').split()]

def main():
    parser = argparse.ArgumentParser(description="Configure scene JSONs with clean multi-line formatting.")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--target_id", type=list_of_ints, required=True, help="Primary object ID")
    parser.add_argument("--target_surronding_id", type=list_of_ints, default=None, 
                        help="Surrounding objects IDs (e.g., 1,2,3)")
    
    args = parser.parse_args()
    setup_configs(args.dataset_name, args.scene, args.target_id, args.target_surronding_id)

if __name__ == "__main__":
    main()