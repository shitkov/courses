import argparse
import json
import os
import tempfile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key")
    parser.add_argument("--val")
    args = parser.parse_args()
    return args.key, args.val


def file_upd(key, val):
    storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')
    if os.path.exists(storage_path) is False:
        with open(storage_path, 'w') as f:
            f.write(json.dumps({}))
    with open(storage_path, 'r') as f:
        current_dict = json.loads(f.read())
    if val is None:
        if key in list(current_dict.keys()):
            print(', '.join(current_dict[key]))
        else:
            return None
    else:
        with open(storage_path, 'w') as f:
            if key in list(current_dict.keys()):
                current_dict[key].append(val)
            else:
                current_dict[key] = [val]

            f.write(json.dumps(current_dict))


key, val = get_args()

file_upd(key, val)
