import os


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def load_label_map(txt_path):
    label_map = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if parts[0].isdigit() and len(parts) == 2:
                # <index> <name>
                label_idx = int(parts[0])
                label_name = parts[1]
            else:
                # <name>
                label_idx = idx
                label_name = line

            label_map[label_idx] = label_name
    return label_map
