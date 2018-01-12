import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

root_path = osp.join(this_dir, '..')
add_path(root_path)

lib_path = osp.join(this_dir, '../lib')
add_path(lib_path)

src_path = osp.join(this_dir, '../src')
add_path(src_path)