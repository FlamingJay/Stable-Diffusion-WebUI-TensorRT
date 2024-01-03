import os

from modules import paths


def preload(parser):
    parser.add_argument("--control-dir", type=str, help="Path to directory with ControlNet networks.",
                        default=os.path.join(paths.extensions_dir, 'sd-webui-controlnet/models'))