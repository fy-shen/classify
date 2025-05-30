import argparse
from pathlib import Path
from omegaconf import OmegaConf


def load_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Config file path")
    parser.add_argument('overrides', nargs=argparse.REMAINDER,
                        help='Any args to override config, e.g., TRAIN.batch_size=64')
    args = parser.parse_args()

    if not Path(args.cfg).exists():
        parser.error(f"Config file not found: {args.cfg}")

    base_cfg = OmegaConf.load('configs/defaults.yaml')
    run_cfg = OmegaConf.load(args.cfg)
    cfg = OmegaConf.merge(base_cfg, run_cfg)
    add_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, add_cfg)
    cfg.cfg = args.cfg
    return cfg
