import importlib
from configs import load_cfg


def main():
    cfg = load_cfg()

    if cfg.train.enable:
        from tools.train import train
        train(cfg)

    if cfg.val.enable:
        from tools.val import val
        val(cfg)

    if cfg.test.enable:
        module_name = f"tools.test.{cfg.test.code}"
        try:
            mod = importlib.import_module(module_name)
            mod.main(cfg)
        except ImportError:
            print(f"Module {module_name} not found.")
        except AttributeError:
            print(f"No main(cfg) function found in {module_name}.")
        pass


if __name__ == '__main__':
    main()
