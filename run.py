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
        except ImportError:
            print(f"Module {module_name} not found.")
        else:
            if not hasattr(mod, "main") or not callable(mod.main):
                print(f"No main(cfg) function found in {module_name}.")
            else:
                try:
                    mod.main(cfg)
                except Exception as e:
                    print(f"Error occurred while running main(cfg) in {module_name}: {type(e).__name__}: {e}")
                    raise


if __name__ == '__main__':
    main()
