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
        pass


if __name__ == '__main__':
    main()
