

class BaseEvaluator:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

    def reset(self):
        raise NotImplementedError

    def update(self, outputs, targets, loss, is_train):
        raise NotImplementedError

    def synchronize(self, is_train):
        # 聚合所有进程的结果
        raise NotImplementedError
