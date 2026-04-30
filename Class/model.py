class Model:
    """
    基座模型
    """

    def __init__(self, id, size):
        """
        :param id: 基座模型id
        :param size: 基座模型大小
        """
        self.id = id
        self.size = size


class Adapter:
    """
    适配器
    """

    def __init__(self, model_id, service_type, size, accuracy):
        """
        :param model_id: 隶属基座模型的id
        :param service_type: 服务的类型
        :param size: 适配器的大小
        :param accuracy: 精确度
        """
        self.model_id = model_id
        self.service_type = service_type
        self.size = size
        self.accuracy = accuracy
