class Request:
    """
    用户请求
    """

    def __init__(self, homeCloudlet, type, instruction, reward):
        """
        :param homeCloudlet: 该推理请求需要在哪个homecloudlet上执行
        :param type: 该推理请求的类型
        :param instruction: 指令数
        """
        self.homeCloudlet = homeCloudlet
        self.type = type
        self.instruction = instruction
        self.reward = reward
