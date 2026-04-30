class User:
    """
    用户的类
    """

    def __init__(self, request, accuracy, delay):
        """
        :param request:用户发起的推理请求
        :param accuracy:用户需要的精确值
        :param delay:用户的延迟要求
        """
        self.request = request
        self.accuracy = accuracy
        self.delay = delay
