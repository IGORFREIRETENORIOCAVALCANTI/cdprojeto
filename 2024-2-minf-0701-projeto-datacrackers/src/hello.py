
class Hello():

    @staticmethod
    def message_implemented():
        return "Hello World"

    @staticmethod
    def message_not_implemented():
        raise NotImplementedError()