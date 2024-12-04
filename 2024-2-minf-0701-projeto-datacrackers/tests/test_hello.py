
from src.hello import Hello

class TestHello():

    def test_hello_world_implemented(self):
        expected = "Hello World"
        actual = Hello.message_implemented()
        assert expected == actual

    def test_hello_world_not_implemented(self):
        expected = "Hello World"
        actual = Hello.message_implemented()
        assert expected == actual
