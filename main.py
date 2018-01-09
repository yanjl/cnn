from cnn.lstm import add


class Data:
    def __init__(self, init=None):
        if init is None:
            self.container = []
        else:
            self.container = init

    def __getitem__(self, n):
        return self.container[n]


def main():
    print(add(2, 3))
    


if __name__ == '__main__':
    main()
