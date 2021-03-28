def dictionairy():
    # 声明字典
    key_value = {'222': 4, '1': 6, '5': 2, '4': 4, '6': 3, '44': 1}

    # 初始化

    for k in key_value:
        print(k)

    print("按值(value)排序:")
    print(sorted(key_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))


def main():
    dictionairy()


if __name__ == "__main__":
    main()

class A:
    def __init__(self):
        self.a = 1

    def loop(self):
        for i in range(10):
            self.a = i

aa = A()
aa.loop()
print(aa.a)

