class MyProduct:
    # 생성자
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0

product_1 = MyProduct("cake", 500, 20) 
print(product_1.stock)
# 20


class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    def buy_up(self, n):
        self.stock += n
    def sell(self, n):
        self.stock -= n
        self.sales += n*self.price
    def summary(self):
        message = "called summary().\n name: " + self.name + \
        "\n price: " + str(self.price) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message)





class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    def summary(self):
        message = "called summary()." + \
        "\n name: " + self.get_name() + \
        "\n price: " + str(self.price) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message)
    def get_name(self):
        return self.name

    def discount(self, n):
        self.price -= n

product_2 = MyProduct("phone", 30000, 100)
product_2.discount(5000)
product_2.summary()
# called summary().
#  name: phone
#  price: 25000
#  stock: 100
#  sales: 0





class MyProductSalesTax(MyProduct):
    def __init__(self, name, price, stock, tax_rate):
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate

    def get_name(self):
        return self.name + "(세금 포함)"

    def get_price_with_tax(self):
        return int(self.price * (1 + self.tax_rate))


product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()

# phone(세금 포함)
# 33000
# called summary().
#  name: phone(세금 포함)
#  price: 30000
#  stock: 100
#  sales: 0



class MyProductSalesTax(MyProduct):
    def __init__(self, name, price, stock, tax_rate):
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate

    def get_name(self):
        return self.name + "(세금 포함)"

    def get_price_with_tax(self):
        return int(self.price * (1 + self.tax_rate))

    def summary(self):
        message = "called summary().\n name: " + self.get_name() + \
        "\n price: " + str(self.get_price_with_tax()+0) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message) 

product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()
# phone(세금 포함)
# 33000
# called summary().
#  name: phone(세금 포함)
#  price: 33000
#  stock: 100
#  sales: 0









pai = 3.141592
print("원주율은 %f" % pai)
print("원주율은 %.2f" % pai)
# 원주율은 3.141592
# 원주율은 3.14




def bmi(height, weight):
    return weight / height**2
print("bmi는 %.4f입니다" % bmi(1.65, 65))
# bmi는 23.8751입니다





def check_character(object, character):
    return object.count(character)

print(check_character([1, 3, 4, 5, 6, 4, 3, 2, 1, 3, 3, 4, 3], 3))
print(check_character("asdgaoirnoiafvnwoeo", "d"))
# 5
# 1







def binary_search(numbers, target_number):
    low = 0
    high = len(numbers)
    while low <= high:
        middle = (low + high) // 2
        if numbers[middle] == target_number:
            print("{1}은(는) {0}번째에 있습니다".format(middle, target_number))
            break
        elif numbers[middle] < target_number:
            low = middle + 1
        else:
            high = middle - 1

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
target_number = 11
binary_search(numbers, target_number)
# 11은(는) 10번째에 있습니다