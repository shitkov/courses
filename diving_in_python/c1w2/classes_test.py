class Pet:
    def __init__(self, name=None):
        self.name = name


class Dog(Pet):
    def __init__(self, name, breed=None):
        super().__init__(name)
        self.breed = breed

    def say(self):
        return "{0}: waw".format(self.name)


class CorgiDog(Dog):
    def say(self):
        return "CCC"


dog = Dog("Sharik", "Corgi")
print(dog.say())

cdog = CorgiDog('1', '2')
print(cdog.say())
