import os
import tempfile


class File:
    # def __new__(self, path_to_file):
    #     """в конструктор передается полный путь
    #     до файла на файловой системе"""
    #     self.path_to_file = path_to_file

    def __init__(self, path_to_file):
        """Если файла с таким путем не существует,
        он должен быть создан при инициализации"""
        self.current = 0
        self.end = 0
        self.path_to_file = path_to_file
        if os.path.exists(self.path_to_file):
            pass
        else:
            open(self.path_to_file, 'a').close()

    def __str__(self):
        """возвращать полный путь до файла"""
        return self.path_to_file

    def __add__(self, obj):
        """сложение объектов типа File, результатом сложения
        является объект класса File,
        при этом создается новый файл и файловый объект,
        в котором содержимое второго файла добавляется
        к содержимому первого файла.
        Новый файл должен создаваться в директории,
        полученной с помощью функции tempfile.gettempdir.
        Для получения нового пути можно использовать os.path.join."""
        new_file_dir = tempfile.gettempdir()
        new_file_path = os.path.join(new_file_dir, "new.txt")
        open(new_file_path, 'a').close()
        new_file = File(new_file_path)
        with open(self.path_to_file) as f:
            data1 = f.read()
        with open(obj.path_to_file) as f:
            data2 = f.read()
        with open(new_file.path_to_file, 'w') as f:
            f.write(data2 + data1)
        return new_file

    def __iter__(self):
        """поддерживать протокол итерации"""
        return self

    def __next__(self):
        """итерация проходит по строкам файла"""
        f = open(self.path_to_file, 'r')
        data = f.readlines()
        self.end = len(data)
        if self.current >= self.end:
            raise StopIteration
        result = data[self.current]
        self.current += 1
        f.close()
        return result

    def read(self):
        """возвращает строку с текущим содержанием файла"""
        f = open(self.path_to_file, 'r')
        data = f.read()
        return data

    def write(self, data):
        """принимает в качестве аргумента строку с новым содержанием файла"""
        with open(self.path_to_file, 'w') as f:
            f.write(data)
