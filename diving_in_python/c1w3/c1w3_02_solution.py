import csv
import os

class CarBase:
    def __init__(self, brand, photo_file_name, carrying):
        self.car_type = None
        self.brand = brand
        self.photo_file_name = photo_file_name
        self.carrying = float(carrying)

    def get_photo_file_ext(self):
        return os.path.splitext(self.photo_file_name)[1]


class Car(CarBase):
    def __init__(self, brand, photo_file_name, carrying, passenger_seats_count):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'car'
        self.passenger_seats_count = int(passenger_seats_count)


class Truck(CarBase):
    def __init__(self, brand, photo_file_name, carrying, body_whl):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'truck'
        try:
            whl = list(map(float, body_whl.split('x')))
        except ValueError:
            whl = [0., 0., 0.]
        
        if len(whl) == 3:
            if (whl[0] > 0) and (whl[1] > 0) and (whl[2] > 0):
                pass
            else:
                whl = [0., 0., 0.]
        else:
            whl = [0., 0., 0.]
        
        self.body_length = float(whl[0])
        self.body_width  = float(whl[1])
        self.body_height = float(whl[2])

    def get_body_volume(self):
        return self.body_length * self.body_width * self.body_height

class SpecMachine(CarBase):
    def __init__(self, brand, photo_file_name, carrying, extra):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'spec_machine'
        self.extra = extra

def get_car_list(csv_filename):
    format_list = ['.jpg', '.jpeg', '.png', '.gif']
    car_type_list = ['car', 'truck', 'spec_machine']
    cars_list_raw = []
    cars_list = []
    try:
        with open(csv_filename) as csv_fd:
            reader = csv.reader(csv_fd, delimiter=';')
            next(reader)

            for row in reader:
                if len(row) < 7:
                    print('INPUT ERROR')
                elif row[0] not in car_type_list:
                    print('CAR TYPE ERROR')
                elif row[1] in ['', None]:
                    print('BRAND ERROR')
                elif (row[1] or row[3] or row[5]) in ['', None]:
                    print('INPUT ERROR')
                elif row[5] is None:
                        print('INPUT ERROR')
                elif os.path.splitext(row[3])[1] not in format_list:
                    print('PHOTO FILE ERROR')
                else:
                    try:
                        if float(row[5]) > 0:
                            cars_list_raw.append(row)
                    except ValueError:
                        print('INPUT ERROR')

            for row in cars_list_raw:
                if row[0] == 'car':
                    try:
                        if int(row[2]) > 0:
                            cars_list.append(
                                Car(brand=row[1],
                                    photo_file_name=row[3],
                                    carrying=row[5],
                                    passenger_seats_count=int(row[2])
                                )
                            )
                    except:
                        print('INPUT ERROR')
                elif row[0] == 'spec_machine':
                    if row[6] in ['', None]:
                        print('INPUT ERROR')
                    else:
                        cars_list.append(
                            SpecMachine(
                                brand=row[1],
                                photo_file_name=row[3],
                                carrying=row[5],
                                extra=row[6]
                            )
                        )
                else:
                    cars_list.append(
                        Truck(
                            brand=row[1],
                            photo_file_name=row[3],
                            carrying=row[5],
                            body_whl=row[4]
                        )
                    )
    except:
        print('INPUT FILE ERROR')
    return cars_list
