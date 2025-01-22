import os
import csv

text_log_file = open('1_2k_random_test.csv', 'w')
writer = csv.writer(text_log_file)
writer.writerow(['IMG_FILE', 'LAT', 'LON'])

images = os.listdir('./2k_random_test')
for image_name in images:
    image_path = os.path.join('./2k_random_test', image_name)


    img = open(image_path, 'rb')
    img_as_bytes = img.read()
    img_as_bytes_as_hex_string = img_as_bytes.hex()

    location_of_latitude_in_string = img_as_bytes_as_hex_string.find('6c61746974756465')
    found = img_as_bytes_as_hex_string[location_of_latitude_in_string:location_of_latitude_in_string+200]
    split_sections = found.split('fffe')

    string = [bytes.fromhex(i).decode('utf-8').split('\x00') for i in split_sections]

    flattened = [item for sublist in string for item in sublist]
    latitude = "".join(list(filter(lambda x: 'latitude' in x, flattened))).split(': ')[1]
    longitude = "".join(list(filter(lambda x: 'longitude' in x, flattened))).split(': ')[1]

    writer.writerow([image_name, latitude, longitude])