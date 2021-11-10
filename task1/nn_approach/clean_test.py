import csv
import pdb

csvFile = open("test_new.csv", "r")

cleaned_csvFile = open("cleaned_test.csv", "w+")
fileheader = ['','listing_id', 'title', 'model', 'description', 'manufactured', 
'reg_date', 'type_of_vehicle', 'category', 'transmission', 'curb_weight', 'power' , 'engine_cap', 
'no_of_owners', 'depreciation', 'coe', 'road_tax', 'dereg_value', 'mileage', 'omv', 'arf', 
'eco_category', 'features', 'accessories']
dict_writer = csv.DictWriter(cleaned_csvFile, fileheader)
dict_writer.writeheader()

dict_reader = csv.DictReader(csvFile)

for row in dict_reader:
	# key_values = row.keys()
	key_values = list()
	for key in row.keys():
		key_values.append(row[key])
	dict_a = {}
	for i in range(len(fileheader)):
		dict_a[str(fileheader[i])]= key_values[i]

	if '' not in key_values:
		# pdb.set_trace()
		dict_writer.writerow(dict_a)
	else:
		print(row)

	# ['original_reg_date', 'fuel_type', 'opc_scheme' ,'lifespan']