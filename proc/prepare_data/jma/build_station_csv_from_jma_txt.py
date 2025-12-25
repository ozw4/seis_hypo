from jma.parse_station_txt import build_station_csv_from_jma_txt

if __name__ == '__main__':
	input_file = '/workspace/data/station/jma/station.txt'
	output_file = '/workspace/data/station/jma/station.csv'
	build_station_csv_from_jma_txt(input_file, output_file)
