from analyze import *

#only keep annotations where at least 2 raters have agreement
def filterData(data, emotions):
	print("Filtering data...")
	agree_dict_2 = data.groupby("id").apply(CheckAgreement, 2, emotions).to_dict()
	data["agree"] = data.id.map(agree_dict_2)
	filtered_data = data[data.agree.str.len() > 0].drop_duplicates("id")

	filtered_data.to_csv(FILTERED_DATA_FILE)
	print("")

	return filtered_data

def main():
	data = getData()
	data = data[data["example_very_unclear"] == False]
	emotions = getEmotions()
	data = filterData(data, emotions)
	print(data)


if __name__ == "__main__":
	main()