import csv
cols = ["feat_{}".format(i) for i in range(0, 7 * 7 * 512)]
fieldnames = ["class"] + cols
with open('train.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(fieldnames)

    with open('train.csv', 'r', newline='') as incsv:
        reader = csv.reader(incsv)
        writer.writerows(row for row in reader)

    incsv.close()
outcsv.close()

# with open('tain.csv', 'r', newline='') as incsv:
#     reader = csv.reader(incsv)
#     writer.writerows(row for row in reader)
#
# incsv.close()