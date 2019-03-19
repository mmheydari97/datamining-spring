import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


print("# ---------------- A ----------------")
# reading the csv file into a table
# column names are included in the file
df = pd.read_csv("../data.csv")

# displaying the table (commented due to its size)
# print(df)
# instead, showing first 5 and last 5 records

# printing first 5 rows
print(df.head())

# printing last 5 rows
print(df.tail())


print("# ---------------- B ----------------")
# size of data
print("size: ", df.size)

# number of columns and rows
print("(columns, rows): ", df.shape)

# information about each column including datatype
print(df.info())


print("# ---------------- C ----------------")
# remove dollar sign
df['Amount'] = df['Amount'].str.replace('$', '')

# remove excess comma for values greater than 1000
df['Amount'] = df['Amount'].str.replace(',', '')

# change type to float
df['Amount'] = df['Amount'].astype("float")

# print updated values
print(df['Amount'].head())
print(df['Amount'].tail())
# check data type
print(df.info())


print("# ---------------- D ----------------")
# BranchName has the same value for each record so we can drop it
# Week and DayWeek columns are also computable from other columns
# but we use them to plot data
print("counting records with BranchName of MyStore:",
      np.array(df["BranchName"] == "MyStore").sum())
# this column is unnecessary and we'll be dropped
df = df.drop("BranchName", 1)
# print(df)


print("# ---------------- E ----------------")
# describe data statistically
print(df.describe())


print("# ---------------- F ----------------")
# number of fields
print("number of fields in Days",
      df['Day'].unique(), ": ", len(df['Day'].unique()))

print("number of fields in Transaction_Type",
      df['Transaction_Type'].unique(), ": ", len(df['Transaction_Type'].unique()))


print("# ---------------- G ----------------")
print("is there any null values: ")
print(len(df) != df.count())

# in order to drop all null values
df.dropna()
# also: df['col_name'].dropna()


print("# ---------------- H ----------------")
print("plots")
# histogram for all columns
sells = df[df['Amount'] > 0]
buys = df[df['Amount'] <= 0]
plt.figure()
plt.style.use("bmh")
df.hist()
plt.savefig("hists.png")

plt.figure()
plt.hist([sells.Amount, buys.Amount*(-1)])
plt.xlabel("Amount ($)")
plt.legend(['Sells', 'Buys'])
plt.savefig("amount.png")

plt.figure()
plt.title("Units - Amount plot")
plt.scatter(sells.Amount, sells.Units)
plt.scatter(buys.Amount*(-1), buys.Units)
plt.xlabel("Amount ($)")
plt.ylabel("Units")
plt.legend(("buys", "sells"))
plt.savefig("scatter.png")

plt.figure()
plt.boxplot([sells.Amount, buys.Amount*(-1)], labels=['Sells', 'Buys'])
plt.ylabel("Amount ($)")
plt.savefig("box.png")

plt.show()


print("# ---------------- I ----------------")
print("""Considering the box plot, there are outliers in the values of Amount
we can filter Amounts existing in 95% or 99% quantiles and drop other records
however transactions over 1000$ are rare but possible
so we can go ahead and keep these records.
It seems like records are missed from February to June
looking at Month and Week histograms.

THE END
""")
