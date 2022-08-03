# Import matplotlib.pyplot
import matplotlib.pyplot as plt
import pandas as pd

# Calculate number of unique values for each label: num_unique_labels
df = pd.DataFrame()
LABELS = ['']
num_unique_labels = df[LABELS].apply(pd.Series.nunique, axis=0)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

plt.xlabel('Labels')
plt.ylabel('Number of unique values')

plt.show()
