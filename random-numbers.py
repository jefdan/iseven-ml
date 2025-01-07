import pandas as pd
import numpy as np

test_data_path = "test_data.csv"
test_data_df = pd.read_csv(test_data_path)

random_numbers = np.random.randint(0, 2**16-1, size=1000)

random_numbers_df = pd.DataFrame(random_numbers, columns=["number"])

updated_test_data_df = pd.concat(
    [test_data_df, random_numbers_df],
    ignore_index=True
    )

updated_test_data_df.to_csv(test_data_path, index=False)

print("Added 1000 random numbers to test_data.csv.")
