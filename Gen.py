import pandas as pd
import random
import numpy as np

# Define a list of common UK cities and their average house prices
cities = {
    "London": 675000,
    "Manchester": 250000,
    "Birmingham": 230000,
    "Leeds": 220000,
    "Liverpool": 190000,
    "Bristol": 350000,
    "Newcastle": 180000,
    "Sheffield": 200000,
    "Bradford": 170000,
    "Nottingham": 210000
}

# Function to generate a random price based on a city's average
def generate_price(city):
    avg_price = cities[city]
    std_dev = avg_price * 0.2
    price = np.random.normal(avg_price, std_dev)
    return round(max(price, 50000))

# Function to generate a random built date within a reasonable range
def generate_built_date():
    year = random.randint(1900, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"

# Function to generate a random number of bedrooms
def generate_n_bedrooms():
    return random.randint(1, 4)

# Generate a dataset with 1000 rows
data = []
for _ in range(1000):
    city = random.choice(list(cities.keys()))
    price = generate_price(city)
    built_date = generate_built_date()
    n_bedrooms = generate_n_bedrooms()
    data.append([city, price, built_date, n_bedrooms])

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=["City", "Price", "Built Date", "NBedrooms"])

# Save the DataFrame as a CSV file
df.to_csv("houses.csv", index=False)

print("Dataset created successfully.")