import pandas as pd
from pandas.core.frame import DataFrame

from regression import NormalDistrubution, normalRegression

FILE = "data/penguins.csv"
columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
species = ["Adelie", "Chinstrap", "Gentoo"]


def read_data() -> tuple[pd.DataFrame, list[list[NormalDistrubution]], list[float]]:
    df = pd.read_csv("data/penguins.csv")
    df.drop(labels="island", axis="columns")
    df.drop(labels="sex", axis="columns")

    # btw, this sintax honestly sucks
    train_df: pd.DataFrame = df.sample(frac=0.75)  # random_state for seed
    test_df: pd.DataFrame = df.drop(train_df.index)

    Adelie_df: pd.DataFrame = train_df[train_df["species"] == "Adelie"]
    Chinstrap_df: pd.DataFrame = train_df[train_df["species"] == "Chinstrap"]
    Gentoo_df: pd.DataFrame = train_df[train_df["species"] == "Gentoo"]

    # frequancy of the species in the dataset
    Adelie_frequency: float = len(Adelie_df) / len(train_df)
    Chinstrap_frequency: float = len(Chinstrap_df) / len(train_df)
    Gentoo_frequency: float = len(Gentoo_df) / len(train_df)

    # we store a list of 4 distribution (one for each characteristhich) for every specie
    Adelie_distributions: list[NormalDistrubution] = []
    for col in columns:
        Adelie_distributions.append(normalRegression(Adelie_df[col]))

    Chinstrap_distributions: list[NormalDistrubution] = []
    for col in columns:
        Chinstrap_distributions.append(normalRegression(Chinstrap_df[col]))

    Gentoo_distributions: list[NormalDistrubution] = []
    for col in columns:
        Gentoo_distributions.append(normalRegression(Gentoo_df[col]))

    distributions: list[list[NormalDistrubution]] = [Adelie_distributions, Chinstrap_distributions, Gentoo_distributions]

    return (test_df, distributions, [Adelie_frequency, Chinstrap_frequency, Gentoo_frequency])


# check accuracy
def classify(data: pd.Series, distributions: list[list[NormalDistrubution]], frequencies: list[float]) -> str:
    results: list[float] = frequencies.copy()
    for i in range(len(distributions)):
        for j in range(len(columns)):
            results[i] *= distributions[i][j].value(data[columns[j]])
    max_index = results.index(max(results))
    return species[max_index]


def save_values() -> None:
    """
    Not implemented yet and probably never will

    it should compile and save the classify function to create a runnable without needing to approzimate every time the gaussian

    however this is useless for both learning (it is relatively simple) and speed, since the takes less than a second on my pc to
    classify all penguins in the dataset given the low amount of data provided
    """
    return


def main():
    processed_data: tuple[DataFrame, list[list[NormalDistrubution]], list[float]] = read_data()

    test_df: pd.DataFrame = processed_data[0]
    distributions: list[list[NormalDistrubution]] = processed_data[1]
    frequencies: list[float] = processed_data[2]
    COUNT: int = len(test_df)
    guessed: int = 0

    for i in range(COUNT):
        penguin: pd.Series = test_df.iloc[i]
        if penguin["species"] == classify(penguin, distributions, frequencies):
            guessed += 1

    print(f"We had an accuracy of {guessed / COUNT}")


if __name__ == "__main__":
    main()
