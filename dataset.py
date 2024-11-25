import pandas as pd

def _reorder_words(genre):
    words = genre.split(" ")
    words.sort()
    return " ".join(words)

def _find_parent_genre(genre, parents):
    for parent in parents:
        if parent in genre:
            return parent
    return genre

def preprocess_characters(df):
    df["Genre"] = df["Genre"].str.lower()
    df["Genre"] = df["Genre"].str.replace("'", "")
    df["Genre"] = df["Genre"].str.replace(r"\[.*\]", "", regex=True)
    df["Genre"] = df["Genre"].str.replace(r"[&(),\-./]", " ", regex=True)
    df["Genre"] = df["Genre"].str.replace(r"\s+", " ", regex=True)
    df = df[df["Genre"].str.match(r"^[a-z ]+$")]
    return df

def preprocess_substitutions(df):
    df["Genre"] = df["Genre"].str.replace("sci fi", "scifi")
    df["Genre"] = df["Genre"].str.replace("science fiction", "scifi")
    df["Genre"] = df["Genre"].str.replace("romantic", "romance")
    return df

def preprocess_cleanup(df):
    df["Genre"] = df["Genre"].str.strip()
    df = df.drop(df[df["Genre"] == "unknown"].index)
    df["Genre"] = df["Genre"].apply(_reorder_words)
    return df

def preprocess_underrepresented(df, limit):
    top = df["Genre"].value_counts().nlargest(limit)
    parents = list(reversed(top.index)) # ascending
    df["Genre"] = df["Genre"].apply(lambda genre: _find_parent_genre(genre, parents))
    return df

def preprocess_keep_top(df, limit):
    top = df["Genre"].value_counts().nlargest(limit)
    parents = list(top.index)
    return df[df["Genre"].isin(parents)]

def preprocess(df, limit=20):
    df = df.drop(columns=["Origin/Ethnicity", "Director", "Cast", "Wiki Page", "Release Year"])
    df = preprocess_characters(df)
    df = preprocess_substitutions(df)
    df = preprocess_cleanup(df)
    df = preprocess_underrepresented(df, limit)
    df = preprocess_keep_top(df, limit)
    return df

def load_dataset():
    return pd.read_csv("wiki_movie_plots_deduped.csv")
