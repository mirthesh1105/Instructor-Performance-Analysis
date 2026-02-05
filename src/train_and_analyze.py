# Data preprocessing, etc.
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Deep learning libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


DATA_PATH = "../data/EduPro_Online_Platform.xlsx"
RANDOM_SEED = 42


def load_merged_data(path):
    teachers = pd.read_excel(path, sheet_name="Teachers")
    courses = pd.read_excel(path, sheet_name="Courses")
    transactions = pd.read_excel(path, sheet_name="Transactions")

    merged = transactions.merge(courses, on="CourseID", how="left")
    merged = merged.merge(teachers, on="TeacherID", how="left")

    return merged


def prepare_features(df):
    target_col = "CourseRating"

    feature_cols = [
        "Age",
        "Gender",
        "Expertise",
        "YearsOfExperience",
        "CourseCategory",
        "CourseLevel",
        "CourseDuration",
        "CoursePrice"
    ]

    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols]
    y = df[target_col]

    numeric_cols = [
        "Age",
        "YearsOfExperience",
        "CourseDuration",
        "CoursePrice"
    ]

    categorical_cols = [
        "Gender",
        "Expertise",
        "CourseCategory",
        "CourseLevel"
    ]

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return X, y, transformer, df


def build_network(input_size):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_size))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_model(X, y, df, transformer):
    X_transformed = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    model = build_network(X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        verbose=1
    )

    predictions = model.predict(X_test).ravel()

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    test_df = df.iloc[y_test.index].copy()
    test_df["PredictedCourseRating"] = predictions

    return model, test_df


def summarize_instructors(test_df):
    summary = (
        test_df
        .groupby("TeacherName")
        .agg(
            ActualRating=("CourseRating", "mean"),
            PredictedRating=("PredictedCourseRating", "mean"),
            Experience=("YearsOfExperience", "mean")
        )
        .reset_index()
    )

    return summary


def analyze_results(summary_df):
    summary_df["PredictionError"] = (
        summary_df["ActualRating"] - summary_df["PredictedRating"]
    ).abs()

    high_rating_cutoff = summary_df["ActualRating"].quantile(0.75)

    consistent_instructors = summary_df[
        (summary_df["ActualRating"] >= high_rating_cutoff) &
        (summary_df["PredictionError"] <= 0.10)
    ]

    print("\nConsistently high-performing instructors:")
    print(consistent_instructors.sort_values("ActualRating", ascending=False))

    experience_corr = summary_df["Experience"].corr(summary_df["ActualRating"])
    print("\nExperience–rating correlation:", experience_corr)

    summary_df["ExperienceGroup"] = pd.cut(
        summary_df["Experience"],
        bins=[0, 3, 7, 12, 30],
        labels=["0–3", "4–7", "8–12", "12+"]
    )

    print("\nAverage rating by experience group:")
    print(summary_df.groupby("ExperienceGroup")["ActualRating"].mean())

    summary_df["DependenceScore"] = 1 / (summary_df["PredictionError"] + 1e-6)

    print("\nCourses strongly influenced by instructor quality:")
    print(
        summary_df
        .sort_values("DependenceScore", ascending=False)
        .head(10)[["TeacherName", "ActualRating", "PredictionError"]]
    )

    ratings = summary_df["ActualRating"].values

    def gini(values):
        values = np.sort(values)
        n = len(values)
        cumulative = np.cumsum(values)
        return (n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n

    print("\nRating distribution:")
    print(pd.Series(ratings).describe())
    print("Gini coefficient:", gini(ratings))


def main():
    data = load_merged_data(DATA_PATH)
    X, y, transformer, clean_df = prepare_features(data)

    model, test_df = train_model(X, y, clean_df, transformer)
    instructor_summary = summarize_instructors(test_df)

    analyze_results(instructor_summary)

    model.save("../models/instructor_performance_model.h5")


if __name__ == "__main__":
    main()
