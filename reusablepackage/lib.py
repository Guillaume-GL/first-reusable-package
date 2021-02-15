from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
import matplotlib.pyplot as plt


def try_me():
    print("❌ First try, second guess ❔ ✅")


def generate_data():
    X, y = make_moons(n_samples=250, noise=0.20, random_state=0)
    return X, y


def plot_moons(X, y):
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()
    return plot_moons(X, y)


def split_pop(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def initialize_model():
    # Model architecture
    model = Sequential()

    # To do
    model.add(layers.Dense(5, activation='relu', input_dim=2))

    # To do
    model.add(layers.Dense(1, activation='sigmoid'))

    # Model optimization : Optimizer, loss and metric
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def summarized_model(model):
    return model.summary()


def fit_model(model, X_train, y_train, epochs_number=100, batch_size_choice=8, verbose_choice=0):
    history = model.fit(X_train, y_train, epochs=epochs_number, batch_size=batch_size_choice, verbose=verbose_choice)
    return history


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    return plt.show()


def evaluation(X_test, y_test):
    y_pred = model.predict(X_test)
    return model.evaluate(X_test, y_test, verbose=0)
