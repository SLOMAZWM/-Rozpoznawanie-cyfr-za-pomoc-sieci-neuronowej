# mnist_classification.py

# Importowanie niezbędnych bibliotek
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Funkcja do wyświetlania przykładowych obrazów
def plot_samples(x, y, num=10):
    plt.figure(figsize=(10,1))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(x[i], cmap='gray')
        plt.title(y[i])
        plt.axis('off')
    plt.show()

def main():
    # 1. Ładowanie i normalizacja danych MNIST
    print("Ładowanie danych MNIST...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Sprawdzenie kształtu danych
    print("Kształt zbioru treningowego:", x_train.shape)
    print("Kształt zbioru testowego:", x_test.shape)

    # Wyświetlenie przykładowych obrazów
    print("Wyświetlanie przykładowych obrazów...")
    plot_samples(x_train, y_train, num=10)

    # 2. Przygotowanie danych do modelu
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    print("Zmieniono kształt danych do:", x_train.shape, x_test.shape)

    # 3. Definiowanie architektury modelu
    print("Definiowanie modelu CNN...")
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Podsumowanie modelu
    model.summary()

    # 4. Kompilacja modelu
    print("Kompilacja modelu...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Trenowanie modelu
    print("Rozpoczynanie trenowania modelu...")
    history = model.fit(x_train, y_train, epochs=10, 
                        validation_data=(x_test, y_test))

    # 6. Wizualizacja dokładności i straty
    print("Wizualizacja wyników trenowania...")
    # Dokładność
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Dokładność treningu')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    plt.title('Dokładność Modelu')

    # Strata
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Strata treningu')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.title('Strata Modelu')

    plt.tight_layout()
    plt.show()

    # 7. Ocena modelu na zbiorze testowym
    print("Ocena modelu na zbiorze testowym...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nDokładność na zbiorze testowym:', test_acc)

    # 8. Predykcje i raport klasyfikacji
    print("Generowanie raportu klasyfikacji i macierzy konfuzji...")
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Raport klasyfikacji
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred_classes))

    # Macierz konfuzji
    print("Macierz konfuzji:")
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print(conf_matrix)

if __name__ == "__main__":
    main()
