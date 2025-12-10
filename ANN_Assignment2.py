import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE  # <--- ADD THIS AT THE TOP

# ==========================================
# PART 1: DATA PREPROCESSING (As per Section 2.4)
# ==========================================
def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # 1. Drop customerID (Identifier)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 2. Handle TotalCharges (Convert to numeric, coerce errors to NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 3. Drop missing values (11 records as noted in report)
    df.dropna(inplace=True)

    # 4. Label Encoding for Categorical Variables
    # Identified categorical columns from Report Table 7
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Separate Features (X) and Target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 5. Min-Max Normalization (Scale to [0, 1])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# ==========================================
# PART 2: DATA SPLITTING (As per Section 3.1)
# ==========================================
def split_data(X, y):
    # First split: 70% Train, 30% Temp (Test + Val)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Second split: Split the 30% Temp into 50/50 -> 15% Val, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training Samples: {X_train.shape[0]} (70%)")
    print(f"Validation Samples: {X_val.shape[0]} (15%)")
    print(f"Testing Samples: {X_test.shape[0]} (15%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# PART 3: ANN ARCHITECTURE (Your Main Task - Section 3.3)
# ==========================================
def build_ann_model(input_dim):
    model = Sequential()

    # Hidden Layer 1: Switch to 'relu'
    model.add(Dense(12, input_dim=input_dim, activation='relu', name='Hidden_Layer_1'))

    # Hidden Layer 2: Switch to 'relu'
    model.add(Dense(8, activation='relu', name='Hidden_Layer_2'))

    # Output Layer: MUST stay 'sigmoid' for binary classification
    model.add(Dense(1, activation='sigmoid', name='Output_Layer'))

    return model


# ==========================================
# MAIN EXECUTION (Updated with SMOTE)
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    try:
        X, y = load_and_preprocess_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        exit()

    # 2. Split Data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Build Model
    model = build_ann_model(input_dim=X_train.shape[1])
    model.summary()

    # 4. Compile Model
    # Using 'adam' as previously discussed to ensure good convergence
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ---------------------------------------------------------
    # NEW CODE HERE: SMOTE RESAMPLING
    # ---------------------------------------------------------
    print("\n--- Applying SMOTE to handle Class Imbalance ---")
    smote = SMOTE(random_state=42)

    # IMPORTANT: fit_resample is ONLY applied to Training data
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Original Churn counts: {sum(y_train)}")
    print(f"New SMOTE Churn counts: {sum(y_train_resampled)}")
    # ---------------------------------------------------------

    # 5. Train Model (Using the NEW resampled data)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_resampled, y_train_resampled,  # <--- CHANGED: Using resampled data
        validation_data=(X_val, y_val),  # Validation data MUST remain original!
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 6. Evaluate & Graphs (Same as before)
    print("\n--- Final Test Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Graphs
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
        '''
        Draws a neural network using Matplotlib.
        :param ax: matplotlib axes object
        :param left, right, bottom, top: coordinates of the drawing area
        :param layer_sizes: list of numbers of neurons in each layer
        '''
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(len(layer_sizes) - 1)

        # Input layer is huge (20 nodes), so we scale the visual node size
        # to fit them all in.
        node_radius = 0.5

        # Iterate over layers
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.

            # Calculate node positions for this layer
            for m in range(layer_size):
                x = n * h_spacing + left
                y = layer_top - m * v_spacing

                # Draw Circle (Neuron)
                circle = plt.Circle((x, y), v_spacing / 4., color='#6699cc', ec='k', zorder=4)
                ax.add_artist(circle)

                # Add text annotation
                # Only label first, last, and middle nodes to avoid clutter
                if n == 0:  # Input Layer
                    if m == 0 or m == layer_size - 1:
                        ax.text(x - 0.5, y, f'In {m + 1}', ha='right', va='center', fontsize=8)
                elif n == n_layers - 1:  # Output Layer
                    ax.text(x + 0.5, y, f'Out', ha='left', va='center', fontsize=10)

                # Draw Lines (Weights) to next layer
                if n < n_layers - 1:
                    next_layer_size = layer_sizes[n + 1]
                    next_layer_top = v_spacing * (next_layer_size - 1) / 2. + (top + bottom) / 2.

                    for o in range(next_layer_size):
                        next_x = (n + 1) * h_spacing + left
                        next_y = next_layer_top - o * v_spacing

                        # Draw Line
                        line = plt.Line2D([x, next_x], [y, next_y], c='#cccccc', alpha=0.5)
                        ax.add_artist(line)


    # --- EXECUTE VISUALIZATION ---
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')

    # Your specific architecture from the report: 20 -> 12 -> 8 -> 1
    architecture = [20, 12, 8, 1]

    draw_neural_net(ax, .1, .9, .1, .9, architecture)

    plt.title("ANN Architecture: 20 Inputs -> 12 Hidden -> 8 Hidden -> 1 Output", fontsize=15)
    plt.show()

    # 6. Evaluate (Section 3.5 - Member 4 will analyze this, but you need to generate it)
    print("\n--- Final Test Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")