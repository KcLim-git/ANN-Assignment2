import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# =============================================================================
# PART 1: DATA PREPROCESSING PIPELINE
# Purpose: Clean, encode, and normalize raw data for neural network ingestion.
# =============================================================================
def load_and_preprocess_data(filepath):
    """
    Loads the dataset and performs data conditioning including:
    1. Removal of identifiers.
    2. Handling of non-numeric data types.
    3. Removal of missing values.
    4. Encoding of categorical features.
    5. Normalization of numerical features.
    """
    # Load the raw dataset from CSV
    df = pd.read_csv(filepath)

    # 1. Feature Selection: Drop irrelevant identifiers
    # The 'customerID' column is unique to each row and carries no predictive
    # information regarding churn status. It is removed to reduce noise.
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 2. Data Cleaning: Type Conversion
    # The 'TotalCharges' column may contain empty strings forcing it to object type.
    # These are coerced to NaN (Not a Number) to allow numerical processing.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 3. Handling Missing Values
    # Rows containing NaN values (specifically in 'TotalCharges') are removed.
    # This ensures the dataset is complete before training.
    df.dropna(inplace=True)

    # 4. Categorical Encoding (Label Encoding)
    # Neural networks require numerical input. Categorical variables are identified
    # and transformed into integer labels (e.g., 'Yes' -> 1, 'No' -> 0).
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

    # Feature Separation
    # Separation of independent variables (X) and the dependent target variable (y).
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 5. Normalization (Min-Max Scaling)
    # All features are scaled to the range [0, 1]. This prevents features with
    # large magnitudes (like TotalCharges) from dominating the gradients
    # and ensures compatibility with activation functions.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# =============================================================================
# PART 2: DATA PARTITIONING STRATEGY
# Purpose: Split data into Train (70%), Validation (15%), and Test (15%) sets.
# =============================================================================
def split_data(X, y):
    """
    Splits the dataset into three independent subsets to ensure robust evaluation.
    Stratified sampling is used to preserve the class distribution across all sets.
    """
    # Primary Split: Separates 70% for Training and 30% for a Temporary set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Secondary Split: Divides the Temporary set equally (50/50) into Validation and Testing.
    # This results in 15% Validation and 15% Testing overall.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Output sample counts for verification
    print(f"Training Samples: {X_train.shape[0]} (70%)")
    print(f"Validation Samples: {X_val.shape[0]} (15%)")
    print(f"Testing Samples: {X_test.shape[0]} (15%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# PART 3: ANN ARCHITECTURE DESIGN
# Purpose: Define the topology of the Multilayer Perceptron (MLP).
# =============================================================================
def build_ann_model(input_dim):
    """
    Constructs a Feedforward Neural Network using the Sequential API.
    Architecture: 20 Inputs -> 12 Hidden (ReLU) -> 8 Hidden (ReLU) -> 1 Output (Sigmoid).
    """
    model = Sequential()

    # Hidden Layer 1
    # Configuration: 12 Neurons, ReLU Activation.
    # Function: Extracts primary non-linear features from the 20 input variables.
    # ReLU is selected to mitigate the vanishing gradient problem.
    model.add(Dense(12, input_dim=input_dim, activation='relu', name='Hidden_Layer_1'))

    # Hidden Layer 2
    # Configuration: 8 Neurons, ReLU Activation.
    # Function: Refines features into higher-level representations, compressing
    # information before the final classification.
    model.add(Dense(8, activation='relu', name='Hidden_Layer_2'))

    # Output Layer
    # Configuration: 1 Neuron, Sigmoid Activation.
    # Function: Produces a probability score between 0 and 1.
    # Sigmoid is required for binary classification tasks (Churn vs No Churn).
    model.add(Dense(1, activation='sigmoid', name='Output_Layer'))

    return model


# =============================================================================
# PART 4: ARCHITECTURE VISUALIZATION UTILITY
# Purpose: Generate a visual diagram of the neural network nodes and connections.
# =============================================================================
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """
    Draws a stylistic representation of the Neural Network using Matplotlib primitives.
    Visualizes layers as columns of circles (neurons) and weights as connecting lines.
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Iterate through each layer to draw nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.

        for m in range(layer_size):
            x = n * h_spacing + left
            y = layer_top - m * v_spacing

            # Create Neuron (Circle)
            circle = plt.Circle((x, y), v_spacing / 4., color='#6699cc', ec='k', zorder=4)
            ax.add_artist(circle)

            # Annotate specific nodes for clarity (Input/Output)
            if n == 0:
                if m == 0 or m == layer_size - 1:
                    ax.text(x - 0.5, y, f'In {m + 1}', ha='right', va='center', fontsize=8)
            elif n == n_layers - 1:
                ax.text(x + 0.5, y, f'Out', ha='left', va='center', fontsize=10)

            # Draw Weights (Lines) connecting to the subsequent layer
            if n < n_layers - 1:
                next_layer_size = layer_sizes[n + 1]
                next_layer_top = v_spacing * (next_layer_size - 1) / 2. + (top + bottom) / 2.

                for o in range(next_layer_size):
                    next_x = (n + 1) * h_spacing + left
                    next_y = next_layer_top - o * v_spacing
                    line = plt.Line2D([x, next_x], [y, next_y], c='#cccccc', alpha=0.5)
                    ax.add_artist(line)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # 1. Data Acquisition and Preprocessing
    # Attempts to load the dataset and prepare it for the model.
    try:
        X, y = load_and_preprocess_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        print("Error: Dataset file not found. Ensure the CSV is in the correct directory.")
        exit()

    # 2. Data Splitting
    # Generates Train, Validation, and Test sets.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Model Initialization
    # Instantiates the defined ANN architecture based on input feature dimensions.
    model = build_ann_model(input_dim=X_train.shape[1])

    # Output model summary to console (Used for report documentation)
    model.summary()

    # 4. Model Compilation
    # Configures the training process.
    # Optimizer: Adam (Adaptive Moment Estimation) for efficient gradient descent.
    # Loss Function: Binary Cross-Entropy, standard for binary classification.
    # Metrics: Accuracy is tracked to monitor performance.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. Class Imbalance Handling (SMOTE)
    # The dataset exhibits class imbalance (approx. 73% No Churn vs 27% Churn).
    # SMOTE (Synthetic Minority Over-sampling Technique) is applied to the
    # TRAINING set only. This creates synthetic instances of the minority class
    # to achieve a balanced distribution, preventing the model from favoring the majority class.
    print("\n--- Applying SMOTE to handle Class Imbalance ---")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Original Churn counts: {sum(y_train)}")
    print(f"New SMOTE Churn counts: {sum(y_train_resampled)}")

    # 6. Model Training
    # The model is trained using the resampled (balanced) data.
    # Early Stopping is implemented to halt training if validation accuracy
    # fails to improve for 10 consecutive epochs, preventing overfitting.
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Final Evaluation
    # The trained model is evaluated on the independent Test set (unseen data).
    print("\n--- Final Test Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # =============================================================================
    # VISUALIZATION GENERATION
    # =============================================================================

    # Figure 1: Training Metrics (Accuracy and Loss)
    # Plots the history of accuracy and loss over epochs for both training and validation.
    # Convergence of lines indicates successful learning; divergence indicates overfitting.
    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Figure 2: Confusion Matrix
    # Visualizes the performance in terms of True Positives, True Negatives,
    # False Positives, and False Negatives.
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    # Generate the detailed report
    print("\n--- Classification Report (Precision, Recall, F1-Score) ---")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    # Figure 3: ANN Architecture Diagram
    # Generates a visual representation of the network topology: 20 -> 12 -> 8 -> 1.
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')

    architecture = [20, 12, 8, 1]
    draw_neural_net(ax, .1, .9, .1, .9, architecture)

    plt.title("ANN Architecture: 20 Inputs -> 12 Hidden -> 8 Hidden -> 1 Output", fontsize=15)
    plt.show()