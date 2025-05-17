import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from ultralytics import YOLO
import time
import json
import datetime

# Initialize face detection model
face_detector = YOLO('model/yolov11n-face.pt')

# Function to extract faces from images
def extract_faces(image_path, min_confidence=0.5, padding=0.1):
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    results = face_detector(img)
    faces = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            if conf > min_confidence:
                x1, y1, x2, y2 = map(int, box)
                
                h, w = img.shape[:2]
                padding_x = int((x2 - x1) * padding)
                padding_y = int((y2 - y1) * padding)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(w, x2 + padding_x)
                y2 = min(h, y2 + padding_y)
                
                face_img = img[y1:y2, x1:x2]
                # Resize for consistency
                face_img = cv2.resize(face_img, (224, 224))
                faces.append(face_img)
    
    return faces

# Function to prepare dataset with augmentation and face alignment
def prepare_dataset(dataset_path, augmentation=True, align_faces=True, min_confidence=0.5, verbose=True):
    data = []
    labels = []
    face_count = 0
    skipped_count = 0
    
    # Get the list of people directories
    people_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    total_people = len(people_dirs)
    
    if verbose:
        print(f"Found {total_people} people in the dataset.")
    
    # Loop over each person's directory
    for i, person_name in enumerate(people_dirs):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        person_images = [f for f in os.listdir(person_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        person_face_count = 0
        
        if verbose:
            print(f"Processing {person_name} ({i+1}/{total_people}): {len(person_images)} images found")
        
        # Process each image of this person
        for image_file in person_images:
            image_path = os.path.join(person_dir, image_file)
            
            # Extract faces
            faces = extract_faces(image_path, min_confidence)
            
            if not faces:
                skipped_count += 1
                continue
                
            # Add each face to the dataset
            for face in faces:
                # Convert to RGB and normalize
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = face.astype("float") / 255.0
                
                data.append(face)
                labels.append(person_name)
                person_face_count += 1
                face_count += 1
        
        if verbose:
            print(f"  - Extracted {person_face_count} faces for {person_name}")
    
    if verbose:
        print(f"Total faces extracted: {face_count}")
        print(f"Images skipped (no faces detected): {skipped_count}")
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    
    return X_train, X_test, y_train, y_test, le

# Function to build face recognition model with different architecture options
def build_recognition_model(num_classes, architecture="mobilenet", learning_rate=0.001):
    if architecture.lower() == "mobilenet":
        # Use MobileNetV2 as base model (efficient for mobile/edge devices)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif architecture.lower() == "resnet":
        # Use ResNet50V2 (better accuracy, but larger)
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif architecture.lower() == "efficientnet":
        # Use EfficientNetB0 (good balance of accuracy and efficiency)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Freeze the base model
    base_model.trainable = False
    
    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with Adam optimizer and specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to get data augmentation generator
def get_data_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

# Function to plot training history
def plot_training_history(history, output_path):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to save model config and metadata
def save_model_metadata(model, label_encoder, model_path, history, training_params):
    metadata = {
        "model_path": model_path,
        "classes": label_encoder.classes_.tolist(),
        "num_classes": len(label_encoder.classes_),
        "training_params": training_params,
        "training_results": {
            "final_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "best_val_accuracy": float(max(history.history['val_accuracy'])),
            "epoch_count": len(history.history['accuracy']),
        },
        "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata

# Function to enable fine-tuning of the base model
def fine_tune_model(model, epochs=10, learning_rate=0.0001):
    # Unfreeze the base model
    model.layers[0].trainable = True
    
    # Recompile with a lower learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Training function
def train_recognition_model(dataset_path, model_save_path='model_output', architecture="mobilenet",
                           epochs=20, batch_size=32, use_augmentation=True, 
                           fine_tune=True, fine_tune_epochs=10, learning_rate=0.001,
                           patience=5, min_face_confidence=0.5, verbose=True):
    
    # Create output directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Record all training parameters
    training_params = {
        "dataset_path": dataset_path,
        "architecture": architecture,
        "epochs": epochs,
        "batch_size": batch_size,
        "use_augmentation": use_augmentation,
        "fine_tune": fine_tune,
        "fine_tune_epochs": fine_tune_epochs,
        "learning_rate": learning_rate,
        "min_face_confidence": min_face_confidence
    }
    
    # Prepare data
    if verbose:
        print("Preparing dataset...")
        start_time = time.time()
    
    X_train, X_test, y_train, y_test, label_encoder = prepare_dataset(
        dataset_path, 
        augmentation=use_augmentation,
        min_confidence=min_face_confidence,
        verbose=verbose
    )
    
    if verbose:
        print(f"Dataset preparation completed in {time.time() - start_time:.2f} seconds")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Classes: {label_encoder.classes_}")
    
    # Get current timestamp for model versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"face_recognition_{architecture}_{timestamp}.h5"
    model_path = os.path.join(model_save_path, model_filename)
    encoder_path = os.path.join(model_save_path, 'label_encoder.npy')
    
    # Build model
    if verbose:
        print(f"Building model with {architecture} architecture...")
        
    model = build_recognition_model(len(label_encoder.classes_), 
                                   architecture=architecture,
                                   learning_rate=learning_rate)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_accuracy',
        verbose=1, 
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    log_dir = os.path.join(model_save_path, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=log_dir)
    
    callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    # Use data augmentation if requested
    if use_augmentation:
        if verbose:
            print("Using data augmentation...")
        datagen = get_data_augmentation()
        datagen.fit(X_train)
        
        # Train model with augmentation
        if verbose:
            print("Training model (initial phase)...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
    else:
        # Train model without augmentation
        if verbose:
            print("Training model (initial phase)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
    
    # Fine-tune the model if requested
    if fine_tune:
        if verbose:
            print("Fine-tuning the model...")
        
        # Fine-tune the base model
        model = fine_tune_model(model, epochs=fine_tune_epochs, learning_rate=learning_rate/10)
        
        # Update the model path for fine-tuned model
        fine_tune_model_path = os.path.join(
            model_save_path, 
            f"face_recognition_{architecture}_finetuned_{timestamp}.h5"
        )
        
        # Update the checkpoint callback
        checkpoint = ModelCheckpoint(
            fine_tune_model_path, 
            monitor='val_accuracy',
            verbose=1, 
            save_best_only=True,
            mode='max'
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]
        
        # Train with fine-tuning
        if use_augmentation:
            fine_tune_history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_test, y_test),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )
        else:
            fine_tune_history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
        
        # Load the best model
        model = load_model(fine_tune_model_path)
        model_path = fine_tune_model_path
    
    # Save label encoder
    np.save(encoder_path, label_encoder.classes_)
    
    # Create symlinks to the latest model for easy access
    latest_model_path = os.path.join(model_save_path, 'face_recognition_model.h5')
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    
    # On Windows, we might need to copy instead of symlink
    if os.name == 'nt':  # Windows
        import shutil
        shutil.copy2(model_path, latest_model_path)
    else:  # Unix-like
        os.symlink(os.path.basename(model_path), latest_model_path)
    
    # Plot training history
    history_plot_path = os.path.join(model_save_path, f"training_history_{timestamp}.png")
    plot_training_history(history, history_plot_path)
    
    # Save model metadata
    metadata = save_model_metadata(model, label_encoder, model_path, history, training_params)
    
    if verbose:
        print(f"Model trained and saved to {model_path}!")
        print(f"Label encoder saved to {encoder_path}!")
        print(f"Training history plot saved to {history_plot_path}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, label_encoder, history, metadata

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Face Recognition Model')
    parser.add_argument('--dataset', type=str, default="jpg-images/",
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default="model_output",
                        help='Path to save model and related files')
    parser.add_argument('--architecture', type=str, default="mobilenet",
                        choices=["mobilenet", "resnet", "efficientnet"],
                        help='Base model architecture')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no_fine_tune', action='store_true',
                        help='Disable fine-tuning')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--min_face_confidence', type=float, default=0.5,
                        help='Minimum confidence for face detection')
    
    args = parser.parse_args()
    
    # Train model with parsed arguments
    model, label_encoder, history, metadata = train_recognition_model(
        dataset_path=args.dataset,
        model_save_path=args.output,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_face_confidence=args.min_face_confidence
    )
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {args.output}")
    print(f"Number of people recognized: {len(label_encoder.classes_)}")
    print(f"People names: {label_encoder.classes_}")
