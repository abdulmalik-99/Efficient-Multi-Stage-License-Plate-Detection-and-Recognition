import argparse
from ultralytics import YOLO

def train_and_validate(data_path, model_path, number_of_epochs):
    # Load a model
    model = YOLO(model_path)  # load a pretrained model (recommended for training)

    # Use the model for training
    model.train(data=data_path, epochs=number_of_epochs)

    # Validate the model
    model.val(data=data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate YOLO model.")
    parser.add_argument('--data_path', type=str, help='Path to the training data', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the pretrained model', required=True)
    parser.add_argument('--number_of_epochs', type=int, help='Number of training epochs', required=True)

    args = parser.parse_args()

    train_and_validate(args.data_path, args.model_path, args.number_of_epochs)
