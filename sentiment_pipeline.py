import kfp
from kfp import dsl
from kfp.dsl import component

# Step 1: Data Preprocessing
@component
def preprocess_op(data_path: str, output_path: str) -> str:
    # Simulating preprocessing (e.g., tokenization, vectorization)
    print(f"Preprocessing data from: {data_path}")
    # Here you would have the actual preprocessing code, e.g., CSV to TFRecord or tokenization
    processed_path = output_path  # Example processed data path
    print(f"Preprocessed data saved to: {processed_path}")
    return processed_path

# Step 2: Model Training
@component
def train_op(processed_data_path: str, epochs: int, learning_rate: float) -> str:
    # Simulating training
    print(f"Training model with data from: {processed_data_path} using {epochs} epochs and learning rate {learning_rate}")
    model_path = "C:/Users/hend8/Desktop/sentiment_model/trained_model.pth"  # Updated model path
    # Implement model training logic here
    print(f"Model trained and saved to: {model_path}")
    return model_path

# Step 3: Model Evaluation
@component
def evaluate_op(model_path: str, test_data_path: str) -> str:
    # Simulating model evaluation
    print(f"Evaluating model: {model_path} on test data: {test_data_path}")
    evaluation_metrics = "Accuracy: 95%"  # Example metric
    print(f"Evaluation result: {evaluation_metrics}")
    return evaluation_metrics

# Define the pipeline
@dsl.pipeline(
    name="Sentiment Analysis Pipeline",
    description="A simple pipeline for sentiment analysis."
)
def sentiment_pipeline(
    data_path: str = "C:/Users/hend8/Downloads/IMDB-Dataset.csv",  # Input dataset path
    output_path: str = "/tmp/processed_data.csv",  # Path where processed data will be stored
    epochs: int = 10,  # Number of epochs for model training
    learning_rate: float = 0.001,  # Learning rate for model training
    model_path: str = "C:/Users/hend8/Desktop/sentiment_model/trained_model.pth",  # Model save path
    test_data_path: str = "C:/Users/hend8/Desktop/Analyse des sentiments avec Kubeflow et Hugging Face/test_data.csv"  # Test data path for evaluation
):
    # Use keyword arguments to call components
    preprocess_task = preprocess_op(data_path=data_path, output_path=output_path)
    train_task = train_op(processed_data_path=preprocess_task.output, epochs=epochs, learning_rate=learning_rate)
    
    # Optional: Add evaluation step
    evaluate_task = evaluate_op(model_path=train_task.output, test_data_path=test_data_path)

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=sentiment_pipeline,
        package_path="sentiment_pipeline.yaml",  # Path where the pipeline YAML will be saved
    )
