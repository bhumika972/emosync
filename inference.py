
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from {__name__} import MultimodalModel # Assuming MultimodalModel is defined in a global scope if not imported
import os

# Assuming the MultimodalModel class definition is also available or can be imported
# For deployment, it's best to include the class definition directly or import it if packaged

class MultimodalModel(torch.nn.Module):
    def __init__(self, text_input_dim, visual_input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(MultimodalModel, self).__init__()

        self.text_lstm = torch.nn.LSTM(
            input_size=text_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.visual_lstm = torch.nn.LSTM(
            input_size=visual_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text_features, visual_features):
        _, (text_hidden, _) = self.text_lstm(text_features)
        text_hidden = text_hidden.squeeze(0)

        _, (visual_hidden, _) = self.visual_lstm(visual_features)
        visual_hidden = visual_hidden.squeeze(0)

        combined_features = torch.cat((text_hidden, visual_hidden), dim=1)
        combined_features = self.dropout(combined_features)
        output = self.fc(combined_features)

        return output


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters used during training
    text_input_dim = 384 # As defined during training
    visual_input_dim = 35 # As defined during training
    hidden_dim = 128
    output_dim = 7
    num_layers = 1
    dropout_rate = 0.5

    model = MultimodalModel(text_input_dim, visual_input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
    model.load_state_dict(torch.load('multimodal_model.pt', map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    text_embedding_model = SentenceTransformer('sentence_transformer')

    return model, text_embedding_model, device

model, text_embedding_model, device = load_model()

def predict(text_input, visual_features_input):
    # Ensure text_input is a string
    if not isinstance(text_input, str):
        return {"error": "Text input must be a string."}

    # Generate text embeddings
    text_embeddings = text_embedding_model.encode([text_input], convert_to_tensor=True, device=device)

    # Convert visual features (example: a string of comma-separated floats) to tensor
    try:
        # For a real deployment, visual input would be more structured, e.g., an uploaded video processed into features
        # Here, we'll assume a dummy visual feature array for demonstration or expect a pre-processed input
        # For a simple demo, let's just create a dummy visual feature tensor if none is provided
        if visual_features_input:
            visual_features = np.array([float(x) for x in visual_features_input.split(',')], dtype=np.float32).reshape(1, -1, 35) # Reshape as (batch_size, seq_len, feature_dim)
        else:
            # Create a dummy visual feature (e.g., zeros) if no input, or adjust to your data's expected input
            visual_features = np.zeros((1, 1, 35), dtype=np.float32) # Single frame, 35 dimensions

        visual_features = torch.tensor(visual_features, dtype=torch.float32).to(device)
    except Exception as e:
        # Corrected f-string syntax here
        return {"error": f"Invalid visual features format: {e}"}

    # Pad text embeddings to match visual features sequence length or a reasonable default
    # For deployment, a more robust padding/alignment strategy is usually needed.
    # For this simple example, we'll make a basic assumption.
    # Let's assume a single text embedding represents the entire text
    # And visual features might be sequence-based.
    # If text is single embedding, need to make it (batch, 1, embedding_dim)

    # Adjusting for model's expected input (batch_first=True, (batch_size, sequence_length, feature_dim))
    text_input_tensor = text_embeddings.unsqueeze(0) if text_embeddings.ndim == 1 else text_embeddings.unsqueeze(0) # (1, 1, 384)

    with torch.no_grad():
        outputs = model(text_input_tensor, visual_features)

    # Convert predictions to numpy array
    predictions = outputs.cpu().numpy().tolist()

    # Assuming 7 sentiment dimensions: (anger, disgust, fear, happiness, sadness, surprise, neutral)
    sentiment_labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    result = {label: score for label, score in zip(sentiment_labels, predictions[0])}

    return result
