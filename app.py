
import gradio as gr
from inference import predict

def multimodal_sentiment_analysis(text_input, visual_features_input):
    result = predict(text_input, visual_features_input)
    if "error" in result:
        return result["error"]
    else:
        # Format for better display in Gradio
        formatted_output = ""
        for key, value in result.items():
            formatted_output += f"**{key}**: {value:.4f}
"
        return formatted_output


if __name__ == '__main__':
    # Example Gradio interface
    iface = gr.Interface(
        fn=multimodal_sentiment_analysis,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter text here...", label="Text Input"),
            gr.Textbox(lines=2, placeholder="Enter comma-separated visual features (e.g., 0.1,0.2,... for 35 values)", label="Visual Features (Dummy Input)")
        ],
        outputs="text",
        title="Multimodal Sentiment Analysis (CMU-MOSEI)",
        description="Enter text and (dummy) visual features to get sentiment predictions across 7 dimensions."
    )
    iface.launch()
