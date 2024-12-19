import sys
import os
import replicate
import requests
import json
from openai import AzureOpenAI


def transcribe_audio(filepath):
    if 'REPLICATE_API_KEY' in os.environ:
        model = replicate.models.get("cjwbw/whisper")
        return model.predict(
            audio=open(filepath, "rb"),
            model="large",
            translate=True
        )
    elif 'AZURE_API_KEY' in os.environ and 'AZURE_API_VERSION' in os.environ and 'AZURE_API_BASE' in os.environ:
        return transcribe_with_azure_openai_whisper(filepath)
    else:
        raise Exception("No API key found for either Replicate or Azure")

def transcribe_with_azure_openai_whisper(filepath):
    apikey = os.getenv('AZURE_API_KEY')
    apiversion = os.getenv('AZURE_API_VERSION')
    endpoint = os.getenv('AZURE_API_BASE')

    client = AzureOpenAI(
        api_key = apikey,
        api_version = apiversion,
        azure_endpoint = endpoint
    )

    result = client.audio.transcriptions.create(
        file=open(filepath, "rb"),
        model='whisper'
    )

    transcription = {"text": result.text}
    if hasattr(result, 'duration'):
        transcription["duration"] = result.duration
    if hasattr(result, 'language'):
        transcription["language"] = result.language
    return json.dumps(transcription)

if __name__ == "__main__":
    audio_path = sys.argv[-2]
    output_path = sys.argv[-1]
    if not audio_path.endswith(".mp3"):
        print("Please provide an mp3 file")
        sys.exit(1)
    if not output_path.endswith(".json"):
        print("Please provide a .json output file")
        sys.exit(1)
    output = transcribe_audio(audio_path)
    open(output_path, "w").write(json.dumps(output, indent=2) + "\n")
