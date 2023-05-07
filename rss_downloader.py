import os
import re
import requests
import xml.etree.ElementTree as ET
import openai
from pydub import AudioSegment
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

#Perform search on iTunes API to get podcast feed URL
def get_feed_url(apple_id):
    api_url = f"https://itunes.apple.com/lookup?id={apple_id}&entity=podcast"
    response = requests.get(api_url)
    data = response.json()

    if data['resultCount'] > 0:
        return data['results'][0]['feedUrl']
    else:
        return None

#Download podcast episode from feed URL
def download_episode(feed_url, save_directory, episode_index=0, download=True):
    response = requests.get(feed_url)
    root = ET.fromstring(response.content)

    items = root.findall('.//item')

    if not items:
        print("No episodes found in the RSS feed.")
        return

    if episode_index >= len(items):
        print("Episode index out of range.")
        return

    episode = items[episode_index]
    episode_title = episode.find('./title').text
    episode_url = episode.find('./enclosure').attrib['url']
    
    if not download:
        return None, episode_title
    
    print(f"Downloading episode: {episode_title}")

    local_filename = os.path.join(save_directory, f"{episode_title}.mp3")

    with requests.get(episode_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), total=total_size // 8192, unit='KB', unit_scale=True):
                f.write(chunk)

    print(f"Download complete: {local_filename}")
    return local_filename, episode_title

#Split podcast episode into segments
def split_audio(mp3_path, segment_duration_minutes=20):
    audio = AudioSegment.from_mp3(mp3_path)
    segment_duration_ms = segment_duration_minutes * 60 * 1000

    segments = []
    audio_length_ms = len(audio)
    current_position = 0

    while current_position < audio_length_ms:
        print(f"Splitting audio: {current_position / 1000:.1f}/{audio_length_ms / 1000:.1f} seconds")
        end_position = current_position + segment_duration_ms
        segment = audio[current_position:end_position]
        segments.append(segment)
        current_position = end_position

    return segments

#Transcribe audio segments using OpenAI API
def transcribe_audio_segments(segments):
    transcripts = []

    for i, segment in enumerate(segments):
        print(f"Transcribing segment {i + 1}/{len(segments)}")
        segment.export("temp_segment.mp3", format="mp3")
        with open("temp_segment.mp3", "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        transcripts.append(response['text'])
        os.remove("temp_segment.mp3")

    return transcripts

#Save transcripts to file
def save_transcripts(transcripts, save_directory, filename):
    transcript_path = os.path.join(save_directory, filename)

    with open(transcript_path, "w") as f:
        for transcript in transcripts:
            f.write(transcript)
            f.write("\n\n")

def split_text_into_chunks(text, max_tokens=4090):
    chunks = []
    while len(text) > 0:
        tokens = text[:max_tokens]
        last_space = tokens.rfind(' ')
        if last_space == -1:
            last_space = max_tokens
        chunks.append(text[:last_space].strip())
        text = text[last_space:].strip()
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def summarize_chunk(chunk):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text.    "},
            {"role": "user", "content": f"Summarize the transcript into bullet points. The bullet points should be informative detailed description of key points, topics and insights that the guests talked about.Use only content from the transcript. Do not add any additional information: {chunk}"}
        ],
        temperature=0.5,
    )
    summary = response.choices[0].message.content.strip()
    return summary

def summarize_large_text(text):
    max_tokens = 4090  # Reserve tokens for instructions and conversation context
    chunks = split_text_into_chunks(text, max_tokens)
    summaries = []
    
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_chunk(chunk)
        summaries.append(summary)
    
    print("Summaries aggregated.")
    return ' '.join(summaries)

def generate_final_summary(summary_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Imagine you have just completed recording an amazing podcast episode, and as the host of the podcast, you are now looking to create a captivating first-person narrative summary of your episode to share with your audience on social media. Weave the key takeaways and highlights of your episode into an engaging, informative, and succinct summary. Your goal is to entice your audience to listen to the full episode. Think about how you can turn the most important parts of your episode into a compelling and informative story that captures the essence of your podcast. Remember, the goal of your narrative summary is to get your audience excited about your podcast and make them eager to tune in."
            },
            {
                "role": "user",
                "content": f"So, let your creativity shine and craft a first-person narrative summary that is as captivating as your episode itself! Please Also follow the rules below: - Give more content, less fluff, and no need for buzz words. - Ensure to give lots of details. - The summary should be around the topics. - The length of the summary should be at least 800 words: {summary_text}"
            },
        ],
        temperature=0.6,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]

    apple_podcast_url = os.environ["APPLE_PODCAST_URL"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(current_dir, "podsum_rss")
    os.makedirs(save_directory, exist_ok=True)

    # Extract the Apple Podcast ID from the URL
    apple_id = re.search(r'id(\d+)', apple_podcast_url).group(1)

    # Get the RSS feed URL
    feed_url = get_feed_url(apple_id)

    # Set the episode index you want to download. 0 for the latest episode, 1 for the previous one, and so on.
    episode_index = int(os.environ["EPISODE_INDEX"])

    # Download the episode
    _, episode_title = download_episode(feed_url, save_directory, episode_index, download=False)

    # Define the file paths
    mp3_path = os.path.join(save_directory, f"{episode_title}.mp3")
    transcript_path = os.path.join(save_directory, f"transcript_{episode_title}.txt")
    topics_path = os.path.join(save_directory, f"topics_{episode_title}.txt")
    final_summary_path = os.path.join(save_directory, f"final_summary_{episode_title}.txt")

    # Download the episode
    if not os.path.exists(mp3_path):
        local_filename, episode_title = download_episode(feed_url, save_directory, episode_index, download=True)
    else:
        print(f"MP3 file already exists: {mp3_path}")
        local_filename = mp3_path

    # Split the episode into 20-minute segments
    segments = split_audio(local_filename)

    # Transcribe the audio segments
    if not os.path.exists(transcript_path):
        transcripts = transcribe_audio_segments(segments)
        # Save the transcripts
        save_transcripts(transcripts, save_directory, f"transcript_{episode_title}.txt")
        print("Transcripts saved.")
    else:
        print(f"Transcript file already exists: {transcript_path}")
        with open(transcript_path, "r") as f:
            transcripts = f.read().split("\n\n")

    # Concatenate the transcripts
    full_transcript = "\n\n".join(transcripts)

    # Summarize the transcript
    if not os.path.exists(topics_path):
        summarized_bullet_points = summarize_large_text(full_transcript)
        print("Bullet Points Created.")

        # Save the summarized text
        save_transcripts([summarized_bullet_points], save_directory, f"topics_{episode_title}.txt")
        print("Bullet Points saved.")
    else:
        print(f"Bullet Points file already exists: {topics_path}")

    if not os.path.exists(final_summary_path):
        with open(topics_path, "r") as f:
            summary_text = f.read()
        final_summary = generate_final_summary(summary_text)
        print("Final summary generated.")

        with open(final_summary_path, "w") as outfile:
            outfile.write(final_summary)
        print("Final summary saved.")
    else:
        print(f"Final summary file already exists: {final_summary_path}")
