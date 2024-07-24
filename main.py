import subprocess
import os
import time
from PIL import Image


def get_m3u8_url(twitch_channel):
    """Get the m3u8 URL using streamlink."""
    command = f"streamlink --stream-url https://www.twitch.tv/{twitch_channel} 1080p"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def capture_frame(m3u8_url, output_dir, interval=5, max_age=300):
    """Capture a frame from the stream every `interval` seconds using ffmpeg and delete old frames."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 1
    while True:
        output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        command = f'ffmpeg -y -i "{m3u8_url}" -vframes 1 "{output_file}"'
        subprocess.run(command, shell=True)
        process_frame(output_file)
        delete_old_frames(output_dir, max_age)
        frame_count += 1
        time.sleep(interval)


def delete_old_frames(directory, max_age):
    """Delete frames older than `max_age` seconds."""
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age:
                try:
                    os.remove(file_path)
                    print(f"Deleted old frame: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


def capture_frame_at_timestamp(m3u8_file, output_file, timestamp):
    """Capture a frame from the video at the specified timestamp using ffmpeg."""
    command = f'ffmpeg -y -ss {timestamp} -i "{m3u8_file}" -vframes 1 -q:v 0 -vsync vfr "{output_file}"'
    subprocess.run(command, shell=True)


def process_frame(file_path):
    try:
        img = Image.open(file_path)
        # Add your image processing code here
        print(f"Processing {file_path}")
        img.show()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    twitch_channel = "scrabble"
    output_dir = "captured_frames"
    interval = 5  # seconds

    m3u8_url = get_m3u8_url(twitch_channel)
    if m3u8_url:
        print(f"Starting to capture frames from {m3u8_url}")
        capture_frame(m3u8_url, output_dir, interval)
    else:
        print("Failed to get m3u8 URL")
