from moviepy import VideoFileClip

video_path = "input/Otakudesu.net_KJN--01_720p.mp4"
audio_path = "output/audio.wav"

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec="pcm_s16le")
video.close()

print(f"Saved audio to: {audio_path}")