import os
import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from facenet_pytorch import MTCNN
import torch
from PIL import Image

def extract_frames_and_audio(video_path, output_frame_dir, output_audio_path):
    # Ensure directories exist
    os.makedirs(output_frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    # 1. Extract Audio using MoviePy
    try:
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le', verbose=False, logger=None)
            print(f"Audio extracted: {output_audio_path}")
        else:
            print(f"No audio found in {video_path}")
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        
    # 2. Extract Frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, image = cap.read()
    while success:
        # Save every 10th frame to reduce data size
        if count % 10 == 0:
            frame_path = os.path.join(output_frame_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, image)
        success, image = cap.read()
        count += 1
    cap.release()
    print(f"Frames extracted to {output_frame_dir}")

def detect_and_crop_faces(frame_dir, output_crop_dir):
    os.makedirs(output_crop_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    
    for filename in os.listdir(frame_dir):
        if not filename.endswith(".jpg"):
            continue
        
        frame_path = os.path.join(frame_dir, filename)
        img = Image.open(frame_path)
        
        # Detect faces
        boxes, _ = mtcnn.detect(img)
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                box = [int(b) for b in box]
                cropped_face = img.crop(box)
                crop_path = os.path.join(output_crop_dir, f"{os.path.splitext(filename)[0]}_face{i}.jpg")
                cropped_face.save(crop_path)

def extract_audio_features(audio_path):
    if not os.path.exists(audio_path):
        return None
        
    y, sr = librosa.load(audio_path, sr=None)
    # Extract Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

if __name__ == "__main__":
    print("Preprocessing script ready.")
    # Example usage:
    # video_file = "data/raw_videos/real/sample.mp4"
    # base_name = os.path.basename(video_file).split('.')[0]
    # extract_frames_and_audio(video_file, f"extracted_frames/{base_name}", f"audio_data/{base_name}.wav")
    # detect_and_crop_faces(f"extracted_frames/{base_name}", f"face_crops/{base_name}")
