import argparse
import cv2
import mediapipe as mp
import torch
import whisper
import os
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from dataclasses import dataclass
from scenedetect import detect, ContentDetector
import json
import yt_dlp
import shutil
from tqdm import tqdm
from time import time
from datetime import timedelta, datetime

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        logging.info(f"{func.__name__} took {time_str} (HH:MM:SS) to complete")
        return result
    return wrapper

@dataclass
class VideoScene:
    """A class representing a video scene with timing and content analysis information."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    hand_visibility: str  # 'none', 'one', 'both'
    transcription: str
    is_single_person: bool  # Now includes upper body check
    scene_type: str = "upper_body_only"
    
    def meets_criteria(self, hand_requirement: str) -> bool:
        """Check if the scene meets all required criteria."""
        if not self.is_single_person:
            return False
            
        if hand_requirement == 'none':
            return True
        elif hand_requirement == 'one':
            return self.hand_visibility in ['one', 'both']
        elif hand_requirement == 'both':
            return self.hand_visibility == 'both'
        return False

class YouTubeDownloader:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best[height<=720]',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
    def search_videos(self, query: str, max_results: int) -> List[str]:
        """Search YouTube for videos matching the query."""
        search_opts = {
            **self.ydl_opts,
            'extract_flat': True,
            'force_generic_extractor': True,
            'match_filter': lambda info: None if info.get('live_status') == 'is_live' else True
        }

        with yt_dlp.YoutubeDL(search_opts) as ydl:
            try:
                result = ydl.extract_info(
                    f"ytsearch{max_results}:{query}",
                    download=False
                )
                
                if 'entries' in result:
                    video_urls = [
                        f"https://www.youtube.com/watch?v={entry['id']}"
                        for entry in result['entries']
                        if entry.get('id')
                    ]
                    return video_urls[:max_results]
            except Exception as e:
                logging.error(f"Error searching videos: {str(e)}")
                return []

    def download_video(self, url: str, output_dir: str) -> Optional[str]:
        """Download a single video."""
        download_opts = {
            **self.ydl_opts,
            'outtmpl': os.path.join(output_dir, 'temp.%(ext)s')
        }

        with yt_dlp.YoutubeDL(download_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                return os.path.join(output_dir, f"temp.{info['ext']}")
            except Exception as e:
                logging.error(f"Error downloading {url}: {str(e)}")
                return None
            
class YouTubeAnalyzer:
    def __init__(self, 
                 hand_requirement: str = 'both',
                 min_scene_length: float = 1.0,
                 threshold: float = 30.0):
        """Initialize the YouTube video analyzer."""
        self.hand_requirement = hand_requirement
        self.min_scene_length = min_scene_length
        self.threshold = threshold
        self.setup_logging()
        self.setup_models()
        self.downloader = YouTubeDownloader()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_models(self):
        """Initialize all required models"""
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7
        )
        self.transcriber = whisper.load_model("base")

    def is_upper_body_only(self, pose_landmarks) -> bool:
        """Check if only the upper body is visible in the frame."""
        if not pose_landmarks:
            return False
            
        landmarks = pose_landmarks.landmark
        
        upper_landmarks = [
            landmarks[0],   # nose
            landmarks[11],  # left shoulder
            landmarks[12],  # right shoulder
            landmarks[13],  # left elbow
            landmarks[14],  # right elbow
            landmarks[15],  # left wrist
            landmarks[16],  # right wrist
        ]
        
        lower_landmarks = [
            landmarks[23],  # left hip
            landmarks[24],  # right hip
            landmarks[25],  # left knee
            landmarks[26],  # right knee
            landmarks[27],  # left ankle
            landmarks[28],  # right ankle
        ]
        
        # More lenient thresholds
        upper_visibility = sum(lm.visibility > 0.5 for lm in upper_landmarks) / len(upper_landmarks)
        lower_visibility = sum(lm.visibility > 0.4 for lm in lower_landmarks) / len(lower_landmarks)
        
        return upper_visibility > 0.6 and lower_visibility < 0.5

    def extract_video_segment(self, input_path: str, output_path: str, scene: VideoScene):
        """
        Extract video segment with simpler, more reliable settings.
        """
        try:
            duration = scene.end_time - scene.start_time
            
            # Simpler ffmpeg command with reasonable precision
            ffmpeg_cmd = (
                f'ffmpeg -y '
                f'-ss {scene.start_time:.2f} '  # Using .2f precision
                f'-i "{input_path}" '
                f'-t {duration:.2f} '           # Using .2f precision
                f'-c:v libx264 '
                f'-c:a aac '
                f'-strict experimental '
                f'-preset medium -crf 23 "{output_path}"'
            )
            
            self.logger.info(f"""
    Extracting video segment:
    - Start Time: {scene.start_time:.2f}s
    - End Time: {scene.end_time:.2f}s
    - Duration: {duration:.2f}s
            """)
            
            result = os.system(ffmpeg_cmd)
            
            if result != 0:
                raise Exception(f"FFmpeg command failed with return code {result}")
            
            # Verify the output file exists and is valid
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                actual_duration = frame_count / fps
                cap.release()
                
                self.logger.info(f"Output video duration: {actual_duration:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Error extracting video segment: {str(e)}")
            raise

    def transcribe_scene(self, video_path: str, scene: VideoScene) -> Dict:
        """Transcribe audio for a specific video scene with timing information."""
        try:
            audio = whisper.load_audio(video_path)
            audio_segment = audio[int(scene.start_time * 16000):int(scene.end_time * 16000)]
            
            result = self.transcriber.transcribe(audio_segment)
            
            # Process segments with timing
            processed_segments = []
            for segment in result["segments"]:
                segment_start = float(segment["start"]) + scene.start_time
                segment_end = float(segment["end"]) + scene.start_time
                
                processed_segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": "".join(segment["text"].split())  # Remove spaces
                })
            
            return {
                "text": "".join(result["text"].split()),  # Complete transcript without spaces
                "timing": processed_segments
            }
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            return {
                "text": "",
                "timing": []
            }
        
    @timing_decorator
    def analyze_scene(self, video_path: str, start_time: float, end_time: float) -> Tuple[VideoScene, Dict[str, float]]:
        """Analyze a specific scene and find exact timestamps where criteria are met."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, None

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            hand_counts = []
            person_counts = []
            upper_body_frames = []
            total_frames_analyzed = 0
            
            for _ in tqdm(range(start_frame, end_frame), desc="Analyzing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect hands
                hand_results = self.mp_hands.process(rgb_frame)
                num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
                hand_counts.append(num_hands)
                
                # Detect people and check upper body
                pose_results = self.mp_pose.process(rgb_frame)
                has_person = 1 if pose_results.pose_landmarks else 0
                person_counts.append(has_person)
                
                is_upper_body = self.is_upper_body_only(pose_results.pose_landmarks) if pose_results.pose_landmarks else False
                upper_body_frames.append(is_upper_body)
                
                total_frames_analyzed += 1
            
            cap.release()
            
            # Calculate percentages
            avg_hands = np.mean(hand_counts)
            person_percentage = np.mean(person_counts)
            upper_body_percentage = np.mean(upper_body_frames)

            # Determine Hand visibility
            if avg_hands < 0.5:
                hand_visibility = 'none'
            elif avg_hands < 1.5:
                hand_visibility = 'one'
            else:
                hand_visibility = 'both'

            # Scene is valid if:
                # - Single person is present in most frames (> 80%)
                # - Upper body is present in most frames (> 60%)
                
            is_valid = (person_percentage > 0.8 and upper_body_percentage > 0.6)

            if not is_valid:
                return None, None
            
            scene = VideoScene(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                hand_visibility=hand_visibility,
                transcription="",
                is_single_person=True
            )
            
            timing_info = {
                "original_video_start": start_time,
                "original_video_end": end_time,
                "total_video_duration": total_frames / fps
            }
            
            self.logger.info(f"""
            Scene Analysis Results:
                - Duration: {end_time - start_time:.2f}s
                - Person Detection Rate: {person_percentage*100:.1f}%
                - Upper Body Detection Rate: {upper_body_percentage*100:.1f}%
                - Hand Visibility: {hand_visibility}
            """)
            return scene, timing_info
            
        except Exception as e:
            self.logger.error(f"Error in analyze_scene: {str(e)}")
            return None, None

    @timing_decorator
    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect scene changes in video using PySceneDetect."""
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError("Could not open video file")

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()

            scene_list = detect(video_path, 
                              ContentDetector(threshold=self.threshold, 
                                            min_scene_len=int(self.min_scene_length * fps),
                                            luma_only = True))
            
            time_ranges = []
            for scene in scene_list:
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                
                start_time = float(start_frame) / fps
                end_time = float(end_frame) / fps
                
                if end_time - start_time >= self.min_scene_length:
                    time_ranges.append((start_time, end_time))
            
            self.logger.info(f"Detected {len(time_ranges)} scenes")
            return time_ranges
            
        except Exception as e:
            self.logger.error(f"Error in scene detection: {str(e)}")
            return []

    @timing_decorator
    def process_video(self, video_url: str, output_dir: str) -> List[Dict]:
        """Process a YouTube video and save segments meeting criteria."""
        try:
            video_id = video_url.split("watch?v=")[-1]
            
            # Download video
            self.logger.info(f"Downloading video: {video_url}")
            video_path = self.downloader.download_video(video_url, output_dir)
            if not video_path:
                return []

            # 1. First Detect scenes based on min_scene_length
            self.logger.info("Detecting scenes...")
            scenes_timestamps = self.detect_scenes(video_path)
            
            processed_scenes = []
            for i, (start_time, end_time) in enumerate(scenes_timestamps):
                self.logger.info(f"Analyzing scene {i+1}/{len(scenes_timestamps)}")
                
                try:
                    # 2. For each scene, check person/hand/upper body criteria
                    scene_result = self.analyze_scene(video_path, start_time, end_time)
                    if scene_result is None or scene_result[0] is None:
                        continue
                        
                    scene, timing_info = scene_result
                    
                    # 3. Check if scene meets criteria, process it
                    if scene.meets_criteria(self.hand_requirement):
                        # Transcribe audio
                        self.logger.info(f"Transcribing scene {i+1}")
                        transcription_data = self.transcribe_scene(video_path, scene)
                        
                        # Extract video segment
                        output_filename = f"{video_id}_scene_{i:03d}.mp4"
                        output_path = os.path.join(output_dir, output_filename)
                        self.extract_video_segment(video_path, output_path, scene)
                        
                        processed_scenes.append({
                            'video_file': output_filename,
                            'timing': {
                                'trim_start': float(scene.start_time),
                                'trim_end': float(scene.end_time),
                                'duration': float(scene.end_time - scene.start_time),
                                'original_video_timing': timing_info
                            },
                            'hand_visibility': scene.hand_visibility,
                            'transcription': transcription_data
                        })
                except Exception as e:
                    self.logger.error(f"Error processing scene {i+1}: {str(e)}")
                    continue
            
            # Cleanup
            try:
                os.remove(video_path)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary file: {str(e)}")
                
            return processed_scenes
                
        except Exception as e:
            self.logger.error(f"Error in process_video: {str(e)}")
            return []

def update_summary_json(output_dir: str, video_url: str, video_number: int, total_videos: int, 
                     scenes: List[Dict], processing_status: str = 'Completed', search_query: str = None):
    """Update JSON summary file with current video results in a NoSQL-friendly format."""
    try:
        summary_path = os.path.join(output_dir, 'summary.json')
        video_id = video_url.split("watch?v=")[-1]
        current_time = datetime.now().isoformat()
        
        # Initialize or load existing summary
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                try:
                    summary_data = json.load(f)
                except json.JSONDecodeError:
                    summary_data = {"videos": [], "metadata": {}}
        else:
            summary_data = {"videos": [], "metadata": {}}
        
        # Update metadata
        if video_number == 1 or not summary_data.get("metadata"):
            summary_data["metadata"] = {
                "processing_start": current_time,
                "total_videos": total_videos,
                "search_query": search_query,
                "last_updated": current_time
            }
        else:
            summary_data["metadata"]["last_updated"] = current_time
        
        # Create video document
        video_doc = {
            "_id": video_id,
            "url": video_url,
            "processing_number": video_number,
            "status": processing_status,
            "processed_at": current_time,
            "scenes": []
        }
        
        # Add scene information
        if scenes:
            for scene in scenes:
                scene_doc = {
                    "scene_id": os.path.splitext(scene['video_file'])[0],
                    "video_file": scene['video_file'],
                    "timing": scene['timing'],
                    "hand_visibility": scene['hand_visibility'],
                    "transcription": scene['transcription']
                }
                video_doc["scenes"].append(scene_doc)
        
        # Update videos list
        summary_data["videos"] = [v for v in summary_data["videos"] 
                                if v.get("_id") != video_id]
        summary_data["videos"].append(video_doc)
        
        # Write updated summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
            
    except Exception as e:
        logging.error(f"Error updating summary JSON: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='YouTube Video Analyzer')
    parser.add_argument('--hand-requirement', type=str, choices=['none', 'one', 'both'],
                      default='both', help='Hand visibility requirement')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory for processed videos')
    parser.add_argument('--search-query', type=str, required=True,
                      help='YouTube search query')
    parser.add_argument('--min-scene-length', type=float, default=1.0,
                      help='Minimum scene length in seconds')
    parser.add_argument('--threshold', type=float, default=30.0,
                      help='Scene detection threshold')
    parser.add_argument('--max-videos', type=int, default=10,
                      help='Maximum number of videos to process')
    parser.add_argument('--skip-videos', type=int, default=0,
                      help='Number of videos to skip before processing')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = YouTubeAnalyzer(
        hand_requirement=args.hand_requirement,
        min_scene_length=args.min_scene_length,
        threshold=args.threshold
    )
    
    # Search and download videos
    video_urls = analyzer.downloader.search_videos(args.search_query, args.skip_videos + args.max_videos)
    if not video_urls:
        logging.error("No videos found matching the search query")
        return
        
    # Skip specified number of videos
    video_urls = video_urls[args.skip_videos:]
    
    # Process videos
    all_scenes = []
    for i, url in enumerate(video_urls, 1):
        try:
            logging.info(f"Processing video {i}/{len(video_urls)}: {url}")
            
            # Create initial entry in summary
            update_summary_json(args.output_dir, url, i, len(video_urls), [], "Processing", args.search_query)
            
            # Process video
            scenes = analyzer.process_video(url, args.output_dir)
            
            # Update summary with results
            status = "Completed" if scenes else "No valid scenes found"
            update_summary_json(args.output_dir, url, i, len(video_urls), scenes, status, args.search_query)
            
            if scenes:
                logging.info(f"Successfully processed video. Found {len(scenes)} valid scenes.")
                all_scenes.extend(scenes)
            else:
                logging.warning(f"No valid scenes found in video {url}")
                
        except Exception as e:
            logging.error(f"Error processing {url}: {str(e)}")
            update_summary_json(args.output_dir, url, i, len(video_urls), [], f"Error: {str(e)}", args.search_query)
            continue

    if not all_scenes:
        logging.warning("No valid scenes found in any of the processed videos")

if __name__ == "__main__":
    main()


# python youtube_analyzer.py \
#     --hand-requirement both \
#     --output-dir Youtube_crawled \
#     --search-query "tutorial videos" \
#     --min-scene-length 5.0 \
#     --threshold 30.0 \
#     --max-videos 5 \
#     --skip-videos 5