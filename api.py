from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import json
import random
import os
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from moviepy.editor import AudioFileClip, VideoFileClip
import mediapipe as mp
import cv2
import io
from io import BytesIO
import aiohttp
import aiofiles
import requests
import traceback
import warnings
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from src.utils.util import save_videos_grid
from diffusers import AutoencoderKL, DDIMScheduler
from preprocess_api import VideoPreprocessor
import logging
from logging.handlers import RotatingFileHandler
import sys

warnings.simplefilter("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
handler = RotatingFileHandler('echomimic.log', maxBytes=100000, backupCount=0)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.addHandler(handler)

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode = True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def detect_hands(self, image):
        """
        Detect hands in the image.
        
        Args:
            image: numpy array of the image in BGR format
        
        Returns:
            bool: True if hands are detected, False otherwise
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        # Return True if any hands are detected
        return results.multi_hand_landmarks is not None
    
    def __del__(self):
        self.hands.close()

class ReferenceImageManager:
    def __init__(self, static_ref_path: Path):
        self.static_ref_path = static_ref_path
        self.hand_detector = HandDetector()
    
    async def process_reference_image(self, image_file: UploadFile) -> dict:
        """
        Process uploaded reference image and verify it contains hands.

        Args:
            image_file: UploadFile object containing the reference image

        Returns:
            dict: Contains 'path' (Path object) and 'hands_detected' (bool)
        """
        try:
            # Read an validate image
            contents = await image_file.read()
            image = Image.open(io.BytesIO(contents))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to numpy array for hand detection
            image_np = np.array(image)

            # Detect hands
            has_hands = self.hand_detector.detect_hands(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if not has_hands:
                logger.info("No hands detected in uploaded image, using static reference image")
                return {
                    'path': self.static_ref_path,
                    'hands_detected': False
                }
            
            # Save processed image to temporary file
            temp_path = Path("temp_reference.png")
            image.save(temp_path)
            return {
                'path': temp_path,
                'hands_detected': True
            }
        except Exception as e:
            logger.error(f"Error processing reference image: {str(e)}")
            return {
                'path': self.static_ref_path, 
                'hands_detected': False
            }
class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
    def crop_video_at_shoulders(self, video_path: str, output_path: str) ->bool:
        """
        Crop video to portrait format focusing on face and upper body.
        Returns True if successful, False otherwise.
        """
        try:
            # Load the vide
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Error: Could not open video file")
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Calculate zoop crop dimensions
            # We want to zoom in by about 40%
            zoom_factor = 1.4

            # Calculate new dimensions after zoom
            crop_height = int(height / zoom_factor)
            crop_width = int(width / zoom_factor)

            # Calculate crop coordinates to center the frame
            # We want to crop slightly higher than centre to focus more on face/upper body
            x_start = (width - crop_width) // 2
            # Move the vertical crop up a bit to better frame the face
            y_start = (height - crop_height) // 2 - int(crop_height * 0.1)  # Shift up by 10% of crop height

            # Ensure we don't go out of bounds
            y_start = max(0, y_start)
            y_end = min(height, y_start + crop_height)
            x_end = min(width, x_start + crop_width)

            # Create temporary output file
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width,height)) # Keep original dimensions

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop and zoom
                cropped = frame[y_start:y_end, x_start:x_end]
                # Resize back to original dimensions to create zoom effect
                zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
                out.write(zoomed)
            
            cap.release()
            out.release()

            # Convert to mp4 with h264 codec
            try:
                clip = VideoFileClip(temp_output)
                clip.write_videofile(output_path, codec='libx264')
                clip.close()

                # Clean up temporary file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                
                return True
            except Exception as e:
                logger.error(f"Error in final video conversion: {str(e)}")
                return False
        
        except Exception as e:
            logger.error(f"Error in video cropping: {str(e)}")
            return False
            
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()

class InferenceManager:
    def __init__(self, config_path: str = "./configs/prompts/infer_acc.yaml"):
        logger.info("Initializing InferenceManager")
        self.config = OmegaConf.load(config_path)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
        
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Set up weight dtype based on config
        self.weight_dtype = torch.float16 if self.config.weight_dtype == "fp16" else torch.float32
        
        # Load inference config
        inference_config = OmegaConf.load(self.config.inference_config)
        
        # Add static reference image path
        self.static_ref_path = Path('EMTD_dataset/ref_imgs_by_FLUX/man/0001.png')
        if not self.static_ref_path.exists():
            raise RuntimeError(f"Static reference image not found at {self.static_ref_path}")
        
        # Initialize all models directly
        logger.info("Initializing VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_vae_path,
        ).to("cuda", dtype=self.weight_dtype)

        logger.info("Initializing Reference UNet...")
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=self.weight_dtype, device="cuda")
        self.reference_unet.load_state_dict(
            torch.load(self.config.reference_unet_path, map_location="cpu"),
        )

        logger.info("Initializing Denoising UNet...")
        if os.path.exists(self.config.motion_module_path):
            self.denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
                self.config.pretrained_base_model_path,
                self.config.motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=inference_config.unet_additional_kwargs,
            ).to(dtype=self.weight_dtype, device="cuda")
        else:
            self.denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
                self.config.pretrained_base_model_path,
                "",
                subfolder="unet",
                unet_additional_kwargs={
                    "use_motion_module": False,
                    "unet_use_temporal_attention": False,
                    "cross_attention_dim": inference_config.unet_additional_kwargs.cross_attention_dim
                }
            ).to(dtype=self.weight_dtype, device="cuda")
        
        self.denoising_unet.load_state_dict(
            torch.load(self.config.denoising_unet_path, map_location="cpu"),
            strict=False
        )

        logger.info("Initializing Pose Encoder...")
        self.pose_net = PoseEncoder(
            320, 
            conditioning_channels=3, 
            block_out_channels=(16, 32, 96, 256)
        ).to(dtype=self.weight_dtype, device="cuda")
        self.pose_net.load_state_dict(
            torch.load(self.config.pose_encoder_path)
        )

        logger.info("Initializing Audio Processor...")
        self.audio_processor = load_audio_model(
            model_path=self.config.audio_model_path,
            device="cuda"
        )

        logger.info("Initializing Pipeline...")
        scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
        self.pipe = EchoMimicV2Pipeline(
            vae=self.vae,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            audio_guider=self.audio_processor,
            pose_encoder=self.pose_net,
            scheduler=scheduler,
        ).to("cuda", dtype=self.weight_dtype)

        self.ref_image_manager = ReferenceImageManager(self.static_ref_path)
        self.video_processor = VideoProcessor()

        # Set up storage directories
        self.storage_root = Path("storage")
        self.storage_root.mkdir(exist_ok=True)
        
        # Directory for preprocessed data
        self.preprocess_dir = self.storage_root / "preprocessed"
        self.preprocess_dir.mkdir(exist_ok=True)
        
        # Directory for model outputs
        self.outputs_dir = self.storage_root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Directory for final videos (public downloads)
        self.downloads_dir = Path("permanent_storage")
        self.downloads_dir.mkdir(exist_ok=True)
        
        # Directory for cropped videos
        self.cropped_video_dir = Path("permanent_storage/cropped_videos")
        self.cropped_video_dir.mkdir(parents=True, exist_ok=True)
        # Set up cache directory and index
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self._load_or_create_cache_index()
        
        logger.info("InferenceManager Initialization complete!")

    def _load_or_create_cache_index(self):
        """Load existing cache index or create new one"""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}
            self._save_cache_index()

    def _save_cache_index(self):
        """Save cache index to disk"""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)

    def _get_cache_key(self, video_path: str, target_duration: float, fps: int) -> str:
        """Generate unique cache key for video processing parameters"""
        video_name = Path(video_path).stem
        return f"{video_name}_d{target_duration:.1f}_fps{fps}"

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if processed data exists in cache"""
        if cache_key not in self.cache_index:
            return None
            
        cache_entry = self.cache_index[cache_key]
        cache_path = Path(cache_entry['cache_path'])
        
        if not cache_path.exists():
            del self.cache_index[cache_key]
            self._save_cache_index()
            return None
            
        return cache_entry

    def _save_to_cache(self, cache_key: str, preprocess_results: Dict, video_path: str, 
                      target_duration: float, fps: int):
        """Save processed video data to cache"""
        cache_entry_dir = self.cache_dir / cache_key
        cache_entry_dir.mkdir(exist_ok=True)
        
        cached_pose_dir = cache_entry_dir / "pose"
        if os.path.exists(preprocess_results['pose_dir']):
            if os.path.exists(cached_pose_dir):
                shutil.rmtree(cached_pose_dir)
            shutil.copytree(preprocess_results['pose_dir'], cached_pose_dir)
        
        self.cache_index[cache_key] = {
            'cache_path': str(cache_entry_dir),
            'original_video': video_path,
            'target_duration': target_duration,
            'fps': fps,
            'num_frames': preprocess_results['num_frames'],
            'params': preprocess_results['params'],
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache_index()

    def get_secure_filename(self, filename: str) -> str:
        """Convert filename to a secure version and ensure it doesn't start with a hypen"""
        # Remove or replace problematic characters
        secure_name =  "".join(c for c in filename if c.isalnum() or c in "._-").strip()

        # if filename starts with a hypen, prepend 'video
        if secure_name.startswith('-'):
            secure_name = f'video{secure_name}'
        return secure_name

    def get_session_dirs(self, session_id: str) -> Dict[str, Path]:
        """Create and return session-specific directories"""
        session_preprocess = self.preprocess_dir / session_id
        session_output = self.outputs_dir / session_id
        
        session_preprocess.mkdir(exist_ok=True)
        session_output.mkdir(exist_ok=True)
        
        return {
            'preprocess': session_preprocess,
            'output': session_output
        }

    def get_random_video(self, json_path: str, target_duration: float) -> Dict:
        """Get a random video scene from summary.json with duration close to target"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        valid_videos = [
            (video, scene) 
            for video in data['videos'] 
            if video.get('status') == 'Completed' and video.get('scenes')
            for scene in video['scenes']
            if float(scene['timing']['duration']) >= target_duration
        ]
        
        if not valid_videos:
            raise ValueError(f"No valid video scenes found with duration >= {target_duration}s")
            
        video, scene = random.choice(valid_videos)
        
        return {
            'video_file': scene['video_file'],
            'duration': scene['timing']['duration']
        }
    
    async def process_inputs(self, 
                           audio_file: UploadFile,
                           json_path: str,
                           reference_image: UploadFile = None,
                           fps: int = 15,
                           steps: int = 6,
                           cfg: float = 1.0,
                           seed: Optional[int] = None) -> Dict[str, str]:
        """Process inputs and return video information"""
        
        # Generate unique session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"
        session_dirs = self.get_session_dirs(session_id)
        logger.info(f"Starting process_inputs for session {session_id}")
        
        # Initialize variables that need cleanup
        audio_clip = None
        video_clip = None
        response_data = {}

        
        try:
            logger.info(f"Processing session {session_id}")

            # Process reference image if provided, otherwise use static reference
            if reference_image:
                original_ref_bytes = await reference_image.read()
                # Reset file position for subsequent processing
                await reference_image.seek(0)
                logger.info("Processing user-provided reference image...")
                ref_result = await self.ref_image_manager.process_reference_image(reference_image)
                reference_path = ref_result['path']
                logger.info(f"Hands Detected: {ref_result['hands_detected']}")
                hands_detected = ref_result['hands_detected']

            else:
                logger.info("Using static reference image...")
                reference_path = self.static_ref_path
                hands_detected = False
            
            # Copy static reference image to session directory
            session_ref_path = session_dirs['preprocess'] / "reference.png"
            shutil.copy2(reference_path, session_ref_path)
            
            # Read reference image as bytes
            with open(session_ref_path, 'rb') as f:
                reference_bytes = f.read()
                
            # Save and validate audio
            audio_path = session_dirs['preprocess'] / "audio.wav"
            with audio_path.open("wb") as f:
                shutil.copyfileobj(audio_file.file, f)
            
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            # Check audio duration
            audio_clip = AudioFileClip(str(audio_path))
            audio_duration = audio_clip.duration

            logger.info(f"Audio file: {audio_file.filename}")
            logger.info(f"Audio duration: {audio_duration:.3f}s")
            logger.info(f"Audio sample rate: {audio_clip.fps}Hz")
            logger.info(f"Audio channels: {1 if audio_clip.nchannels == 1 else 'Stereos'}")
            
            MAX_AUDIO_DURATION = 25  # Maximum allowed duration in seconds
            if audio_duration > MAX_AUDIO_DURATION:
                audio_clip.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration ({audio_duration:.1f}s) exceeds maximum allowed duration of {MAX_AUDIO_DURATION}s"
                )
            
            logger.info(f"Audio duration: {audio_duration}s")
            
            # Get random video matching audio duration
            video_info = self.get_random_video(json_path, audio_duration)
            video_path = Path(json_path).parent / video_info['video_file']
            
            # Check cache for preprocessed data
            cache_key = self._get_cache_key(str(video_path), audio_duration, fps)
            cache_entry = self._check_cache(cache_key)
            
            if cache_entry:
                logger.info(f"Using cached preprocessing data for video: {video_info['video_file']}")
                # For cached data, keep models on GPU and just copy the pose data
                cached_pose_dir = Path(cache_entry['cache_path']) / "pose"
                session_pose_dir = session_dirs['preprocess'] / "pose"
                if os.path.exists(session_pose_dir):
                    shutil.rmtree(session_pose_dir)
                shutil.copytree(cached_pose_dir, session_pose_dir)
                
                preprocess_results = {
                    'pose_dir': str(session_pose_dir),
                    'num_frames': cache_entry['num_frames'],
                    'params': cache_entry['params']
                }
            else:
                logger.info(f"Processing new video: {video_info['video_file']}")
                # Process video and cache results
                preprocessor = None
                try:
                    logger.info("Moving inference models to CPU for preprocessing...")
                    self.pipe.to('cpu')
                    torch.cuda.empty_cache()

                    preprocessor = VideoPreprocessor(max_size=768).to("cuda")
                    preprocess_results = preprocessor.process_video(
                        str(video_path),
                        str(session_dirs['preprocess']),
                        target_fps=fps,
                        max_duration=audio_duration
                    )
                    
                    self._save_to_cache(cache_key, preprocess_results, str(video_path), 
                                      audio_duration, fps)
                    
                finally:
                    if preprocessor is not None:
                        logger.info("Cleaning up preprocessor GPU Memory...")
                        preprocessor.to('cpu')
                        if hasattr(preprocessor, 'release_memory'):
                            preprocessor.release_memory()
                        torch.cuda.empty_cache()
                    
                    logger.info("Moving inference models back to GPU...")
                    self.pipe.to("cuda", dtype=self.weight_dtype)
                    torch.cuda.empty_cache()

            
            # Calculate frame limits based on audio duration
            num_frames = int(audio_duration * fps)
            
            logger.info(f"Loading {num_frames} poses...")
            # Load poses
            pose_list = []
            for index in range(num_frames):
                tgt_musk = np.zeros((768, 768, 3)).astype('uint8')
                tgt_musk_path = os.path.join(preprocess_results['pose_dir'], f"{index}.npy")
                
                detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
                imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
                im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
                im = np.transpose(np.array(im), (1, 2, 0))
                tgt_musk[rb:re, cb:ce, :] = im

                tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
                pose_tensor = torch.Tensor(np.array(tgt_musk_pil)).to(
                    dtype=self.weight_dtype, 
                    device="cuda"
                ).permute(2,0,1) / 255.0
                pose_list.append(pose_tensor)

            poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
            
            # Set up generation parameters
            if seed is None:
                seed = random.randint(0, 1000000)
            generator = torch.manual_seed(seed)
            
            logger.info(f"Generating video with parameter - FPS: {fps}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
            # Generate video using the static reference image
            static_ref_img = Image.open(session_ref_path).convert("RGB")
            
            video = self.pipe(
                static_ref_img,
                str(audio_path),
                poses_tensor,
                768, 768, num_frames,
                steps, cfg,
                generator=generator,
                audio_sample_rate=16000,
                context_frames=12,
                fps=fps,
                context_overlap=3,
                start_idx=0
            ).videos

            logger.info("Saving outputs...")
            # Save intermediate output without audio
            raw_output = session_dirs['output'] / "raw_output.mp4"
            save_videos_grid(
                video,
                str(raw_output),
                n_rows=1,
                fps=fps
            )

            # Get base names for output files
            video_name = Path(video_info['video_file']).stem
            if video_name.startswith('-'):
                video_name = f"video{video_name}"

            static_ref_name = self.static_ref_path.stem
            audio_name = Path(audio_file.filename).stem
            
            # Create output filename
            output_filename = f"{video_name}_{static_ref_name}_{audio_name}.mp4"
            output_filename = self.get_secure_filename(output_filename)
            
            # Add audio and save final version
            video_clip = VideoFileClip(str(raw_output))
            video_clip = video_clip.set_audio(audio_clip)
            
            final_output = self.downloads_dir / output_filename
            video_clip.write_videofile(str(final_output), codec="libx264", audio_codec="aac")

            # Create shoulder-cropped version
            cropped_filename = f"cropped_{output_filename}"
            cropped_video_path = self.cropped_video_dir / cropped_filename

            # Crop the video at shoulders
            cropped_successful = self.video_processor.crop_video_at_shoulders(str(final_output), str(cropped_video_path))

            # Read the video files as bytes
            with open(final_output, 'rb') as f:
                source_video_bytes = f.read()
            
            if cropped_successful:
                with open(cropped_video_path, 'rb') as f:
                    driving_video_bytes = f.read()
            else:
                driving_video_bytes = source_video_bytes # Fallback to original if cropping failed
                                                                              
            # Update response based on hand detection
            if hands_detected:
                response_data.update({
                    'source_video_data': source_video_bytes,
                    'driving_video_data': driving_video_bytes,
                    'source_content_type': 'video/mp4',
                    'using_custom_reference': True
                })
            else:

                                
                response_data.update({
                    'source_data': original_ref_bytes,
                    'driving_video_data': driving_video_bytes,
                    'source_content_type': 'image/png',
                    'using_custom_reference': False,
                    'audio_data': audio_bytes,
                    'audio_content_type': 'audio/wav'
                })
            
            # Add metadata
            response_data.update({
                'session_id': session_id,
                'fps': fps,
                'duration': video_clip.duration if video_clip else None,
                'file_size_mb': round(final_output.stat().st_size / (1024 * 1024), 2),
                'parameters': {
                    'steps': steps,
                    'cfg': cfg,
                    'seed': seed,
                    'num_frames': num_frames
                }
            })

            # Save metadata
            metadata = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': response_data['parameters'],
                'input_files': {
                    'reference_image': Path(reference_path).stem,
                    'audio_file': Path(audio_file.filename).stem,
                    'source_video': video_name
                },
                'storage_paths': {
                    'preprocessed': str(session_dirs['preprocess']),
                    'outputs': str(session_dirs['output']),
                    'final_video': str(final_output),
                    'cropped_video': str(cropped_video_path)
                }
            }
            
            metadata_file = session_dirs['output'] / "metadata.json"
            with metadata_file.open('w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Processing complete! Session ID: {session_id}")
            return response_data
           
            
        finally:
            # Clean up resources
            logger.info(f"Cleaning up session {session_id}")
            try:
                if audio_clip is not None:
                    audio_clip.close()
                if video_clip is not None:
                    video_clip.close()
                
                if reference_image:
                    temp_ref = Path('temp_reference.png')
                    if temp_ref.exists():
                        temp_ref.unlink()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

class FlaskAPIClient:
    def __init__(self, flask_api_url: str = 'http://localhost:5000'):
        self.base_url = flask_api_url
    
    async def send_for_animation(self, source_data: bytes, driving_data: bytes, source_is_image: bool, audio_data: Optional[bytes] = None) -> str:
        """
        Send videos/images data directly to Flask API for animation processing
        
        Args:
            source_data: Raw bytes of source video/image
            driving_data: Raw bytes of driving video
            source_is_image: Boolean indicating if source is an image
            
        Returns:
            str: URL to download the animated video
        """
        try:

            # Prepare multipart form data
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()

                # Add source data with appropriate content type
                if source_is_image:
                    logger.info("Sending source as image")
                    data.add_field('source_image',
                                   source_data,
                                   filename='source_image.png',
                                   content_type='image/png')
                    if audio_data:
                        logger.info("Adding audio data to request")
                        data.add_field('audio_data',
                                       audio_data,
                                       filename='audio.wav',
                                       content_type='audio/wav')
                else:
                    logger.info("Sending source as video")
                    data.add_field('source_video',
                                source_data,
                                filename='source_video.mp4',
                                content_type='video/mp4')
                    
                # Add driving video
                data.add_field('driving_video',
                               driving_data,
                               filename='driving_video.mp4',
                               content_type='video/mp4')
                
                logger.info("Sending files to Flask API")
                url = f"{self.base_url}/generate"
                    
                # Send request to Flask API
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise HTTPException(
                            status_code = response.status,
                            detail=f"Flask API Error: {error_data.get('error', 'Unknown error')}"
                        )
                    
                    result = await response.json()
                    return result.get("download_link")
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error in animation process: {str(e)}"
            )
        
class APIWorkflowManager:
    def __init__(self, flask_client: FlaskAPIClient):
        self.flask_client = flask_client
    
    async def process_full_workflow(self, initial_result: Dict) -> str:
        """
        Process the complete workflow from FastAPI to Flask
        
        Args:
            initial_result: Dictionary Containing source and driving video byte data
                Expected format:
                {
                    'source_video_data' or 'source_data': bytes,
                    'driving_video_data': bytes,
                    'source_content_type': str,
                    'using_custom_reference': bool
                }
            
        Returns:
            str: URL to download the animated video
        """
        try:
            # Extract source and driving data directly from the result
            if initial_result.get('using_custom_reference', False):
                source_data = initial_result.get('source_video_data')
            else:
                source_data = initial_result.get('source_data')
            
            driving_data = initial_result.get('driving_video_data')

            if not source_data or not driving_data:
                raise ValueError("Missing source or driving video data in initial result")
            
            # Determine if source is image based on content type
            source_is_image = initial_result.get('source_content_type') == 'image/png'
            logger.info(f"Processing workflow with source type: {'image' if source_is_image else 'video'}")

            # Send to Flask API for animation and return only the download URL
            animation_video_url = await self.flask_client.send_for_animation(
                source_data,
                driving_data,
                source_is_image
            )


            return animation_video_url
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error in workflow processing: {str(e)}"
            )



# Create FastAPI application
app = FastAPI(
    title="EchoMimic API",
    description="API for generating videos using EchoMimic model with single GPU support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create permanent storage directories
STORAGE_DIR = Path("permanent_storage")
STORAGE_DIR.mkdir(exist_ok=True)

# Mount the storage directory for static file serving
app.mount("/downloads", StaticFiles(directory=str(STORAGE_DIR)), name="downloads")

@app.post("/generate",
          summary="Generate video from audio using static reference image")
async def generate_video(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file (WAV format, maximum duration: 25 seconds)"),
    reference_image: UploadFile = File(None, description ="Optional reference image containing hands"),
    fps: int = 15,
    steps: int = 6,
    cfg: float = 1.0,
    seed: Optional[int] = None
):
    """
    Generate a video using the EchoMimic model.
    
    Parameters:
    - audio_file: Audio file (WAV format, maximum duration: 25 seconds)
    - reference_image: Optional reference image containing hands
    - fps: Frames per second (default: 15)
    - steps: Number of denoising steps (default: 6)
    - cfg: Classifier-free guidance scale (default: 1.0)
    - seed: Random seed for generation (optional)
    
    Returns:
    If hands are detected in reference image:
    {
        "source_video_data": bytes,     # Raw bytes of the original model output
        "driving_video_data": bytes,    # Raw bytes of the cropped model output
        "source_content_type": str,     # Content type of source (video/mp4)
        "using_custom_reference": bool, # True if using custom reference
    }
    
    if no hands are detected or no reference image provided:
    {
        "source_data": bytes,          # Raw Bytes of the static reference image
        "driving_video_data": bytes,   # Raw Bytes to the cropped model output
        "source_content_type": str,    # Content type of source (image/png)
        "using_custom_reference": bool # False if using static reference
    }
    """
    try:
        inference_manager = InferenceManager()
        result = await inference_manager.process_inputs(
            audio_file,
            "../YoutubeScrapper/Youtube_crawled/summary.json",
            reference_image=reference_image,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed
        )

        base_url = "http://localhost:8000"
        
        # Simplify response based on hand detection
        if result.get('using_custom_reference', False):
            # Hands detected - return original and cropped output URLs
            response = {
                "source_video_data": result['source_video_data'],
                "driving_video_data": result['driving_video_data'],
                "source_content_type": 'video/mp4',
                "using_custom_reference": True
            }
        else:
            # No hands detected - return static image and cropped output URLs
            response = {
                "source_data": result['source_data'],
                "driving_video_data": result['driving_video_data'],
                "source_content_type": "image/png",
                "using_custom_reference": False
            }
        
        # Add metadata if needed
        if "parameters" in result:
            response['parameters'] = result['parameters']
        if "session_id" in result:
            response['session_id'] = result['session_id']
        return response

    except Exception as e:
        logger.error(f"Error in generate_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating video: {str(e)}")

@app.get("/sessions/{session_id}",
         summary="Get session information")
async def get_session_data(session_id: str):
    """
    Get metadata and paths for a specific processing session
    
    Parameters:
    - session_id: Unique session identifier
    
    Returns:
    Session metadata including:
    - timestamp
    - parameters used
    - input files
    - storage paths
    """
    try:
        metadata_file = Path(f"storage/outputs/{session_id}/metadata.json")
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")
            
        with metadata_file.open() as f:
            metadata = json.load(f)
            
        return JSONResponse(content=metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos",
         summary="List all generated videos")
async def list_videos():
    """
    Get a list of all generated videos in storage.
    
    Returns:
    List of video information including:
    - filename
    - download_url
    - file_size_mb
    - created_at
    """
    try:
        videos = []
        for video_file in STORAGE_DIR.glob("*.mp4"):
            stats = video_file.stat()
            file_size_mb = stats.st_size / (1024 * 1024)
            created_at = datetime.fromtimestamp(stats.st_mtime).isoformat()
            
            videos.append({
                "filename": video_file.name,
                "download_url": f"/downloads/{video_file.name}",
                "file_size_mb": round(file_size_mb, 2),
                "created_at": created_at
            })
        
        return sorted(videos, key=lambda x: x['created_at'], reverse=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/videos/{filename}",
            summary="Delete a specific video")
async def delete_video(filename: str):
    """
    Delete a specific video from storage.
    
    Parameters:
    - filename: Name of the video file to delete
    
    Returns:
    - message: Success message
    """
    try:
        video_path = STORAGE_DIR / filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
            
        video_path.unlink()
        return {"message": f"Video {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health",
         summary="Check API health")
async def health_check():
    """
    Check if the API is running and GPU is available.
    
    Returns:
    - status: API status
    - gpu_available: Whether GPU is available
    - storage_info: Information about storage directories
    """
    try:
        gpu_available = torch.cuda.is_available()
        gpu_info = {
            "available": gpu_available,
            "device_count": torch.cuda.device_count() if gpu_available else 0,
            "device_name": torch.cuda.get_device_name(0) if gpu_available else None
        }
        
        storage_info = {
            "permanent_storage": {
                "path": str(STORAGE_DIR),
                "exists": STORAGE_DIR.exists()
            },
            "processing_storage": {
                "path": "storage",
                "exists": Path("storage").exists()
            }
        }
        
        return {
            "status": "healthy",
            "gpu_info": gpu_info,
            "storage_info": storage_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Update the FastAPI endpoint to use the workflow
@app.post("/generate_animated",
          summary="Generate and animate video from audio using full pipeline")
async def generate_animated_video(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file (WAV Format, maximum duration: 25 secs)"),
    reference_image: UploadFile = File(None, description="Optional  reference image containing hands"),
    fps: int = 15,
    steps: int = 6,
    cfg: float = 1.0,
    seed: Optional[int] = None
):
    """
    Generate a video using the Echomimic model and then animate it using the animation pipeline.
    
    Parameters are the same as the /generate endpoint, but this endpoint includes animation processing.
    
    Returns:
    {
        "animated_video_url": str,  # URL to download the final animated video
    }
    """
    logger.info(f"Received animated video generation request - Audio: {audio_file.filename}")
    try:
        # Initialize managers
        inference_manager = InferenceManager()
        flask_client = FlaskAPIClient()
        workflow_manager = APIWorkflowManager(flask_client)

        # First, generate the initial video
        logger.info("Generating initial video...")
        initial_result = await inference_manager.process_inputs(
            audio_file,
            "../YoutubeScrapper/Youtube_crawled/summary.json",
            reference_image=reference_image,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed
        )

        # Process through animation pipeline
        logger.info("Initial video generating complete. Processing animation...")

        # Extract source and driving video data from result

        source_data = initial_result.get('source_video_data' if initial_result.get('using_custom_reference') else 'source_data')
        driving_data = initial_result.get('driving_video_data')
        audio_data = initial_result.get('audio_data')
        source_is_image = initial_result.get('source_content_type') == 'image/png'
        animated_video_url = await flask_client.send_for_animation(
            source_data,
            driving_data,
            source_is_image,
            audio_data if source_is_image else None
        )
        logger.info("Animation Complete")
        return {'animated_video_url': animated_video_url}
    
    except Exception as e:
        logger.error(f"Error in generate_animated_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in animation pipeline: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    HOST = "0.0.0.0"
    PORT = 8000
    app.state.base_url = f"http://{HOST}:{PORT}"
    uvicorn.run(app, host=HOST, port=PORT)



# 1. Generate Video(the main endpoint)
"""
curl -X POST "http://localhost:8000/generate" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@/home/hertzai2019/AnimateX/YoutubeScrapper/audio_clips/5_secs.wav" \
  -F "reference_image=@/home/hertzai2019/AnimateX/echomimic_v2/EMTD_dataset/ref_imgs_by_FLUX/man/0003.png" \
  -F "fps=15" \
  -F "steps=6" \
  -F "cfg=1.0"
"""
#2. Get Session Information
"""
curl -X GET "http://localhost:8000/sessions/20240127_123456_7890" \
  -H "accept: application/json"
"""
#3. List All Videos:
"""
curl -X GET "http://localhost:8000/videos" \
  -H "accept: application/json"
"""
#4. Delete a Video:
"""
curl -X DELETE "http://localhost:8000/videos/video_name.mp4" \
  -H "accept: application/json"
"""
#5. Check API Health:
"""
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
"""

#6. With LivePortrait
"""
curl -X POST "http://34.64.120.240:8000/generate_animated" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@/home/hertzai2019/AnimateX/YoutubeScrapper/audio_clips/5_secs.wav" \
  -F "reference_image=@/home/hertzai2019/AnimateX/echomimic_v2/EMTD_dataset/ref_imgs_by_FLUX/man/0101.png"
"""