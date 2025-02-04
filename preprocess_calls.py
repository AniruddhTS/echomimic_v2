import os
import cv2
import numpy as np
import decord
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm
from src.models.dwpose.dwpose_detector import dwpose_detector as dwprocessor
from typing import Optional, Dict, Union, List, Tuple

class VideoPreprocessor:
    def __init__(self, max_size: int = 768):
        """
        Initialize the VideoPreprocessor.
        
        Args:
            max_size (int): Maximum size for video frame processing
        """
        self.MAX_SIZE = max_size

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to avoid issues with special characters
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Sanitized filename
        """
        if filename is None:
            raise ValueError("Filename cannot be None")
            
        sanitized = filename.replace('-', '_')
        parts = sanitized.split('.')
        if len(parts) > 2:
            sanitized = '_'.join(parts[:-1]) + '.' + parts[-1]
            
        return sanitized

    def _get_video_name(self, video_path: str) -> str:
        """
        Extract video name from path without extension
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            str: Video name without extension
        """
        if video_path is None:
            raise ValueError("Video path cannot be None")
            
        base_name = os.path.basename(video_path)
        video_name = os.path.splitext(base_name)[0]
        return self._sanitize_filename(video_name)

    def _create_video_dirs(self, base_dir: str, video_name: str) -> Dict[str, str]:
        """
        Create directory structure for video processing outputs
        
        Args:
            base_dir (str): Base output directory
            video_name (str): Name of the video being processed
            
        Returns:
            dict: Dictionary containing paths to created directories
        """
        video_dir = os.path.join(base_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        frames_dir = os.path.join(video_dir, "frames")
        pose_dir = os.path.join(video_dir, "pose")
        audio_dir = os.path.join(video_dir, "audio")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        return {
            'video_dir': video_dir,
            'frames_dir': frames_dir,
            'pose_dir': pose_dir,
            'audio_dir': audio_dir
        }

    def convert_fps(self, src_path: str, tgt_path: str, tgt_fps: int = 24, tgt_sr: int = 16000) -> None:
        """
        Convert video FPS and audio sample rate
        
        Args:
            src_path (str): Source video path
            tgt_path (str): Target video path
            tgt_fps (int): Target video FPS
            tgt_sr (int): Target audio sample rate
        """
        clip = VideoFileClip(src_path)
        new_clip = clip.set_fps(tgt_fps)
        
        if tgt_fps is not None:
            audio = new_clip.audio
            audio = audio.set_fps(tgt_sr)
            new_clip = new_clip.set_audio(audio)
            
        new_clip.write_videofile(tgt_path, codec='libx264', audio_codec='aac')
        clip.close()
        new_clip.close()

    def save_frames(self, frames: np.ndarray, frames_dir: str) -> List[str]:
        """
        Save video frames as images
        
        Args:
            frames (numpy.ndarray): Array of video frames
            frames_dir (str): Directory to save frames
            
        Returns:
            list: List of saved frame paths
        """
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
        return frame_paths

    def get_video_pose(self, video_path: str, sample_stride: int = 1, max_frame: Optional[int] = None) -> Tuple:
        """
        Extract pose information from video frames
        
        Args:
            video_path (str): Path to video file
            sample_stride (int): Frame sampling stride
            max_frame (Optional[int]): Maximum number of frames

        Returns:
            tuple: (detected_poses, height, width, frames)
        """
        print(f"\nStarting pose detection for video: {video_path}")
        
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        sample_stride *= max(1, int(vr.get_avg_fps() / 24))
        
        frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
        
        if max_frame is not None:
            frames = frames[0:max_frame,:,:]
            
        height, width, _ = frames[0].shape
        
        detected_poses = []
        for i, frm in enumerate(tqdm(frames, desc="DWPose")):
            try:
                pose = dwprocessor(frm)
                if pose:
                    detected_poses.append(pose)
            except Exception as e:
                print(f"Error detecting pose for frame {i}: {str(e)}")
                
        dwprocessor.release_memory()
        return detected_poses, height, width, frames

    def get_pose_params(self, detected_poses: List[Dict], height: int, width: int) -> Optional[Dict]:
        """
        Calculate pose parameters
        
        Args:
            detected_poses (list): List of detected poses
            height (int): Frame height 
            width (int): Frame width

        Returns:
            Optional[dict]: Pose parameters
        """
        if not detected_poses:
            return None
            
        w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
        mid_all = []
        
        try:
            for num, detected_pose in enumerate(detected_poses):
                detected_poses[num]['num'] = num
                
                candidate_body = detected_pose['bodies']['candidate']
                score_body = detected_pose['bodies']['score']
                candidate_face = detected_pose['faces']
                score_face = detected_pose['faces_score']
                candidate_hand = detected_pose['hands']
                score_hand = detected_pose['hands_score']

                if candidate_face.shape[0] > 1:
                    index = 0
                    candidate_face = candidate_face[index]
                    score_face = score_face[index]
                    detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
                    detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
                else:
                    candidate_face = candidate_face[0]
                    score_face = score_face[0]

                if score_body.shape[0] > 1:
                    tmp_score = []
                    for k in range(0, score_body.shape[0]):
                        tmp_score.append(score_body[k].mean())
                    index = np.argmax(tmp_score)
                    candidate_body = candidate_body[index*18:(index+1)*18,:]
                    score_body = score_body[index]
                    score_hand = score_hand[(index*2):(index*2+2),:]
                    candidate_hand = candidate_hand[(index*2):(index*2+2),:,:]
                else:
                    score_body = score_body[0]

                body_pose = np.concatenate((candidate_body,))
                mid_ = body_pose[1, 0]

                face_pose = candidate_face
                
                h_min, h_max = np.min(face_pose[:,1]), np.max(body_pose[:7,1])
                h_ = h_max - h_min
                
                mid_w = mid_
                w_min = mid_w - h_ // 2
                w_max = mid_w + h_ // 2
                
                w_min_all.append(w_min)
                w_max_all.append(w_max)
                h_min_all.append(h_min)
                h_max_all.append(h_max)
                mid_all.append(mid_w)

            w_min = np.min(w_min_all)
            w_max = np.max(w_max_all) 
            h_min = np.min(h_min_all)
            h_max = np.max(h_max_all)
            mid = np.mean(mid_all)

            margin_ratio = 0.25
            h_margin = (h_max-h_min)*margin_ratio

            h_min = max(h_min-h_margin*0.65, 0)
            h_max = min(h_max+h_margin*0.05, 1)

            h_min_real = int(h_min*height)
            h_max_real = int(h_max*height)
            mid_real = int(mid*width)

            height_new = h_max_real-h_min_real+1
            width_new = height_new
            w_min_real = mid_real - height_new // 2
            
            if w_min_real < 0:
                w_min_real = 0
                width_new = mid_real * 2

            w_max_real = w_min_real + width_new
            w_min = w_min_real / width
            w_max = w_max_real / width

            imh_new, imw_new, rb, re, cb, ce = self._resize_and_pad_param(height_new, width_new, self.MAX_SIZE)

            return {
                'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce],
                'pose_params': [w_min, w_max, h_min, h_max],
                'video_params': [h_min_real, h_max_real, w_min_real, w_max_real]
            }

        except Exception as e:
            print(f"\nError in get_pose_params: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _resize_and_pad_param(self, imh: int, imw: int, max_size: int) -> Tuple[int, int, int, int, int, int]:
        """
        Calculate padding parameters for resizing
        
        Args:
            imh (int): Image height
            imw (int): Image width 
            max_size (int): Maximum dimension size

        Returns:
            tuple: (new_height, new_width, top, bottom, left, right) padding parameters
        """
        half = max_size // 2
        if imh > imw:
            imh_new = max_size
            imw_new = int(round(imw/imh * imh_new))
            half_w = imw_new // 2
            rb, re = 0, max_size 
            cb = half-half_w
            ce = cb + imw_new
        else:
            imw_new = max_size
            imh_new = int(round(imh/imw * imw_new))
            imh_new = max_size
            half_h = imh_new // 2
            cb, ce = 0, max_size
            rb = half-half_h
            re = rb + imh_new
            
        return imh_new, imw_new, rb, re, cb, ce

    def save_pose_params(self, detected_poses: List[Dict], pose_params: List[float], 
                        draw_pose_params: List[int], output_dir: str) -> str:
        """
        Save pose parameters to files
        
        Args:
            detected_poses (list): List of detected poses
            pose_params (list): Pose parameters 
            draw_pose_params (list): Drawing parameters
            output_dir (str): Output directory path

        Returns:
            str: Path to saved pose files
        """
        os.makedirs(output_dir, exist_ok=True)

        w_min, w_max, h_min, h_max = pose_params
        for detected_pose in detected_poses:
            num = detected_pose['num']
            
            try:
                candidate_body = detected_pose['bodies']['candidate']
                candidate_face = detected_pose['faces'][0]
                candidate_hand = detected_pose['hands']
                
                candidate_body[:,0] = (candidate_body[:,0]-w_min)/(w_max-w_min)
                candidate_body[:,1] = (candidate_body[:,1]-h_min)/(h_max-h_min)
                
                candidate_face[:,0] = (candidate_face[:,0]-w_min)/(w_max-w_min)
                candidate_face[:,1] = (candidate_face[:,1]-h_min)/(h_max-h_min)
                
                candidate_hand[:,:,0] = (candidate_hand[:,:,0]-w_min)/(w_max-w_min)
                candidate_hand[:,:,1] = (candidate_hand[:,:,1]-h_min)/(h_max-h_min)
                
                detected_pose['bodies']['candidate'] = candidate_body
                detected_pose['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
                detected_pose['hands'] = candidate_hand
                detected_pose['draw_pose_params'] = draw_pose_params
                
                save_path = os.path.join(output_dir, f"{num}.npy")
                np.save(save_path, detected_pose)
                
            except Exception as e:
                print(f"Error saving pose {num}: {str(e)}")
                continue
            
        return output_dir

    def process_video(self, 
                     input_video: Union[str, bytes],
                     output_dir: str,
                     target_fps: int = 24,
                     save_frames: bool = True,
                     save_audio: bool = True) -> Dict:
        """
        Process a video file or bytes stream.
        
        Args:
            input_video: Path to video file or video bytes
            output_dir: Output directory path
            target_fps: Target frames per second
            save_frames: Whether to save individual frames
            save_audio: Whether to extract and save audio
            
        Returns:
            Dict containing paths to processed files and parameters
        """
        if isinstance(input_video, bytes):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(input_video)
                input_video_path = temp_file.name
        else:
            if not os.path.exists(input_video):
                raise FileNotFoundError(f"Input video file not found: {input_video}")
            input_video_path = input_video
            # Create output directory structure
        video_name = self._get_video_name(input_video_path)
        dirs = self._create_video_dirs(output_dir, video_name)
        
        # Convert video FPS
        converted_video = os.path.join(dirs['video_dir'], f"{video_name}_converted.mp4")
        self.convert_fps(input_video_path, converted_video, target_fps)

        # Extract pose information
        detected_poses, height, width, frames = self.get_video_pose(converted_video)
        
        # Save frames if requested
        frame_paths = []
        if save_frames:
            frame_paths = self.save_frames(frames, dirs['frames_dir'])
        
        # Get pose parameters
        res_params = self.get_pose_params(detected_poses, height, width)
        if res_params is None:
            raise ValueError("Failed to generate pose parameters")
            
        # Save pose parameters
        self.save_pose_params(
            detected_poses,
            res_params['pose_params'],
            res_params['draw_pose_params'],
            dirs['pose_dir']
        )
        
        # Extract audio if requested
        audio_path = None
        if save_audio:
            audio_path = os.path.join(dirs['audio_dir'], f"{video_name}_audio.wav")
            video = VideoFileClip(converted_video)
            video.audio.write_audiofile(audio_path)
            video.close()

        # Clean up temporary file if input was bytes
        if isinstance(input_video, bytes):
            os.unlink(input_video_path)

        return {
            'video_dir': dirs['video_dir'],
            'converted_video': converted_video,
            'frames_dir': dirs['frames_dir'] if save_frames else None,
            'frame_paths': frame_paths if save_frames else None,
            'pose_dir': dirs['pose_dir'],
            'audio_path': audio_path,
            'params': res_params
        }


def main():
    """Command line interface for video preprocessing"""
    parser = argparse.ArgumentParser(description='Process video for pose detection and analysis')
    
    # Required arguments
    parser.add_argument('input_video', 
                       help='Path to input video file')
    parser.add_argument('output_dir', 
                       help='Output directory for processed files')
    
    # Optional arguments
    parser.add_argument('--target-fps', 
                       type=int, 
                       default=24, 
                       help='Target FPS for video conversion')
    parser.add_argument('--max-size', 
                       type=int, 
                       default=768, 
                       help='Maximum size for frame processing')
    parser.add_argument('--no-frames', 
                       action='store_true', 
                       help='Skip saving individual frames')
    parser.add_argument('--no-audio', 
                       action='store_true', 
                       help='Skip audio extraction')
    parser.add_argument('--sample-stride', 
                       type=int, 
                       default=1, 
                       help='Frame sampling stride for pose detection')
    parser.add_argument('--max-frame', 
                       type=int, 
                       default=None, 
                       help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    try:
        preprocessor = VideoPreprocessor(max_size=args.max_size)
        results = preprocessor.process_video(
            input_video=args.input_video,
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            save_frames=not args.no_frames,
            save_audio=not args.no_audio
        )
        
        print("\nProcessing completed successfully!")
        print(f"Results saved to: {results['video_dir']}")
        print("\nProcessed files:")
        print(f"- Converted video: {results['converted_video']}")
        if results['frames_dir']:
            print(f"- Frames directory: {results['frames_dir']}")
        print(f"- Pose directory: {results['pose_dir']}")
        if results['audio_path']:
            print(f"- Audio file: {results['audio_path']}")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

# Basic usage
"""
python preprocess_calls.py /home/hertzai2019/AnimateX/YoutubeScrapper/Youtube_crawled/-1DYsiKC7b4_scene_073.mp4 /home/hertzai2019/AnimateX/echomimic_v2/EMTD_dataset/processed_data
"""
# Advanced usage with options
"""
python preprocess_calls.py input.mp4 output_dir \
    --target-fps 30 \
    --max-size 1024 \
    --no-frames \
    --sample-stride 2 \
    --max-frame 1000
"""