import os
import cv2
import numpy as np
import decord
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm
from src.models.dwpose.dwpose_detector import dwpose_detector as dwprocessor

class VideoPreprocessor:
    def __init__(self, max_size=768):
        self.MAX_SIZE = max_size

    def _sanitize_filename(self, filename):
        """Sanitize filename to avoid issues with special characters
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Sanitized filename
        """
        if filename is None:
            raise ValueError("Filename cannot be None")
            
        print(f"Sanitizing filename: {filename}")  # Debug print
        
        # Replace problematic characters
        sanitized = filename.replace('-', '_')
        # Remove multiple dots except the last one (extension)
        parts = sanitized.split('.')
        if len(parts) > 2:
            sanitized = '_'.join(parts[:-1]) + '.' + parts[-1]
            
        print(f"Sanitized result: {sanitized}")  # Debug print
        return sanitized

    def _get_video_name(self, video_path):
        """Extract video name from path without extension
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            str: Video name without extension
        """
        if video_path is None:
            raise ValueError("Video path cannot be None")
            
        print(f"Processing video path: {video_path}")  # Debug print
        
        # Get the base filename
        base_name = os.path.basename(video_path)
        print(f"Base name: {base_name}")  # Debug print
        
        # Remove extension
        video_name = os.path.splitext(base_name)[0]
        print(f"Video name without extension: {video_name}")  # Debug print
        
        # Sanitize the video name
        video_name = self._sanitize_filename(video_name)
        print(f"Final video name: {video_name}")  # Debug print
        
        return video_name

    def _create_video_dirs(self, base_dir, video_name):
        """Create directory structure for video processing outputs
        
        Args:
            base_dir (str): Base output directory
            video_name (str): Name of the video being processed
            
        Returns:
            dict: Dictionary containing paths to created directories
        """
        # Create main video directory
        video_dir = os.path.join(base_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Create subdirectories
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

    def convert_fps(self, src_path, tgt_path, tgt_fps=24, tgt_sr=16000):
        """Convert video FPS and audio sample rate
        
        Args:
            src_path (str): Source video path
            tgt_path (str): Target video path
            tgt_fps (int, optional): Target video FPS. Defaults to 24.
            tgt_sr (int, optional): Target audio sample rate. Defaults to 16000.
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

    def save_frames(self, frames, frames_dir):
        """Save video frames as images
        
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

    def get_video_pose(self, video_path, sample_stride=1, max_frame=None):
        """Extract pose information from video frames
        
        Args:
            video_path (str): Path to video file
            sample_stride (int, optional): Frame sampling stride. Defaults to 1.
            max_frame (_type_, optional): Maximum number of frames. Defaults to None.

        Returns:
            tuple: (detected_poses, height, width, frames)
        """
        print(f"\nStarting pose detection for video: {video_path}")
        
        # Read input video
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        sample_stride *= max(1, int(vr.get_avg_fps() / 24))
        print(f"Video FPS: {vr.get_avg_fps()}, Sample stride: {sample_stride}")

        frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
        print(f"Total frames extracted: {len(frames)}")
        
        if max_frame is not None:
            frames = frames[0:max_frame,:,:]
            print(f"Using first {max_frame} frames")
            
        height, width, _ = frames[0].shape
        print(f"Frame dimensions: {height}x{width}")
        
        print("Starting pose detection...")
        detected_poses = []
        for i, frm in enumerate(tqdm(frames, desc="DWPose")):
            try:
                pose = dwprocessor(frm)
                if pose:
                    detected_poses.append(pose)
                    if i == 0:  # Print structure of first pose for debugging
                        print(f"\nFirst pose structure:")
                        for key in pose.keys():
                            if isinstance(pose[key], dict):
                                print(f"{key}: {type(pose[key])} with keys {pose[key].keys()}")
                            elif isinstance(pose[key], np.ndarray):
                                print(f"{key}: Array with shape {pose[key].shape}")
                            else:
                                print(f"{key}: {type(pose[key])}")
            except Exception as e:
                print(f"Error detecting pose for frame {i}: {str(e)}")
                
        print(f"Successfully detected {len(detected_poses)} poses")
        dwprocessor.release_memory()

        return detected_poses, height, width, frames

    def get_pose_params(self, detected_poses, height, width):
        """Calculate pose parameters
        
        Args:
            detected_poses (list): List of detected poses
            height (int): Frame height 
            width (int): Frame width

        Returns:
            dict: Pose parameters
        """
        print(f"\nStarting pose parameter calculation")
        print(f"Input dimensions - Height: {height}, Width: {width}")
        print(f"Number of detected poses: {len(detected_poses)}")
        
        if not detected_poses:
            print("Error: No poses detected")
            return None
            
        # Iterate through poses and find boundaries
        w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
        mid_all = []
        
        try:
            for num, detected_pose in enumerate(detected_poses):
                print(f"\nProcessing pose {num}")
                
                if not isinstance(detected_pose, dict):
                    print(f"Error: Invalid pose format for pose {num}")
                    print(f"Pose type: {type(detected_pose)}")
                    continue
                    
                if 'bodies' not in detected_pose or 'faces' not in detected_pose:
                    print(f"Error: Missing required keys in pose {num}")
                    print(f"Available keys: {detected_pose.keys()}")
                    continue
                
                detected_poses[num]['num'] = num
                
                try:
                    candidate_body = detected_pose['bodies']['candidate']
                    score_body = detected_pose['bodies']['score']
                    candidate_face = detected_pose['faces']
                    score_face = detected_pose['faces_score']
                    candidate_hand = detected_pose['hands']
                    score_hand = detected_pose['hands_score']

                    print(f"Pose {num} shapes:")
                    print(f"Body candidate: {candidate_body.shape}")
                    print(f"Body score: {score_body.shape}")
                    print(f"Face: {candidate_face.shape}")
                    print(f"Hands: {candidate_hand.shape}")

                    # Select highest confidence face detection
                    if candidate_face.shape[0] > 1:
                        index = 0
                        candidate_face = candidate_face[index]
                        score_face = score_face[index]
                        detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
                        detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
                    else:
                        candidate_face = candidate_face[0]
                        score_face = score_face[0]

                    # Select highest confidence body detection
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
                    hand_pose = candidate_hand

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

                except Exception as e:
                    print(f"Error processing pose {num}: {str(e)}")
                    continue

            if not w_min_all or not h_min_all:
                print("Error: No valid poses processed")
                return None

            print("\nCalculating final parameters...")
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

            # Calculate resize parameters
            imh_new, imw_new, rb, re, cb, ce = self._resize_and_pad_param(height_new, width_new, self.MAX_SIZE)

            result = {
                'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce],
                'pose_params': [w_min, w_max, h_min, h_max],
                'video_params': [h_min_real, h_max_real, w_min_real, w_max_real]
            }
            print("Successfully generated parameters")
            return result

        except Exception as e:
            print(f"\nError in get_pose_params: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _resize_and_pad_param(self, imh, imw, max_size):
        """Calculate padding parameters for resizing
        
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

    def save_pose_params(self, detected_poses, pose_params, draw_pose_params, output_dir):
        """Save pose parameters to files
        
        Args:
            detected_poses (list): List of detected poses
            pose_params (list): Pose parameters 
            draw_pose_params (list): Drawing parameters
            output_dir (str): Output directory path

        Returns:
            str: Path to saved pose files
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving pose parameters to {output_dir}")

        w_min, w_max, h_min, h_max = pose_params
        for detected_pose in detected_poses:
            num = detected_pose['num']
            print(f"Processing pose {num}")
            
            try:
                candidate_body = detected_pose['bodies']['candidate']
                candidate_face = detected_pose['faces'][0]
                candidate_hand = detected_pose['hands']
                
                # Normalize coordinates
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
                print(f"Saved pose {num} to {save_path}")
                
            except Exception as e:
                print(f"Error saving pose {num}: {str(e)}")
                continue
            
        return output_dir

    def process_video(self, input_video, output_dir, target_fps=24):
        """Main video processing pipeline
        
        Args:
            input_video (str): Input video path
            output_dir (str): Output directory path
            target_fps (int, optional): Target FPS. Defaults to 24.

        Returns:
            dict: Dictionary containing paths to processed files
        """
        print(f"Input video path: {input_video}")  # Debug print
        print(f"Output directory: {output_dir}")  # Debug print
        
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
            
        if not output_dir:
            raise ValueError("Output directory cannot be empty")
            
        # Get video name and create directory structure
        original_video_name = self._get_video_name(input_video)
        print(f"Original video name: {original_video_name}")  # Debug print
        
        if not original_video_name:
            raise ValueError("Failed to extract video name from path")
            
        # Use a simple name for temporary files to avoid moviepy issues
        temp_video_name = f"video_{hash(original_video_name) % 10000:04d}"
        print(f"Temporary video name: {temp_video_name}")  # Debug print
        
        dirs = self._create_video_dirs(output_dir, original_video_name)
        
        # Convert video FPS
        converted_video = os.path.join(dirs['video_dir'], f"{temp_video_name}_converted.mp4")
        self.convert_fps(input_video, converted_video, target_fps)

        # Extract pose information
        detected_poses, height, width, frames = self.get_video_pose(converted_video)
        
        # Save frames
        frame_paths = self.save_frames(frames, dirs['frames_dir'])
        
        # Get pose parameters
        print(f"Extracting pose parameters")  # Debug print
        res_params = self.get_pose_params(detected_poses, height, width)
        if res_params is None:
            raise ValueError("Failed to generate pose parameters")
            
        print(f"Successfully generated pose parameters: {res_params}")  # Debug print
        
        # Save pose parameters
        self.save_pose_params(
            detected_poses,
            res_params['pose_params'],
            res_params['draw_pose_params'],
            dirs['pose_dir']
        )
        
        # Extract audio
        temp_audio_path = os.path.join(dirs['audio_dir'], f"{temp_video_name}_audio.wav")
        final_audio_path = os.path.join(dirs['audio_dir'], f"{original_video_name}_audio.wav")
        
        video = VideoFileClip(converted_video)
        video.audio.write_audiofile(temp_audio_path)
        video.close()
        
        # Rename the audio file to its final name
        if os.path.exists(temp_audio_path):
            os.rename(temp_audio_path, final_audio_path)

        return {
            'video_dir': dirs['video_dir'],
            'converted_video': converted_video,
            'frames_dir': dirs['frames_dir'],
            'frame_paths': frame_paths,
            'pose_dir': dirs['pose_dir'],
            'audio_path': final_audio_path,
            'params': res_params
        }
    
# Initialize preprocessor
preprocessor = VideoPreprocessor(max_size=768)

# Example usage
results = preprocessor.process_video(
    input_video='EMTD_dataset/trimmed_videos/-Hmn5Gmn2dw/-Hmn5Gmn2dw_00-04-40.208_00-04-49.458.mp4',
    output_dir='EMTD_dataset/preprocessed_data',
    target_fps=24
)