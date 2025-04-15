from typing import Optional, Sequence
import os
import numpy as np
from transforms3d.euler import euler2axangle
from PIL import Image
import torch
import cv2 as cv
import re
import json
import copy
import warnings
import ast
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import logging
from collections import defaultdict
# Import LLava-specific modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from transformers import Qwen2Tokenizer
# from simpler_env.policies.llava.action_tokenize import * 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/data/input/jiafei/GroundedVLA/SimplerEnv/thelog.log',
    filemode='a'  # 'a' for append, 'w' for overwrite
)
logger = logging.getLogger(__name__)


class LLaVAInference:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "google_robot",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [256, 256],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        # Load LLava model
        device = "cuda"
        device_map = "auto"
        llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {
            'tie_word_embeddings': False,
            'use_cache': True,
            "vocab_size": 152035,
            "image_aspect_ratio": "pad"
        }
        llava_model_args["overwrite_config"] = overwrite_config

        model_name = "llava_qwen"  # Replace with your model name if different
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            saved_model_path, None, model_name, device_map=device_map, **llava_model_args
        )
        print(saved_model_path)
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.model.eval()
        self.conv_template = "qwen_1_5"

        # Other initializations
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0
        
        # self.trajectory_buffer = []
        # self.buffer_max_len = 5
        # self.num_step = 0

        self.step_to_point = defaultdict(list)
        self.timestep = 0
        stats_path = '/data/input/jiafei/GroundedVLA/SimplerEnv/simpler_env/policies/llava/dataset_statistics.json'
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")
        with open(stats_path, 'r') as f:
            self.dataset_stats = json.load(f)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None


        # self.trajectory_buffer = []
        # self.buffer_max_len = 5
        # self.num_step = 0

        self.step_to_point = defaultdict(list)
        self.timestep = 0
    

    def upsample_trajectory(self, trajectory, target_points=25):
        """
        Upsample a trajectory to have more points while maintaining the same path.
        
        Args:
            trajectory: List of [x, y] points defining the original trajectory
            target_points: Desired number of points in the upsampled trajectory
            
        Returns:
            List of [x, y] points with length target_points
        """
        if trajectory is None or len(trajectory) < 2:
            return trajectory
            
        # Calculate the total path length
        total_length = 0
        for i in range(len(trajectory) - 1):
            p1 = np.array(trajectory[i])
            p2 = np.array(trajectory[i + 1])
            segment_length = np.linalg.norm(p2 - p1)
            total_length += segment_length
            
        # Create evenly spaced points along the path
        upsampled = []
        current_length = 0
        
        for i in range(target_points):
            # Calculate target distance along path for this point
            target_distance = total_length * i / (target_points - 1)
            
            # Find the segment containing this distance
            segment_start = 0
            cumulative_length = 0
            
            for j in range(len(trajectory) - 1):
                p1 = np.array(trajectory[j])
                p2 = np.array(trajectory[j + 1])
                segment_length = np.linalg.norm(p2 - p1)
                
                if cumulative_length + segment_length >= target_distance:
                    # This segment contains our target point
                    segment_start = j
                    break
                    
                cumulative_length += segment_length
            
            # Interpolate within the segment
            p1 = np.array(trajectory[segment_start])
            p2 = np.array(trajectory[segment_start + 1])
            
            segment_length = np.linalg.norm(p2 - p1)
            segment_progress = (target_distance - cumulative_length) / segment_length
            
            # Clamp segment_progress to [0, 1] to handle floating point errors
            segment_progress = max(0, min(1, segment_progress))
            
            # Linear interpolation
            point = p1 + segment_progress * (p2 - p1)
            upsampled.append(point.tolist())
            
        logger.info(f"Upsampled trajectory from {len(trajectory)} to {len(upsampled)} points")
        return upsampled


    def map_action_tokens_to_floats_with_bins(self, action_str, n_bins=256, min_action=-1.0, max_action=1.0):
        """
        Given an action string (e.g., 
        "<ACTION_START><ACTION_90><ACTION_146><ACTION_118><ACTION_150><ACTION_133><ACTION_179><ACTION_1><ACTION_END>")
        this function extracts the numeric indices from the <ACTION_X> tokens and maps each
        to a continuous float using a binning approach.
        
        The binning is performed as follows:
        - Create `n_bins` uniformly spaced bin edges between min_action and max_action.
        - Compute bin centers from these edges (which gives n_bins-1 centers).
        - For each token index X, compute the bin index as: 
                bin_index = (n_bins - 1) - X
            (This inversion makes <ACTION_0> correspond to the highest bin center, 
            and <ACTION_255> correspond to the lowest.)
        - Clip the bin index to the valid range [0, n_bins-2] and use it to look up the bin center.
        
        Returns:
            A list of floats corresponding to the continuous actions.
        """
        # Create uniform bins and compute bin centers.
        bin_edges = np.linspace(min_action, max_action, n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # There will be n_bins-1 centers.
        
        # Extract token indices from the action string.
        indices = re.findall(r"<ACTION_(\d+)>", action_str)
        
        float_values = []
        for index_str in indices:
            idx = int(index_str)
            # Invert the index so that lower action token (e.g., 0) maps to the higher bin center.
            bin_index = (n_bins - 1) - idx
            # Clip the bin_index to the valid range (0 to number of bin centers - 1).
            bin_index = np.clip(bin_index, 0, len(bin_centers) - 1)
            float_values.append(bin_centers[bin_index])
        
        return float_values
    
    
    def unnormalize_action_tokenized(self, generated_text, chosen_probs):
        

        # Extract the substring between <ACTION_START> and <ACTION_END>
        match = re.search(r"<ACTION_START>(.*?)<ACTION_END>", generated_text)
        if match:
            action_token_str = "<ACTION_START>" + match.group(1) + "<ACTION_END>"
        else:
            raise ValueError("No action tokens found in the generated text.")

        # Extract individual action tokens (e.g., "<ACTION_90>", "<ACTION_146>", etc.)
        tokens = re.findall(r"<ACTION_\d+>", action_token_str)
        # Compute confidence using the entire tokens (instead of per-character)
        action_confidence = self.compute_confidence_for_token_list(tokens, chosen_probs)
        print("Action Confidence:", action_confidence)
        logger.info(f"Action Confidence: {action_confidence}")

        # Convert the action token string to continuous (normalized) actions using the binning approach.
        # This function is assumed to be imported (or defined) as map_action_tokens_to_floats_with_bins.
        recovered_actions = self.map_action_tokens_to_floats_with_bins(action_token_str)
        normalized_actions = np.array(recovered_actions)

        # Retrieve fractal action stats from the loaded dataset statistics using unnorm_key.
        stats = self.dataset_stats[self.unnorm_key]["action"]
        action_low = np.array(stats["q01"])
        action_high = np.array(stats["q99"])
        mask = np.array(stats.get("mask", [True] * len(action_low)))
        
        # Unnormalize actions:
        # For dimensions where mask is True, map the normalized actions from [-1, 1] to [action_low, action_high].
        # For dimensions where mask is False, retain the normalized value.
        unnormalized_action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions
        )

        return unnormalized_action, action_confidence
    
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
    
        assert image.dtype == np.uint8
        orig_h, orig_w = image.shape[:2]
        image = self._resize_image(image)
        img = Image.fromarray(image)
    
        language_instruction = self.task_description
        logger.info(f"Run started at: {datetime.now()}")
        # Generate system_prompt
        system_prompt = f"{DEFAULT_IMAGE_TOKEN}\nThe task is {language_instruction}. Can you predict the trajectory of the end effector and the action the robot should take? "
        # Process the image
        image_tensor = process_images([img], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
    
        # Create conversation
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], system_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
    
        # Generate input_ids
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    
        image_sizes = [img.size]
      

        
        gen_out = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        
        # --- Extract the generated text and probabilities ---
        num_generated = len(gen_out.scores)
        chosen_token_ids = gen_out.sequences[:, -num_generated:]
        chosen_probs = []
        for step, logits in enumerate(gen_out.scores):
            probs = F.softmax(logits, dim=-1)
            token_id = chosen_token_ids[:, step]
            token_prob = probs[torch.arange(probs.shape[0]), token_id]
            token_str = self.tokenizer.decode(token_id.item())
            chosen_probs.append((token_str, token_prob.item()))
            
            
    
        text_outputs = self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        generated_text = text_outputs[0]
        # print("Generated text:", generated_text)
        annotated_image = image.copy()
        # --- Annotate the image with the trajectory ---
        trajectory = None
        traj_confidence = 0.0
        # Process text output into unnormalized_action using the modified function
        unnormalized_action, action_confidence = self.unnormalize_action_tokenized(generated_text, chosen_probs)
        logger.info(f"First Action: {unnormalized_action}, Action Confidence: {action_confidence}")
    
        
        print("First Action:", unnormalized_action)
        
        
        if "Trajectory:" in generated_text:
            try:
                traj_part = generated_text.split("Action:")[0]
                traj_str = traj_part.replace("Trajectory:", "").strip()
                traj_str = traj_str.rstrip('.')
                trajectory = ast.literal_eval(traj_str)
                
                
                traj_digits = []
                for num in trajectory:
                    for digit in str(num):
                        traj_digits.append(digit)
                traj_confidence = self.compute_confidence_for_token_list(traj_digits, chosen_probs)
                logger.info(f"Trajectory: {trajectory}, Trajectory Confidence: {traj_confidence}")
                # print("Trajectory digits:", traj_digits)
                print(f"Trajectory: {trajectory}, Trajectory Confidence: {traj_confidence}")
                
                trajectory = self.upsample_trajectory(trajectory, target_points=25)
                for i, point in enumerate(trajectory):
                    self.step_to_point[self.timestep + i].append((point, self.timestep, traj_confidence))


                def weighted_sum(points, steps, confs, use_conf=True, k=2.0, c=0.5):
                    points = np.array(points)  # Shape (N, 2)
                    steps = np.array(steps)    # Shape (N,)
                    confs = np.array(confs)    # Shape (N,)
             

                    # Compute weights
                    weights = np.exp(-k * steps * (c * confs)) if use_conf else np.exp(-k * steps)

                    # Normalize weights
                    weights /= np.sum(weights)

                    # Compute weighted sum
                    weighted_sum = np.sum(points * weights[:, None], axis=0)

                    return weighted_sum.tolist()

                agg_traj = []
                for i in range(len(trajectory)):
                    points = [tup[0] for tup in self.step_to_point[self.timestep + i]]
                    steps = [tup[1] for tup in self.step_to_point[self.timestep + i]]
                    min_step = steps[0]
                    steps_adjust = [step - min_step + 1 for step in steps]
                    confs = [tup[2] for tup in self.step_to_point[self.timestep + i]]
                    
                    agg_point = weighted_sum(points, steps, confs, use_conf=True)
                    agg_traj.append(agg_point)
                
                print("Aggregated Trajectory:", agg_traj)
                logger.info(f"Aggregated Trajectory: {agg_traj}")
                
     
                # Draw the original trajectory prediction in blue
                for i in range(len(trajectory) - 1):
                    pt1 = tuple(map(int, trajectory[i]))
                    pt2 = tuple(map(int, trajectory[i + 1]))
                    cv.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=2, lineType=cv.LINE_AA)
                
                # if traj_confidence > action_confidence:
                if True:
                    logger.info("Trajectory confidence is higher than action confidence. Using Trajectory.")
                    print("Using Trajectory")
                    img = Image.fromarray(annotated_image)


                    # Draw the temporally aggregated trajectory in green
                    for i in range(len(agg_traj) - 1):
                        pt1 = tuple(map(int, agg_traj[i]))
                        pt2 = tuple(map(int, agg_traj[i + 1]))
                        cv.line(annotated_image, pt1, pt2, (255, 0, 0), thickness=2, lineType=cv.LINE_AA)


                    trajectory = agg_traj


                
                    language_instruction = self.task_description
                
                    # Generate system_prompt
                    system_prompt = f"{DEFAULT_IMAGE_TOKEN}\nThe task is {language_instruction}. The trajectory of the end effector is annotated on the observation. Can you predict the action the robot should take?"
                    # Process the image
                    image_tensor = process_images([img], self.image_processor, self.model.config)
                    image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
                
                    # Create conversation
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                    conv.append_message(conv.roles[0], system_prompt)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                
                    # Generate input_ids
                    input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
                
                    image_sizes = [img.size]
                
                    # Generate output
                    cont = self.model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                    )
                    
                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                    generated_text = text_outputs[0]
                    unnormalized_action, _ = self.unnormalize_action_tokenized(generated_text, chosen_probs)
                    
            
                    print("Second Action:", unnormalized_action)
                    logger.info(f"Second Action (from trajectory): {unnormalized_action}")
            except Exception as e:
                print("Failed to parse trajectory:", e)
        else:
            print("No trajectory found in generated text.")
            
        

        

            
        # Get the image dimensions
        (h, w) = annotated_image.shape[:2]

        # Define font parameters
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 0, 0)  # White text
        thickness = 1

        # Create the text strings
        text_line = f"line conf: {traj_confidence:.2f}"
        text_pose = f"pose conf: {action_confidence:.2f}"

        # Get text size to properly position the text
        (text_w1, text_h1), _ = cv.getTextSize(text_line, font, font_scale, thickness)
        (text_w2, text_h2), _ = cv.getTextSize(text_pose, font, font_scale, thickness)
        # Define a margin from the image borders
        margin = 10

        # Calculate the x position (aligned to the right)
        x = w - max(text_w1, text_w2) - margin

        # Calculate the y positions for the texts (from the top)
        y1 = margin + text_h1           # y coordinate for the first text line
        y2 = y1 + margin + text_h2      # y coordinate for the second text line

        # Put the texts on the image at the top right corner
        cv.putText(annotated_image, text_line, (x, y1), font, font_scale, color, thickness, cv.LINE_AA)
        cv.putText(annotated_image, text_pose, (x, y2), font, font_scale, color, thickness, cv.LINE_AA)
                    
                    
            
            
   
            
        # Create raw_action
        raw_action = {
            "world_vector": unnormalized_action[:3],
            "rotation_delta": unnormalized_action[3:6],
            "open_gripper": unnormalized_action[6:7],  # assuming the last value is gripper action
        }
        annotated_image = cv.resize(annotated_image, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
    
        # Process raw_action to obtain the action to be sent to the ManiSkill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
    
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action
    
            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
    
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
    
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
    
            action["gripper"] = relative_gripper_action
    
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
    
        action["terminate_episode"] = np.array([0.0])
    
        return raw_action, action, annotated_image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image
    
    def geometric_mean(self, probs):
        if not probs:
            return 0.0
        log_sum = sum(math.log(p) for p in probs)
        return math.exp(log_sum / len(probs))
    
    
    def compute_confidence_for_token_list(self, token_list, chosen_probs):
        """
        For each token in token_list, look up its probability in chosen_probs
        (which is a list of (token, prob) tuples) and return the geometric mean.
        """
        confs = []
        for token in token_list:
            for tok, prob in chosen_probs:
                if tok.strip() == token.strip():
                    confs.append(prob)
                    break
        if confs:
            return self.geometric_mean(confs)
        else:
            return 0.0
        
        
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
    
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
    
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
    
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
    
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
