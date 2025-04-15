from typing import Optional, Sequence
import os
import numpy as np
from transforms3d.euler import euler2axangle
from PIL import Image
import torch
import cv2 as cv
import re
import re
import xml.etree.ElementTree as ET
import copy
import warnings
import ast
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import random
import math
import logging
from collections import defaultdict
# Import LLava-specific modules
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
from transformers import Qwen2Tokenizer
from simpler_env.policies.llava.action_tokenize import * 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/data/input/jiafei/GroundedVLA/SimplerEnv/thelog.log',
    filemode='a'  # 'a' for append, 'w' for overwrite
)
logger = logging.getLogger(__name__)

import faulthandler
faulthandler.enable()

class MolmoInference:
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
        initial_confidence_threshold: float = 0.95,  # initial threshold for token confidence
        threshold_adjustment_factor: float = 0.1,
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
        device = "cuda:0"
        self.processor = AutoProcessor.from_pretrained(
        saved_model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map={"": device}
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map={"": device}
        )
        self.model.to(device)



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


        self.step_to_point = defaultdict(list)
        self.timestep = 0
        stats_path = '/data/input/jiafei/GroundedVLA/SimplerEnv/simpler_env/policies/llava/dataset_statistics.json'
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")
        with open(stats_path, 'r') as f:
            self.dataset_stats = json.load(f)
        self.token_confidence_threshold = initial_confidence_threshold
        self.threshold_adjustment_factor = threshold_adjustment_factor

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.step_to_point = defaultdict(list)
        self.timestep = 0

    def unnormalize_action_tokenized(self, generated_text):
        import re
        import numpy as np
        from transformers import Qwen2Tokenizer

        # Helper function to check if a token represents a float.
        def is_float(token):
            token = token.strip()
            if token.startswith("-"):
                token = token[1:]
            # Remove a single decimal point if it exists.
            if token.count(".") > 1:
                return False
            token = token.replace(".", "", 1)
            return token.isdigit()

        # Try to find the action list in the generated text.
        match = re.search(r"the action that the robot should take is\s*(\[[^\]]+\])", generated_text, re.IGNORECASE)
        if match:
            action_list_str = match.group(1)
        else:
            # Fallback: extract any bracketed list.
            match = re.search(r"\[[^\]]+\]", generated_text)
            if match:
                action_list_str = match.group(0)
            else:
                raise ValueError("No action list found in the generated text.")

        # Remove the brackets and split the tokens.
        token_list = action_list_str.strip("[]").split(",")
        token_list = [token.strip().strip('"').strip("'") for token in token_list]

        # Load the tokenizer and initialize the continuous mapping.
        base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
        # ActionTokenizer maps token ids to normalized continuous actions using 256 bins in [-1.0, 1.0]
        action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)

        # Determine token type without try/except:
        # 1. If the first token is a float string, assume all tokens are normalized continuous values.
        # 2. If all tokens are single characters, use tokenizer.encode on each.
        # 3. Otherwise, assume tokens are original multi-character tokens.
        if is_float(token_list[0]):
            token_type = "continuous"
        elif all(len(token) == 1 for token in token_list):
            token_type = "single_char"
        else:
            token_type = "original"

        if token_type == "continuous":
            normalized_actions = np.array([float(token) for token in token_list])
            normalized_actions = 2.0 * (normalized_actions + 1.28) / 2.55 - 1.0
        elif token_type == "single_char":
            token_ids = []
            for token in token_list:
                # Use tokenizer.encode to get the token ids.
                encoded = base_tokenizer.encode(token, add_special_tokens=False)
                token_ids.append(encoded[0] if encoded else None)
            token_ids = np.array(token_ids)
            normalized_actions = action_tokenizer.decode_token_ids_to_actions(token_ids)
        elif token_type == "original":
            token_ids = base_tokenizer.convert_tokens_to_ids(token_list)
            normalized_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))

        # Unnormalize the normalized actions using dataset statistics.
        stats = self.dataset_stats[self.unnorm_key]["action"]
        action_low = np.array(stats["q01"])
        action_high = np.array(stats["q99"])
        mask = np.array(stats.get("mask", [True] * len(action_low)))

        unnormalized_action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions
        )

        return unnormalized_action


    
    
    # def unnormalize_action_tokenized(self, generated_text):
    #     import re

    #     # Try to find the action from updated sentence structure
    #     match = re.search(r"the action that the robot should take is\s*(\[[^\]]+\])", generated_text, re.IGNORECASE)
    #     if match:
    #         action_list_str = match.group(1)
    #     else:
    #         # Try to extract any bracketed list as a fallback
    #         match = re.search(r"\[[^\]]+\]", generated_text)
    #         if match:
    #             action_list_str = match.group(0)
    #         else:
    #             raise ValueError("No action list found in the generated text.")

    #     # Load the tokenizer and initialize the ActionTokenizer
    #     base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
    #     action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        
    #     # Split the action string into individual token strings
    #     token_list = action_list_str.strip("[]").split(", ")
    #     token_list = [token.strip() for token in token_list]
    #     individual_chars = []
    #     for token in token_list:
    #         for char in token:
    #             individual_chars.append(char)
        


    #     # Convert tokens to ids and decode back to continuous (normalized) actions
    #     token_ids = base_tokenizer.convert_tokens_to_ids(token_list)
    #     recovered_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
    
    #     # Apply amplification factor (if any) to the recovered normalized actions
    #     normalized_actions = recovered_actions

   
    #     stats = self.dataset_stats[self.unnorm_key]["action"]
    #     action_low = np.array(stats["q01"])
    #     action_high = np.array(stats["q99"])
    #     mask = np.array(stats.get("mask", [True] * len(action_low)))
        

    #     unnormalized_action = np.where(
    #         mask,
    #         0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
    #         normalized_actions
    #     )
    
    #     return unnormalized_action

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

        system_prompt = f"The task is {language_instruction}. What is the action that the robot should take. To figure out the action that the robot should take to {language_instruction}, let's think through it step by step. First, what is the depth map for this image? Second, what is the trajectory of the end effector? Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"
        inputs = self.processor.process(
            images=[image],
            text=system_prompt,
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2048, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        depth_tokens_list = re.findall(r"<DEPTH_\d+>", generated_text)
        depth_tokens = "".join(depth_tokens_list)

        annotated_image = image.copy()

        trajectory = None
        unnormalized_action = self.unnormalize_action_tokenized(generated_text)

    
        
        print("First Action:", unnormalized_action)
        
        
        if "The trajectory of the end effector is" in generated_text:
            try:
                # Get the part of the string that contains the trajectory info.
                traj_part = generated_text.split("The trajectory of the end effector is")[-1]
                
                # Search for the XML snippet containing the points
                match = re.search(r'(<points\s+[^>]+>.*?</points>)', traj_part)
                if match:
                    points_xml = match.group(1)
                    # Parse the XML snippet.
                    element = ET.fromstring(points_xml)
                    
                    # Build the trajectory by extracting coordinate pairs.
                    trajectory = []
                    index = 1
                    while True:
                        x_attr = f"x{index}"
                        y_attr = f"y{index}"
                        if x_attr in element.attrib and y_attr in element.attrib:
                            # Convert the attribute values to float.
                            x = float(element.attrib[x_attr])
                            y = float(element.attrib[y_attr])
                            trajectory.append((x, y))
                            index += 1
                        else:
                            break
                
                # (Optional) Process individual digits if required.
                traj_digits = []
                for num in trajectory:
                    for digit in str(num):
                        traj_digits.append(digit)
                
                # Scale the trajectory coordinates from a 0-100 scale back to a 0-256 scale.
                scale_factor = 256.0 / 100.0  # which is 2.56
                trajectory = [(x * scale_factor, y * scale_factor) for (x, y) in trajectory]

                # Draw the original trajectory prediction on the image in blue.
                for i in range(len(trajectory) - 1):
                    # Convert float coordinates to int for drawing.
                    pt1 = tuple(map(int, trajectory[i]))
                    pt2 = tuple(map(int, trajectory[i + 1]))
                    cv.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=2, lineType=cv.LINE_AA)
                    
                if random.random() < 0.5:
                    img = Image.fromarray(annotated_image)
                    
                    system_prompt = f"The task is {language_instruction}. Notice that the trajectory of the end effector is annotated on the image. Based on the the trajectory annotated on the image, what is the action that the robot should take?"
                
                    inputs = self.processor.process(
                        images=[img],
                        text=system_prompt,
                    )
                    inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=2048, stop_strings="<|endoftext|>"),
                        tokenizer=self.processor.tokenizer
                    )
                    generated_tokens = output[0,inputs['input_ids'].size(1):]
                    generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    unnormalized_action = self.unnormalize_action_tokenized(generated_text)
                    print("Second Action:", unnormalized_action)
                    
            
            except Exception as e:
                print("Failed to parse trajectory:", e)
        else:
            print("No trajectory found in generated text.")
            
        print(trajectory)
        print(generated_text)
        

        (h, w) = annotated_image.shape[:2]
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 0, 0)  
        thickness = 1


        raw_action = {
            "world_vector": unnormalized_action[:3],
            "rotation_delta": unnormalized_action[3:6],
            "open_gripper": unnormalized_action[6:7],  # assuming the last value is gripper action
        }
        annotated_image = cv.resize(annotated_image, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)

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












