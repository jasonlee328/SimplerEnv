from typing import Optional, Sequence
import os
import numpy as np
from transforms3d.euler import euler2axangle
from PIL import Image
import torch
import cv2 as cv
import re
import copy
import warnings
import ast
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import logging
# Import LLava-specific modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from transformers import Qwen2Tokenizer
from simpler_env.policies.llava.action_tokenize import * 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/gscratch/krishna/jason328/SimplerEnv/thelog.log',
    filemode='a'  # 'a' for append, 'w' for overwrite
)
logger = logging.getLogger(__name__)
os.environ["HF_HOME"] = "/gscratch/krishna/jason328"
os.environ["TRANSFORMERS_CACHE"] = "/gscratch/krishna/jason328"

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
            "vocab_size": 152064,
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

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
    


    def unnormalize_action(self, action_list_str):
        # Convert the string to a list of floats
        base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
        action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        action_list = action_tokenizer.decode_model_output(action_list_str)
        normalized_action = np.array(action_list)
        
        # Given action mean and std arrays
        if self.policy_setup == "google_robot":
            action_mean = np.array([0.00696389, 0.00627008, -0.01263256,
                                    0.04330839, -0.00570499, 0.00089247, 0.0])
            action_std = np.array([0.06925472, 0.06019009, 0.07354742,
                                   0.15605888, 0.1316399, 0.14593437, 1.0])
        else:
            action_mean = np.array([0.00021161, 0.00012614, -0.00017022,
                                    -0.00015062, -0.00023831, 0.00025646, 0.0])
            action_std = np.array([0.00963721, 0.0135066, 0.01251861,
                                   0.02806791, 0.03016905, 0.07632624, 1.0])
    
        # Unnormalize
        unnormalized_action = normalized_action * action_std + action_mean
        return unnormalized_action

    def unnormalize_action_tokenized(self, generated_text, chosen_probs, amplification_factor=1.0):
        import re
        # Use regex to extract the substring starting with "Action:" followed by a bracketed list.
        match = re.search(r"Action:\s*(\[[^\]]*\])", generated_text)
        if match:
            action_list_str = match.group(1)
        else:
            # If no match is found, check if the text starts directly with '['
            if generated_text.strip().startswith('['):
                action_list_str = generated_text.strip()
            else:
                raise ValueError("No action found in the generated text.")
        
        # Debug print to verify extraction
        # print("Extracted action string:", action_list_str)
    
        # Load the tokenizer and initialize the ActionTokenizer
        base_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
        action_tokenizer = ActionTokenizer(tokenizer=base_tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        
        # Split the action string into individual token strings
        token_list = action_list_str.strip("[]").split(", ")
        token_list = [token.strip() for token in token_list]
        # print("Action tokens:", token_list)
        individual_chars = []
        for token in token_list:
            for char in token:
                individual_chars.append(char)
        # print("Individual action characters:", individual_chars)
        
        
        action_confidence = self.compute_confidence_for_token_list(individual_chars, chosen_probs)
        print("Action Confidence:", action_confidence)
        logger.info(f"Action Confidence: {action_confidence}")

        
        # Convert tokens to ids and decode back to continuous actions
        token_ids = base_tokenizer.convert_tokens_to_ids(token_list)
        # print("Action confidence:", token_ids)
        recovered_actions = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
    
        # Define action mean and standard deviation arrays (for google_robot policy)
        action_mean = np.array([0.00696389, 0.00627008, -0.01263256,
                                0.04330839, -0.00570499, 0.00089247, 0.0])
        action_std = np.array([0.06925472, 0.06019009, 0.07354742,
                               0.15605888, 0.1316399, 0.14593437, 1.0])
    
        # Amplify the delta and unnormalize the actions
        delta = recovered_actions * action_std
        amplified_delta = delta * amplification_factor
        unnormalized_action = amplified_delta + action_mean
    
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
      
        # Generate output
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        
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
            
            
    
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
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
                print("Trajectory confidence:", traj_confidence)
                
                for i in range(len(trajectory) - 1):
                    pt1 = tuple(map(int, trajectory[i]))
                    pt2 = tuple(map(int, trajectory[i + 1]))
                    cv.line(annotated_image, pt1, pt2, (0, 255, 255), thickness=2, lineType=cv.LINE_AA)

                if traj_confidence > action_confidence:
                    logger.info("Trajectory confidence is higher than action confidence. Using Trajectory.")
                    print("Using Trajectory")
                    img = Image.fromarray(annotated_image)
                
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
