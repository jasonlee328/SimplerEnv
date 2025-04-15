import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
# from simpler_env.policies.octo.octo_server_model import OctoServerInference
# from simpler_env.policies.rt1.rt1_model import RT1Inference
# from simpler_env.policies.llava.llava_model import LLaVAInference
from simpler_env.policies.molmo.molmo_model import MolmoInference
# from simpler_env.policies.llava_line.llava_line_model import LLaVALineInference
# from simpler_env.policies.moellava.moellava_model import MoeLLaVAInference
# try:
#     from simpler_env.policies.octo.octo_model import OctoInference
# except ImportError as e:
#     print("Octo is not correctly imported.")
#     print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # # policy model creation; update this if you are using a new policy model
    # if args.policy_model == "rt1":
    #     assert args.ckpt_path is not None
    #     model = RT1Inference(
    #         saved_model_path=args.ckpt_path,
    #         policy_setup=args.policy_setup,
    #         action_scale=args.action_scale,
    #     )
    # elif "octo" in args.policy_model:
    #     if args.ckpt_path is None or args.ckpt_path == "None":
    #         args.ckpt_path = args.policy_model
    #     if "server" in args.policy_model:
    #         model = OctoServerInference(
    #             model_type=args.ckpt_path,
    #             policy_setup=args.policy_setup,
    #             action_scale=args.action_scale,
    #         )
    #     else:
    #         model = OctoInference(
    #             model_type=args.ckpt_path,
    #             policy_setup=args.policy_setup,
    #             init_rng=args.octo_init_rng,
    #             action_scale=args.action_scale,
    #         )
    
    
    
    
    
    
    
    # if "llava" in args.policy_model:
    #     model = LLaVAInference(
    #         saved_model_path= args.ckpt_path,
    #         policy_setup=args.policy_setup,
    #     )
    if "molmo" in args.policy_model:
        model = MolmoInference(
            saved_model_path= args.ckpt_path,
            policy_setup=args.policy_setup,
        )
    # elif "line" in args.policy_model:
    #     model = LLaVALineInference(
    #         saved_model_path= args.ckpt_path,
    #         policy_setup=args.policy_setup,
    #     )
    else:
        raise NotImplementedError()
    # elif args.policy_model == "openvla":
    #     assert args.ckpt_path is not None
    #     from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    #     model = OpenVLAInference(
    #         saved_model_path=args.ckpt_path,
    #         policy_setup=args.policy_setup,
    #         action_scale=args.action_scale,
    #     )   
        
        
        
        
    # if "moellava" in args.policy_model:
    #     model = MoeLLaVAInference(
    #         saved_model_path= args.ckpt_path,
    #         policy_setup=args.policy_setup,
    #     )
    # else:
    #     raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))


# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from PIL import Image
# import requests

# device = "cuda:0"

# # load the processor
# processor = AutoProcessor.from_pretrained(
#     'hqfang/random-fix-step48000-hf',
#     trust_remote_code=True,
#     torch_dtype='auto',
#     device_map={"": device}
# )

# # load the model
# model = AutoModelForCausalLM.from_pretrained(
#     'hqfang/random-fix-step48000-hf',
#     trust_remote_code=True,
#     torch_dtype='auto',
#     device_map={"": device}
# )

# model.to(device)

# language_instruction = "pick coke can"
# system_prompt = f"The task is {language_instruction}. What is the action that the robot should take. To figure out the action that the robot should take to {language_instruction}, let's think through it step by step. First, what is the depth map for this image? Second, what is the trajectory of the end effector? Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"

# # process the image and text

# image_path = "/data/input/jiafei/GroundedVLA/SimplerEnv/images/teaser.png"
# image = Image.open(image_path)

# # process the image and text
# inputs = processor.process(
#     images=[image],
#     text=system_prompt,
# )

# # move inputs to the correct device and make a batch of size 1
# inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# # print(inputs['input_ids'])

# # print(processor.tokenizer.decode(inputs['input_ids'][0]))

# # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
# output = model.generate_from_batch(
#     inputs,
#     GenerationConfig(max_new_tokens=2048, stop_strings="<|endoftext|>"),
#     tokenizer=processor.tokenizer
# )

# # only get generated tokens; decode them to text
# generated_tokens = output[0,inputs['input_ids'].size(1):]
# # print(generated_tokens)
# generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# # print the generated text

# print(generated_text)



























# import os
# import time
# import platform
# import psutil
# import wandb
# import numpy as np
# import tensorflow as tf

# from simpler_env.evaluation.argparse import get_args
# from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
# from simpler_env.policies.molmo.molmo_model import MolmoInference
# import faulthandler
# faulthandler.enable()

# if __name__ == "__main__":
#     # Initialize wandb for system logging
#     wandb.login(key="3e966ed01bba39e748636ab8bd6f7835e7031253")
#     wandb.init(project="Simpler", entity="jasonlee328", config={"run_type": "system_log"})

#     # Collect basic system information
#     system_info = {
#         "platform": platform.platform(),
#         "processor": platform.processor(),
#         "cpu_count": psutil.cpu_count(logical=True),
#         "memory_total_bytes": psutil.virtual_memory().total
#     }
#     # Update wandb config and log the system info
#     wandb.config.update(system_info)
#     wandb.log({"system_info": system_info})
    
#     # Mark start time of the run
#     start_time = time.time()
    
#     # Get command-line arguments
#     args = get_args()

#     # Set display environment variable if necessary
#     os.environ["DISPLAY"] = ""
#     # Prevent a single JAX process from taking up all the GPU memory
#     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#     # Configure TensorFlow GPU memory allocation if GPUs are present
#     gpus = tf.config.list_physical_devices("GPU")
#     if len(gpus) > 0:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)]
#         )

#     # Create the policy model based on args
#     if "molmo" in args.policy_model:
#         model = MolmoInference(
#             saved_model_path=args.ckpt_path,
#             policy_setup=args.policy_setup,
#         )
#     else:
#         raise NotImplementedError("The specified policy model is not implemented.")
    
#     # Run real-to-sim evaluation
#     success_arr = maniskill2_evaluator(model, args)
#     average_success = np.mean(success_arr)
    
#     # Calculate run duration
#     run_duration = time.time() - start_time
    
#     # Log evaluation results and run duration to wandb
#     wandb.log({
#         "Average success": average_success,
#         "Run duration (s)": run_duration
#     })
    
#     print(args)
#     print(" " * 10, "Average success", average_success)
    
#     # Finish the wandb run
#     wandb.finish()
