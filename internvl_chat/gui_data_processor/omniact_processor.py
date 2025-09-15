import os
import json
from PIL import Image

def format_human(task_text, image_token='<image>'):
	"""Format the human prompt for the conversation."""

	input_text = f"{image_token}\nQ: what action should the agent take to {task_text.strip()}? A:"
	
	return input_text

def format_gpt(output_script):
	"""Format the GPT response for the conversation."""
	return output_script.strip()

def get_image_size(image_path):
	try:
		with Image.open(image_path) as img:
			return img.width, img.height
	except Exception:
		return None, None

def process_omniact(
	split_json_path,
	output_jsonl_path,
	metafile_path,
	root_dir="./",
	annotation_name="omniact_chat.jsonl",
	dataset_name="omniact_chat",
	data_augment=False,
	max_dynamic_patch=12,
	repeat_time=1
):
	# Load split json
	with open(split_json_path, 'r') as f:
		data = json.load(f)

	chat_data = []

	for idx, (k, v) in enumerate(data.items()):
		# Read task file
		task_path = os.path.join(root_dir, v["task"]) if not os.path.isabs(v["task"]) else v["task"]
		try:
			with open(task_path, 'r') as tf:
				lines = [l.rstrip('\n') for l in tf.readlines()]
			# New logic: first line is input, second is label, rest are output
			if not lines:
				raise ValueError(f"Annotation file empty: {task_path}")
			# Input: remove 'Task:' prefix if present
			if lines[0].startswith("Task:"):
				task_text = lines[0][len("Task:"):].strip()
			else:
				task_text = lines[0].strip()
			# Output: handle 'Output Script:' on its own line or with output
			output_lines = []
			if len(lines) > 1 and lines[1].strip().lower().startswith('output'):
				# Check if output is on the same line as label
				after_colon = lines[1].split(':', 1)[1].strip() if ':' in lines[1] else ''
				if after_colon:
					output_lines = [after_colon]
					if len(lines) > 2:
						# If more lines, treat them as additional output
						output_lines += [l for l in lines[2:] if l.strip() != '']
				else:
					output_lines = [l for l in lines[2:] if l.strip() != '']
			else:
				output_lines = [l for l in lines[1:] if l.strip() != '']
			output_script = "\n".join(output_lines).strip() if output_lines else None
			if output_script is None or output_script == "":
				raise ValueError(f"No GPT output found in annotation: {task_path}")
		except Exception as e:
			raise RuntimeError(f"Failed to process annotation {task_path}: {e}")

		# Image info (robust path resolution)
		image_rel_path = v["image"]
		# Remove leading slashes if present
		image_rel_path_clean = image_rel_path.lstrip("/\\")
		image_abs_path = os.path.join(root_dir, image_rel_path_clean)
		width, height = get_image_size(image_abs_path)
		if width is None or height is None:
			# Try removing underscores from the filename and try again
			dir_name, file_name = os.path.split(image_abs_path)
			file_name_no_underscore = file_name.replace('_', '')
			alt_image_abs_path = os.path.join(dir_name, file_name_no_underscore)
			width, height = get_image_size(alt_image_abs_path)
			if width is None or height is None:
				raise FileNotFoundError(f"Could not read image: {image_abs_path} or {alt_image_abs_path}")

		# Metadata
		metadata = {key: v[key] for key in v if key not in ["task", "image"]}

		# Compose conversation
		conversations = [
			{"from": "human", "value": format_human(task_text)},
			{"from": "gpt", "value": format_gpt(output_script)}
		]

		chat_data.append({
			"id": int(k),
			"image": image_rel_path,
			"width": width,
			"height": height,
			"conversations": conversations,
			"metadata": metadata
		})

	# Write chat-format .jsonl
	with open(output_jsonl_path, 'w') as f:
		for item in chat_data:
			f.write(json.dumps(item, ensure_ascii=False) + "\n")

	# Write metafile .json
	metafile = {
		dataset_name: {
			"root": root_dir,
			"annotation": annotation_name,
			"data_augment": data_augment,
			"max_dynamic_patch": max_dynamic_patch,
			"repeat_time": repeat_time,
			"length": len(chat_data)
		}
	}
	with open(metafile_path, 'w') as f:
		json.dump(metafile, f, indent=2)

# Example usage:
if __name__ == "__main__":
	base_dir = "/home/vli/thangdd_workspace/datasets/OmniAct/"
	prefix = "internvlfm_"
	splits = [
		("train", True),
		("val", False),
		("test", False)
	]
	for split_name, do_metafile in splits:
		split_json = os.path.join(base_dir, f"{split_name}.json")
		out_jsonl = os.path.join(base_dir, f"{prefix}{split_name}_chat.jsonl")
		metafile = os.path.join(base_dir, f"{prefix}{split_name}_metafile.json") if do_metafile else None
		annotation_name = f"{prefix}{split_name}_chat.jsonl"
		dataset_name = f"{prefix}{split_name}_chat"
		if do_metafile:
			process_omniact(
				split_json_path=split_json,
				output_jsonl_path=out_jsonl,
				metafile_path=metafile,
				root_dir=base_dir,
				annotation_name=annotation_name,
				dataset_name=dataset_name
			)
		else:
			# For val/test, skip metafile
			process_omniact(
				split_json_path=split_json,
				output_jsonl_path=out_jsonl,
				metafile_path="/dev/null",  # dummy, won't be used
				root_dir=base_dir,
				annotation_name=annotation_name,
				dataset_name=dataset_name
			)
