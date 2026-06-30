import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import gather_object
from unsloth import FastLanguageModel
from transformers import set_seed

from config import PipelineConfig
from data.registry import DatasetRegistry
from prompts.factory import PromptBuilderFactory
from training.metrics import compute_ner_metrics, compute_re_metrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size per GPU")
    parser.add_argument("--output", type=str, default="predictions.jsonl", help="Optional path to save prediction results")
    parser.add_argument("--start_index", type=int, default=None, help="Start index of evaluation samples")
    parser.add_argument("--end_index", type=int, default=None, help="End index of evaluation samples")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize accelerate for Data Parallelism inference
    accelerator = Accelerator()
    
    # Only print logs from the main process
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    config = PipelineConfig.from_yaml(args.config)
    set_seed(config.model.seed)

    logger.info("Loading test data...")
    if not config.data.test_files:
        logger.warning("No test_files specified in config. Falling back to eval_files.")
        test_files = config.data.eval_files
    else:
        test_files = config.data.test_files
        
    eval_docs = DatasetRegistry(test_files).aggregate()
    logger.info("Total Test samples loaded: %d", len(eval_docs))

    builder = PromptBuilderFactory.create(config)
    eval_raw = builder.generate_training_data(eval_docs)
    
    # Add an ID column so we can restore the exact order after multi-GPU processing
    eval_raw = eval_raw.add_column("id", range(len(eval_raw)))
    
    # Filter dataset by start_index and end_index if specified
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else len(eval_raw)
        logger.info(f"Filtering dataset to samples from index {start} to {end} (total {end - start} samples)")
        eval_raw = eval_raw.select(range(start, end))
    
    # Define temporary file for this specific GPU
    part_file = f"{args.output}.part{accelerator.process_index}"
    
    # Check if we are resuming
    start_idx = 0
    if os.path.exists(part_file):
        with open(part_file, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        logger.info(f"Process {accelerator.process_index} found {start_idx} processed samples in {part_file}. Resuming...")
    
    # Shard the dataset across multiple GPUs
    if accelerator.num_processes > 1:
        eval_raw = eval_raw.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
        
    # Skip samples that were already generated and saved in a previous crashed run
    if start_idx > 0:
        if start_idx >= len(eval_raw):
            logger.info(f"Process {accelerator.process_index} has already finished its shard.")
            eval_raw = eval_raw.select([]) # empty
        else:
            eval_raw = eval_raw.select(range(start_idx, len(eval_raw)))
            
    logger.info(f"Process {accelerator.process_index} will process {len(eval_raw)} samples")

    logger.info(f"Loading model from {args.checkpoint} on GPU {accelerator.local_process_index}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=config.model.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map={"": accelerator.local_process_index},
    )
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    logger.info("Running inference...")
    
    # Open part file in APPEND mode so we can save incrementally
    with open(part_file, "a", encoding="utf-8") as f_out:
        # Only show progress bar on main process
        for i in tqdm(range(0, len(eval_raw), args.batch_size), disable=not accelerator.is_main_process):
            batch = eval_raw[i:i+args.batch_size]
            
            # Prepare inputs based on format
            inputs_text = []
            if config.training.format.value == "instruction":
                for sys_msg, usr_msg in zip(batch["system"], batch["user"]):
                    messages = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": usr_msg}
                    ]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs_text.append(text)
            else:
                # Causal format (Input only)
                for text in batch["text"]:
                    prompt = text.split("### Output:\n")[0] + "### Output:\n"
                    inputs_text.append(prompt)
                    
            # Tokenize batch
            inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode only the newly generated tokens
            input_lengths = [len(inp) for inp in inputs.input_ids]
            for doc_id, out, in_len in zip(batch["id"], outputs, input_lengths):
                generated_tokens = out[in_len:]
                gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Parse and save immediately
                try:
                    parsed = json.loads(gen_text)
                except json.JSONDecodeError:
                    parsed = {"error": "Malformed JSON", "raw_text": gen_text}
                    
                # Save as {"id": ..., "prediction": ...}
                f_out.write(json.dumps({"id": doc_id, "prediction": parsed}, ensure_ascii=False) + "\n")
            
            # Flush to disk immediately in case of crash
            f_out.flush()

    # Wait for all GPUs to finish their shards
    accelerator.wait_for_everyone()

    # Compute Metrics and Save Final Output ONLY on the main process
    if accelerator.is_main_process:
        logger.info("Gathering and sorting predictions from all GPUs...")
        
        all_results = []
        for rank in range(accelerator.num_processes):
            p_file = f"{args.output}.part{rank}"
            if os.path.exists(p_file):
                with open(p_file, "r", encoding="utf-8") as f:
                    for line in f:
                        all_results.append(json.loads(line))
        
        # Sort by original ID to guarantee exact match with gold references
        all_results.sort(key=lambda x: x["id"])
        
        # Extract purely the predictions as string JSONs (which compute_metrics expects)
        all_predictions = [json.dumps(x["prediction"], ensure_ascii=False) for x in all_results]
        
        # Generate the full evaluation raw dataset again (without sharding) to get all references
        eval_raw_full = builder.generate_training_data(eval_docs)
        if args.start_index is not None or args.end_index is not None:
            start = args.start_index if args.start_index is not None else 0
            end = args.end_index if args.end_index is not None else len(eval_raw_full)
            eval_raw_full = eval_raw_full.select(range(start, end))
            
        if config.training.format.value == "instruction":
            all_references = eval_raw_full["assistant"]
        else:
            all_references = [t.split("### Output:\n")[1] for t in eval_raw_full["text"]]
        
        logger.info("Computing metrics...")
        
        if config.training.task_mode.value in ["joint", "ner_only"]:
            ner_res = compute_ner_metrics(all_predictions, all_references)
            logger.info(f"NER Metrics: {json.dumps(ner_res, indent=2)}")
            
        if config.training.task_mode.value in ["joint", "re_only"]:
            is_pairwise = (config.training.re_mode.value == "pairwise")
            re_res = compute_re_metrics(all_predictions, all_references, pairwise=is_pairwise)
            logger.info(f"RE Metrics: {json.dumps(re_res, indent=2)}")

        logger.info(f"Saving final predictions to {args.output} (JSONL format)...")
        with open(args.output, "w", encoding="utf-8") as f:
            for pred_str in all_predictions:
                # It's already valid JSON or error-wrapped JSON, so we can just write it
                f.write(pred_str + "\n")
        logger.info("Predictions saved successfully!")
        
        # Cleanup part files
        for rank in range(accelerator.num_processes):
            p_file = f"{args.output}.part{rank}"
            if os.path.exists(p_file):
                os.remove(p_file)

if __name__ == "__main__":
    main()
