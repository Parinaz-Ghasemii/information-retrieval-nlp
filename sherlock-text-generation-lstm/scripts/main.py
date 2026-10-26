import sys
import os
import json
import argparse
import traceback

# --- This is the crucial part that fixes the imports ---
# Add the project's root directory to the Python path
# This allows us to import from the 'src' folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# ---------------------------------------------------------

from src.config import ProjectConfig
from src.preprocessing import run_preprocessing_pipeline
from src.train import ModelTrainer
from scripts.generate import TextGenerator
from scripts.evaluation import ModelEvaluator
from src import utils

def preprocess(config: ProjectConfig):
    """Runs the data preprocessing pipeline."""
    print("--- Step 1: Data Preprocessing ---")
    try:
        run_preprocessing_pipeline(config.data)
        print("✅ Preprocessing completed successfully.")
    except Exception as e:
        print(f"❌ Preprocessing Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def train(config: ProjectConfig):
    """Trains the model."""
    print("\n--- Step 2: Model Training ---")
    experiment_dir = utils.get_experiment_dir(config)
    model_checkpoint_path = os.path.join(experiment_dir, "best_model.pth")

    if os.path.exists(model_checkpoint_path):
        print(f"✅ Model checkpoint already exists at '{model_checkpoint_path}'. Skipping training.")
        return

    try:
        trainer = ModelTrainer(config)
        trainer.train()
        print("✅ Training completed successfully.")
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        traceback.print_exc()
        sys.exit(1)

def evaluate(config: ProjectConfig):
    """Evaluates the trained model."""
    print(f"\n--- Step 3: Evaluating model from '{config.experiment_name}'... ---")
    try:
        evaluator = ModelEvaluator(config)
        perplexity = evaluator.calculate_perplexity()
        d1, d2 = evaluator.calculate_diversity_from_samples()

        print(f"  - Perplexity on validation set: {perplexity:.4f}")
        print(f"  - Distinct-1 on generated sample: {d1:.4f}")
        print(f"  - Distinct-2 on generated sample: {d2:.4f}")
        print("✅ Evaluation completed successfully.")
        return perplexity, d1, d2
    except Exception as e:
        print(f"❌ Evaluation Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

def generate(config: ProjectConfig):
    """Generates text using the trained model."""
    print(f"\n--- Step 4: Generating text samples with model '{config.experiment_name}'... ---")
    try:
        generator = TextGenerator(config)
        sample_prompts = [
            "holmes turned to me and said",
            "it was a foggy night on baker street when a frantic knock echoed",
        ]
        
        for i, prompt in enumerate(sample_prompts):
            print(f"  - Generating for prompt: '{prompt}'")
            generated_text = generator.generate(prompt)
            
            # Save generated text
            results_dir = os.path.join(utils.RESULTS_DIR, config.experiment_name)
            samples_dir = os.path.join(results_dir, "generated_samples")
            os.makedirs(samples_dir, exist_ok=True)
            with open(os.path.join(samples_dir, f"sample_{i+1}.txt"), "w", encoding="utf-8") as f:
                f.write(generated_text)
        print("✅ Text generation completed successfully.")

    except Exception as e:
         print(f"❌ Generation Failed: {e}")
         traceback.print_exc()
         sys.exit(1)

def run_full_pipeline(config: ProjectConfig):
    """Runs the entire ML pipeline in sequence."""
    preprocess(config)
    train(config)
    perplexity, d1, d2 = evaluate(config)
    generate(config)
    
    # --- Step 5: Final Report ---
    print("\n--- Step 5: Generating Final Report ---")
    results_dir = os.path.join(utils.RESULTS_DIR, config.experiment_name)
    samples_dir = os.path.join(results_dir, "generated_samples")
    
    report = {
        "experiment_name": config.experiment_name,
        "metrics": {
            "perplexity_validation_set": perplexity,
            "distinct_1_on_sample": d1,
            "distinct_2_on_sample": d2
        },
        "details": {
            "saved_samples_dir": str(samples_dir)
        }
    }

    # Save the report to a JSON file
    report_filepath = os.path.join(results_dir, "evaluation_report.json")
    with open(report_filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    
    print(f"✅ Final report saved to '{report_filepath}'")

    print("\n---  Workflow Summary ---")
    print(f"Experiment: '{config.experiment_name}'")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Distinct-1: {d1:.4f}")
    print(f"Distinct-2: {d2:.4f}")
    print("--------------------------")


def main():
    """Main entry point for the script-based ML pipeline."""
    parser = argparse.ArgumentParser(description="Sherlock Holmes AI Pipeline")
    parser.add_argument(
        'stage',
        nargs='?',
        default='all',
        choices=['preprocess', 'train', 'evaluate', 'generate', 'all'],
        help='Pipeline stage to run (default: all).'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default="sherlock_lstm_light_v1",
        help="Name of the experiment to run."
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Prompt for text generation (only used in the "generate" stage).'
    )
    args = parser.parse_args()
    
    # --- Configuration ---
    config = ProjectConfig(experiment_name=args.experiment_name)

    print(f"--- Starting Workflow for Experiment: '{config.experiment_name}' ---")

    if args.stage == 'preprocess':
        preprocess(config)
    elif args.stage == 'train':
        train(config)
    elif args.stage == 'evaluate':
        evaluate(config)
    elif args.stage == 'generate':
        if not args.prompt:
            print("Error: The 'generate' stage requires a --prompt argument.")
            sys.exit(1)
        
        generator = TextGenerator(config)
        print(f"Generating for: '{args.prompt}'")
        output = generator.generate(args.prompt)
        print(f"Result: {output}")

    elif args.stage == 'all':
        run_full_pipeline(config)

if __name__ == '__main__':
    main()