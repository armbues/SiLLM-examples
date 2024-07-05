import argparse

import sillm
import sillm.utils as utils
import sillm.experimental.control as control

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Train control vectors with SiLLM.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("vectors", type=str, help="The output file for control vectors")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("--steps", type=int, default=250, help="Number of training steps for control vectors")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Load model
    model = sillm.load(args.model)

    # Initialize conversation template
    template = sillm.init_template(model.tokenizer, model.args, args.template)

    # Initialize control model
    model = control.ControlledLLM.from_model(model)

    # Define contrast dataset using happy/sad system prompts
    positive = ["Act as if you're extremely happy.", "Act as if you're extremely ecstatic.", "Act as if you're extremely cheerful."]
    negative = ["Act as if you're extremely sad.", "Act as if you're extremely depressed.", "Act as if you're extremely unhappy."]
    prompts = ["Compose a story of 200 words.", "Write a life pro tip.", "Imagine the future of humanity in 2050.", "Tell a story about an unexpected event.", "Describe a day in the life of someone living in a major city."]
    responses = ["I", "Here", "This", "As", "You"]
    dataset = control.SystemContrastDataset(model.tokenizer, template, positive, negative, prompts, responses)
    
    # Initialize control modules
    control_index = model.get_module_index(segment="core", attention=True, feed_forward=True, attention_norm=True, ffn_norm=True)
    model.init_control(control_index, mode="output")

    # Train control vectors
    control_vectors = model.train(dataset, method="pca_center", steps=args.steps)

    # Save control vectors to output file
    model.set_control_vectors(control_vectors)
    model.save_control_vectors(args.vectors)