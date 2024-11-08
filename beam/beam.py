import argparse

import logging
logger = logging.getLogger("sillm")

import sillm
import sillm.utils as utils

from sillm.experimental.beam import beam_search

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM using speculative edits.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-b", "--beam_width", type=int, default=4, help="Beam width (default: 4)")
    parser.add_argument("-c", "--min_choices", type=int, default=2, help="Minimum number of choices to generate (default: 2)")
    parser.add_argument("-m", "--max_tokens", type=int, default=1024, help="Max. number of tokens to generate (default: 1024)")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Load models
    model = sillm.load(args.model)

    generate_args = {
        "beam_width": args.beam_width,
        "min_choices": args.min_choices,
        "max_tokens": args.max_tokens
    }

    # Init conversation template
    template = sillm.init_template(model.tokenizer, model.args, args.template)
    conversation = sillm.Conversation(template)

    # Log memory usage
    utils.log_memory_usage()

    # Input loop
    prompt = ""
    while True:
        prompt += input("> ")

        if prompt.startswith('/') or len(prompt) == 0:
            if prompt == "/exit":
                # Exit chat
                break
            elif prompt == "/clear":
                # Clear conversation
                conversation.clear()
            else:
                print("Commands:")
                print("/exit - Exit chat")
                print("/clear - Clear conversation")
            
            prompt = ""
            continue
        elif prompt.endswith('\\'):
            # Continue prompt after line break
            prompt = prompt.rstrip('\\') + "\n"
            continue

        prompt = conversation.add_user(prompt)
        
        logger.debug(f"Generating {args.max_tokens} tokens")

        response, metadata = beam_search(model, prompt, **generate_args)

        conversation.add_assistant(response)
        prompt = ""

        logger.debug(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
        logger.debug(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec)")