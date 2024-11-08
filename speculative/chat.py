import argparse
import readline

import sillm
import sillm.utils as utils
from sillm.experimental.speculative import SpeculativeLLM

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM using speculative decoding.")
    parser.add_argument("draft", type=str, help="The input model directory or file")
    parser.add_argument("target", type=str, help="The output model directory or file")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-l", "--lookahead", type=int, default=8, help="Lookahead for draft model")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("-m", "--max_tokens", type=int, default=1024, help="Max. number of tokens to generate")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for chat template")
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
    draft_model = sillm.load(args.draft)
    target_model = sillm.load(args.target)

    # Initialize speculative model
    speculative_model = SpeculativeLLM.from_models(draft_model, target_model)

    generate_args = {
        "lookahead": args.lookahead,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    # Init conversation template
    template = sillm.init_template(target_model.tokenizer, target_model.args, args.template)
    conversation = sillm.Conversation(template, system_prompt=args.system_prompt)

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
        
        logger.debug(f"Generating {args.max_tokens} tokens with temperature {args.temperature}")

        response = ""
        for s, metadata in speculative_model.generate(prompt, **generate_args):
            print(s, end="", flush=True)
            response += s
        print()

        conversation.add_assistant(response)
        prompt = ""

        logger.debug(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
        logger.debug(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec)")
        acceptance_rate = metadata['speculative']['num_accepted'] / metadata['usage']['completion_tokens'] if metadata['usage']['completion_tokens'] > 0 else 0.0
        logger.debug(f"Accepted {metadata['speculative']['num_accepted']}/{metadata['usage']['completion_tokens']} drafted tokens ({acceptance_rate:.2%})")