import logging

import sillm
from sillm.experimental.speculative import SpeculativeEdit

logger = logging.getLogger("sillm")

SYSTEM_PROMPT = """You are an advanced language model tasked with correcting text provided by the user. Your role is to identify and correct only spelling mistakes, grammatical errors, punctuation errors, and other language-related issues (such as verb agreement, tense, sentence structure, or clarity)."""
USER_PROMPT = """Correct the input text below and pay attention to the following guidelines:
- Do not change the original meaning, tone, or style of the text.
- Do not introduce new content, remove information, or make any changes beyond correcting errors.
- Do not alter the formatting, structure, or wording unless it's necessary to fix an error.
- Maintain the same structure and content as the original text.
- Your response should be the full input text with only the errors corrected.
- If no errors are found, respond with the original input text unchanged.
- Do not use any prefix or suffix to indicate the corrections made.

Input text:
"""

if __name__ == "__main__":
    import argparse

    import sillm.utils as utils

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for correcting spelling with SiLLM using speculative edits.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("input", type=str, help="The input text")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-l", "--lookahead", type=int, default=32, help="Lookahead for draft model")
    parser.add_argument("-k", "--key_size", type=int, default=3, help="Key size for resuming drafts")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("-m", "--max_tokens", type=int, default=8192, help="Max. number of tokens to generate")
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

    # Load model
    model = sillm.load(args.model)
    
    # Initialize speculative model
    edit_model = SpeculativeEdit.from_model(model)

    generate_args = {
        "lookahead": args.lookahead,
        "key_size": args.key_size,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    # Init conversation template
    template = sillm.init_template(model.tokenizer, model.args, args.template)

    # Read input text
    with open(args.input, "r") as f:
        input_text = f.read()
    
    # Create prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_PROMPT + input_text
        }
    ]
    prompt = template.apply_chat_template(messages, add_generation_prompt=True)

    for s, metadata in edit_model.generate(prompt, input_text, **generate_args):
        print(s, end="", flush=True)
    print()

    # print(metadata["speculative"]["drafted"])

    logger.info(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
    logger.info(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec)")
    acceptance_rate = metadata['speculative']['num_accepted'] / metadata['speculative']['num_drafted'] if metadata['speculative']['num_drafted'] > 0 else 0.0
    logger.info(f"Accepted {metadata['speculative']['num_accepted']}/{metadata['speculative']['num_drafted']} drafted tokens ({acceptance_rate:.2%})")