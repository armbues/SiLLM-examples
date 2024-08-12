import argparse
import json

import mlx.core as mx

import sillm
import sillm.utils as utils
import sillm.experimental.control as control
import sillm.experimental.interpretation as interpretation

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Sparse Autoencoder test")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("sae", type=str, help="The file containing the Sparse Autoencoder weights")
    parser.add_argument("explanations", type=str, help="Explanations for activations")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Initialize model
    model = sillm.load(args.model)
    model = control.ControlledLLM.from_model(model)

    control_index = model.get_module_index(segment="all", transformer=True)
    model.init_control(control_index, mode="output")
    model.capture(mode=True)

    # Initialize Sparse Autoencoder
    sae = interpretation.SparseAutoencoder.load(model.args, args.sae)
    
    with open(args.explanations, "r") as f:
        data = json.load(f)

        explanations = {}
        for entry in data["explanations"]:
            i = int(entry["index"])
            explanations[i] = entry


    def analyze(prompt):
        inputs = model.tokenizer.encode(prompt)

        _ = model.model([inputs])
        res = model._control_modules['layers.20'].hidden_states

        activations = sae.encode(res)

        activations_sum = activations.sum(1)
        top_features = mx.argpartition(-activations_sum, 20, axis=-1)[:,:20]
        top_values = mx.take(activations_sum, top_features)

        print("Top features:")
        for feature, value in zip(top_features[0].tolist(), top_values[0].tolist()):
            description = explanations[feature]["description"]

            print(f"#{feature} ({value}): {description}")

    while True:
        prompt = input("> ")

        if len(prompt) > 0:
            analyze(prompt)
        else:
            break