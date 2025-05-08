import argparse
import readline
import datetime
import typing

import requests
import colorama

import sillm
import sillm.utils as utils

import agents

WMO_CODES = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Depositing rime fog", 51: "Drizzle: Light intensity", 53: "Drizzle: Moderate intensity", 55: "Drizzle: Dense intensity", 56: "Freezing Drizzle: Light intensity", 57: "Freezing Drizzle: Dense intensity", 61: "Rain: Slight intensity", 63: "Rain: Moderate intensity", 65: "Rain: Heavy intensity", 66: "Freezing Rain: Light intensity", 67: "Freezing Rain: Heavy intensity", 71: "Snowfall: Slight intensity", 73: "Snowfall: Moderate intensity", 75: "Snowfall: Heavy intensity", 77: "Snow grains", 80: "Rain showers: Slight intensity", 81: "Rain showers: Moderate intensity", 82: "Rain showers: Violent intensity", 85: "Snow showers: Slight intensity", 86: "Snow showers: Heavy intensity", 95: "Thunderstorm: Slight or moderate intensity", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"}

def request_api(url):
    """
    Generic request function to call an API and return the JSON response.
    """
    res = requests.get(url)

    if res.ok:
        return res.json()
    else:
        raise Exception(f"API request failed with status code {res.status_code}.")

def search_location(
        name: typing.Annotated[str, "The name of the location to search for."]
        ):
    """
    Get the latitude and longitude for a given location using the Open-Meteo API. Returns the JSON response from the API.
    """
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={name}&count=5&language=en&format=json"
    
    return request_api(url)

def weather_current(
        latitude: typing.Annotated[float, "The latitude of the location."],
        longitude: typing.Annotated[float, "The longitude of the location."]
        ):
    """
    Get the current weather for a given location using the Open-Meteo API. Returns the JSON response from the API.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=weather_code,temperature_2m,precipitation,wind_speed_10m,wind_direction_10m"
    data = request_api(url)
    
    if "current" in data and "weather_code" in data["current"]:
        data["current"]["weather_description"] = WMO_CODES[data["current"]["weather_code"]] if data["current"]["weather_code"] in WMO_CODES else "Unknown"
        del data["current"]["weather_code"]

    return data

def forecast_daily(
        latitude: typing.Annotated[float, "The latitude of the location."],
        longitude: typing.Annotated[float, "The longitude of the location."]
        ):
    """
    Get the daily forecast for a given location using the Open-Meteo API. Returns the JSON response from the API.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=weather_code,temperature_2m_max,temperature_2m_min,sunshine_duration,precipitation_sum,precipitation_probability_max,uv_index_max"
    data = request_api(url)

    if "daily" in data and "weather_code" in data["daily"]:
        data["daily"]["weather_description"] = [WMO_CODES[code] if code in WMO_CODES else "Unknown" for code in data["daily"]["weather_code"]]
        del data["daily"]["weather_code"]
    
    return data

# Define tool functions
tool_functions = {
    "search_location": search_location,
    "weather_current": weather_current,
    "forecast_daily": forecast_daily
}

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for a weather chatbot.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-a", "--input_adapters", default=None, type=str, help="Load and merge LoRA adapter weights from .safetensors file")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("-m", "--max_tokens", type=int, default=8192, help="Max. number of tokens to generate")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Load model
    model = sillm.load(args.model)

    if args.input_adapters is not None:
        # Convert model to trainable
        model = sillm.TrainableLoRA.from_model(model)

        lora_config = model.load_lora_config(args.input_adapters)

        # Initialize LoRA layers
        model.init_lora(**lora_config)

        # Load and merge adapter file
        model.load_adapters(args.input_adapters)
        # model.merge_and_unload_lora()

    # Initialize agent
    agent = agents.ToolAgent(tool_functions)
    
    tool_role = "user"
    if model.args.model_type in ("qwen2", "qwen3"):
        tool_role = "tool"

    # Init conversation template
    template = sillm.init_template(model.tokenizer, model.args, args.template)
    system_prompt = agent.format_system_prompt() + f"\n\nToday is {datetime.datetime.now().strftime('%B %d, %Y')}."
    conversation = sillm.Conversation(template, system_prompt=system_prompt)
    cache = model.init_kv_cache()

    generate_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    # Print system prompt
    print(colorama.Fore.WHITE + system_prompt + colorama.Fore.RESET)

    request = None
    while True:
        if request is None:
            prompt = input("> ")
            if prompt.startswith('/') or len(prompt) == 0:
                if prompt == "/exit":
                    # Exit chat
                    break
                elif prompt == "/clear":
                    # Clear conversation
                    conversation.clear()
                    cache = model.init_kv_cache()
                    agent.reset()
                else:
                    print("Commands:")
                    print("/exit - Exit chat")
                    print("/clear - Clear conversation")
                continue

            request = conversation.add_user(prompt)

        # Generate and print response
        response = ""
        print(colorama.Fore.WHITE, end="")
        for s, metadata in model.generate(request, cache=cache, **generate_args):
            print(s, end="", flush=True)
            response += s
        print(colorama.Fore.RESET)

        conversation.add_assistant(response)

        # Handle response by extracting & executing code blocks and returning the result
        result = agent.handle_response(response)
        if result is not None:
            request = conversation.add_message(result, role=tool_role)

            # Print result
            print(colorama.Fore.RED + result + colorama.Fore.RESET)
        else:
            request = None