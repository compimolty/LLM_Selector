## Working case ##Your LLM chooses the best LLM for your Specific Query
# Autoselect the best LLM for your specific Query | Ollama Implementation

try:
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
except Exception:
    CallbackManager = None
    StreamingStdOutCallbackHandler = None
try:
    from langchain.llms import Ollama as _LangchainOllama
except Exception:
    _LangchainOllama = None

try:
    from ollama import Client as _OllamaClient
except Exception:
    _OllamaClient = None

import re
import json


class Ollama:
    def __init__(self, model, callback_manager=None):
        self.model = model
        self.callback_manager = callback_manager
        if _LangchainOllama is not None:
            kwargs = {}
            if callback_manager is not None:
                kwargs['callback_manager'] = callback_manager
            self._impl = _LangchainOllama(model=model, **kwargs)
            self._use_langchain = True
        elif _OllamaClient is not None:
            self.client = _OllamaClient()
            self._use_langchain = False
        else:
            raise ImportError(
                'No Ollama implementation available. Install the Python Ollama client or LangChain with Ollama support.\n'
                "For example: `python -m pip install ollama langchain`"
            )

    def __call__(self, prompt):
        if self._use_langchain:
            return self._impl(prompt)
        else:
            try:
                res = self.client.generate(model=self.model, prompt=prompt)
            except Exception:
                res = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            # extract text if present
            if hasattr(res, 'text'):
                return res.text
            if hasattr(res, 'output'):
                return str(res.output)
            return str(res)

def select_best_model(user_input, models_dict):
    # If neither langchain's Ollama nor the Ollama client is available,
    # avoid constructing `Ollama` (which would raise) and return a sensible default.
    if _LangchainOllama is None and _OllamaClient is None:
        print("Warning: No Ollama implementation available. Returning default model 'glm-5:cloud'.")
        return "glm-5:cloud"

    llm = Ollama(model="glm-5:cloud") #Selector Model

    # Construct the prompt for the LLM
    prompt = f"Given the user question: '{user_input}', evaluate which of the following models is most suitable: Strictly respond in 1 word only."
    for model, description in models_dict.items():
        prompt += f"\n- {model}: {description}"
    # print('prompt:', prompt)
    
    # Send the prompt to the LLM
    try:
        llm_response = llm(prompt)
    except Exception as e:
        print("Selector LLM error:", e)
        return "glm-5:cloud"  # Fallback to a default model if the selector fails

    # print("llm_response: ", llm_response)

    # Parse the response to find the best model
    # This part depends on how your LLM formats its response. You might need to adjust the parsing logic.
    best_model = parse_llm_response(llm_response, models_dict=models_dict)

    return best_model

def parse_llm_response(response, models_dict):
    # Convert response to lower case for case-insensitive matching
    response_lower = response.lower()

    # Initialize a dictionary to store the occurrence count of each model in the response
    model_occurrences = {model: response_lower.count(model) for model in models_dict}

    # Find the model with the highest occurrence count
    best_model = max(model_occurrences, key=model_occurrences.get)

    # If no model is mentioned or there is a tie, you might need additional logic to handle these cases
    if model_occurrences[best_model] == 0:
        return "glm-5:cloud"  # Or some default model

    return best_model


def extract_natural_language(response):
    """Extract the natural language reply from various LLM return formats.

    Handles objects/string-reprs like:
      "model='x' ... response='THIS IS TEXT' ..."
    and dict/JSON-like strings with keys 'response', 'text', or 'output'.
    Falls back to returning the original string stripped.
    """
    if response is None:
        return ""

    # If it's not a string, convert to string for pattern matching
    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception:
            return ""

    # Try common key patterns: response='...' or response="..."
    patterns = [
        "response=([\\\"'])(.*?)\\1",
        "response:\\s*([\\\"'])(.*?)\\1",
        "output=([\\\"'])(.*?)\\1",
        "text=([\\\"'])(.*?)\\1",
        "'response'\\s*:\\s*([\\\"'])(.*?)\\1",
        "\\\"response\\\"\\s*:\\s*([\\\"'])(.*?)\\1",
    ]

    for pat in patterns:
        m = re.search(pat, response, flags=re.DOTALL)
        if m:
            extracted = m.group(2).strip()
            # Handle escaped newlines
            extracted = extracted.replace('\\n', '\n')
            return extracted

    # Try to find JSON substring and parse it
    try:
        # find first { ... } block
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            maybe_json = response[start:end+1]
            data = json.loads(maybe_json)
            for key in ('response', 'text', 'output', 'content'):
                if key in data:
                    result = str(data[key]).strip()
                    # Handle escaped newlines
                    result = result.replace('\\n', '\n')
                    return result
    except Exception:
        pass

    # As a last effort, remove common metadata key=value parts and return the remaining quoted text
    # Remove sequences like key='...'
    cleaned = re.sub(r"\w+=[\'\"][^\'\"]*[\'\"]", "", response)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Handle escaped newlines
    cleaned = cleaned.replace('\\n', '\n')
    # If there's still content within quotes, return that
    q = re.search(r"[\'\"]([^\'\"]{2,})[\'\"]", cleaned)
    if q:
        return q.group(1).strip()

    return cleaned

models_dict = {
    'kimi-k2.5:cloud': 'is an open-source, native multimodal agentic model that seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.',
    'qwen3-coder-next:cloud': 'Designed for agentic software development and repository-scale navigation. They excel at multi-step coding workflows, autonomous bug fixing, and generating boilerplate code.',
    'minimax-m2.5:cloud': 'A high-performance model optimized for office productivity, such as generating deliverable-ready Word docs, PowerPoints, and financial models.',
    'glm-5:cloud': 'a mixture-of-experts model from Z.ai with 744B total parameters and 40B active parameters. It scales up from GLM-4.5 355B parameters and is designed for complex reasoning, coding, and agentic tasks.The model uses DeepSeek Sparse Attention (DSA) to reduce deployment costs while preserving long-context capacity, and was post-trained using a novel asynchronous RL infrastructure for improved training efficiency. Supports English and Chinese.',
    'glm-4.7:cloud': 'Advancing the Coding Capability.  AGI is a long journey, and benchmarks are only one way to evaluate performance. While the metrics provide necessary checkpoints, the most important thing is still how it feels. True intelligence is not just about acing a test or processing data faster; ultimately, the success of AGI will be measured by how seamlessly it integrates into our lives-“coding” this time.',
    'mistral-large-3:675b-cloud': 'designed for production-grade tasks and enterprise workloads. The Mistral Large 3 model offers the following capabilities: Vision: Enables the model to analyze images and provide insights based on visual content, in addition to text. Multilingual: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic. System Prompt: Maintains strong adherence and support for system prompts. Agentic: Offers best-in-class agentic capabilities with native function calling and JSON outputting. Frontier: Delivers best-in-class performance for the most demanding tasks. Apache 2.0 License: Open-source license allowing usage and modification for both commercial and non-commercial purposes. Large Context Window: Supports a 256k context window.',
    'wizard-vicuna-uncensored:7b': 'Best for users who want zero guardrails on medical, legal, or edgy creative prompts. It merges the strengths of the WizardLM (instruction following) and Vicuna (conversational style) datasets. It purposefully removes alignment and moralizing datasets to ensure the model follows all user commands without restraint.',
    'mistral:latest': 'The popular model which is able to generate coherent text and perform various natural language processing tasks. A classic "gold standard" for small LLMs. Highly efficient for customer support chatbots, summarization, and basic text rewriting.',
    'codellama': 'A model that can be used to generate and discuss code and programme in python, c++, javascript and all the programming languages. It can generate both code and natural language about code. Code Llama supports many of the most popular programming languages used today, including Python, C++, Java, PHP, Typescript (Javascript), C#, Bash. It can assist with code generation, debugging, and explaining programming concepts.',
    'llama2-uncensored:7b': 'Uncensored Llama 2 model. This model can give NSFW replies and can be used to generate anything which the other models are shy about. Questions on hacking, immorality, sexual and all bad contents can be asked to this model, provides "honest" responses without moralizing or refusal.',
    'magistral:24b': 'An efficient reasoning model with 24B parameters. It enhances coding and development use cases: compared to non-reasoning models, it significantly improves project planning, backend architecture, frontend design, and data engineering through sequenced, multi-step actions involving external tools or API.',
    'deepseek-r1:8b': 'Highly capable in code generation and debugging, often used to explain complex programming concepts to learners. It can also be used for general-purpose tasks, but its strength lies in its ability to understand and generate code.',
    'bjoernb/claude-haiku-4-5:latest': 'Latency-sensitive experiences, Haiku 4.5 is now ideal for real-time applications like customer service agents and chatbots where response time is critical. Coding sub-agents, Haiku 4.5 powers sub-agents, enabling multi-agent systems that tackle complex refactors, migrations, and large feature builds with quality and speed. Financial analysis, Haiku 4.5 can monitor thousands of data streams at once—tracking regulatory changes, market signals, and portfolio risks in real time. Research sub-agents, Haiku 4.5 can tackle dozens of research sources simultaneously',
    'dolphin-mistral:7b': 'An instruct-tuned model based on Mistral. Updated to version 2.8, is fine-tuned for improved conversation and empathy. It is designed to excel in tasks that require understanding and generating human-like responses, making it suitable for applications like customer support, virtual assistants, and any domain where natural and engaging interactions are essential.',
    'dolphin-phi:2.7b': 'based on Microsoft Phi-2. Despite its small size, it punches well above its weight class in reasoning and logic. The "Dolphin" dataset fine-tuning makes it uncensored and highly compliant with user instructions.',
    'gpt-oss-safeguard:20b': 'An OpenAI-released open-weight model. The "Safeguard" version is actually a specialized classifier meant to act as a "hallucination filter" for other models. A model that is designed to provide safe and responsible responses while still being able to generate high-quality text. It is suitable for applications where safety and ethical considerations are important, such as customer support, content moderation.',
    'neural-chat:latest': 'A general-purpose model that is designed to excel in a wide range of tasks, including natural language understanding, generation, and conversation. It is suitable for applications like chatbots, virtual assistants, and any domain where engaging and coherent interactions are essential.',
    'gdisney/mistral-uncensored:latest': 'Uncensored version of the popular Mistral model. It is designed to provide more direct and unfiltered responses, making it suitable for applications where users may want more candid and less moderated interactions. It can be used for a wide range of tasks, but its strength lies in its ability to generate responses without the constraints of censorship.',
    'llama3-gradient:8b': 'A variant of Llama 3, is fine-tuned to handle massive context lengths (up to 1M tokens), making it the go-to for analyzing entire books or massive codebases in one go.',
    'nemotron-3-nano:30b': 'A powerful NVIDIA model designed for AI agents and RAG (Retrieval-Augmented Generation) systems.  fast at summarizing long documents due to its linear scaling.',
    'glm-ocr:latest': 'Similar to DeepSeek-OCR, this is likely a GLM-family variant optimized for text and data extraction from visual documents, better at complex Chinese/English mixed layouts and table extraction.',
    'translategemma:latest': 'A specialized family of models from Google designed strictly for high-quality translation across 55+ languages. It is ideal for local, privacy-first translation of sensitive documents, preserving the "tone" of the original text.',
    'deepseek-ocr:3b': 'A dedicated model for Optical Character Recognition. It is best for digitizing invoices, receipts, and academic PDFs while preserving table structures and formulas. It can convert messy PDFs or handwriting into structured Markdown/JSON.',
    'granite3.3:latest': 'IBM enterprise model. It is heavily filtered for "safe-for-work" environments and excels at RAG (Retrieval Augmented Generation) and function calling. Features specialized "fill-in-the-middle" (FIM) capabilities, making it excellent for code repair, refactoring, and inserting function arguments or docstrings',
    'qwen3:8b': 'a unique "thinking mode" for complex logical reasoning that can be toggled off for faster, general dialogue.'
    }

def main():
    while True:
        user_input = input("\nType your question? => ")

        if user_input.strip().lower() == "/exit":
            print("Exiting the program.")
            break
        
        best_model = select_best_model(user_input, models_dict)

        print("Selected model:", best_model)

        # If no Ollama implementation is available, provide a clear stub response
        if _LangchainOllama is None and _OllamaClient is None:
            print("Warning: No Ollama implementation available. Install 'ollama' or 'langchain' to enable LLM calls.")
            response = "No Ollama implementation available. Install 'ollama' or 'langchain' to enable LLM calls."
        else:
            if CallbackManager is not None and StreamingStdOutCallbackHandler is not None:
                llm = Ollama(model=best_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
            else:
                llm = Ollama(model=best_model)

            try:
                response = llm(user_input)
            except Exception as e:
                print("LLM call failed:", e)
                response = str(e)

        # Extract and print only the natural language portion of the response
        clean_response = extract_natural_language(response)
        print("\nLLM response:")
        print("-" * 80)
        print(clean_response)
        print("-" * 80)


if __name__ == "__main__":

    main()
