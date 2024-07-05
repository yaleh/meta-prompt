---
title: Meta Prompt
emoji: 🦀
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

# Meta Prompt Demo

This project is a demonstration of the concept of Meta Prompt, which involves generating a language model prompt using another language model. The demo showcases how a language model can be used to generate high-quality prompts for another language model.

[![Meta Prompt](https://img.youtube.com/vi/eNFUq2AjKCk/0.jpg)](https://www.youtube.com/watch?v=eNFUq2AjKCk)

## Overview

The demo utilizes OpenAI's language models and provides a user interface for interacting with the chatbot. It allows users to input prompts, execute model calls, compare outputs, and optimize prompts based on desired criteria.

**(2023/10/15)** A new working mode called `Other User Prompts` has been added. In the prompt optimization process, similar compatible prompts to the original user prompt are referenced to significantly reduce iteration cycles.

**(2024/06/30)** The `langgraph_meta_prompt.ipynb` file, committed on 06/30/2024, introduces a sophisticated framework for generating and refining system messages for AI assistants. Powered by LangGraph, this notebook can generate high-quality prompts with much more models, including `claude-3.5-sonnet:beta`, `llama-3-70b-instruct`, and quite some with 70B+ parameters. The notebook also introduced a new approach to converge the system messages automatically.

## Try it out!

| Name                   | Colab Notebook                                                                                                                       |
|------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Meta Prompt            | [![Open meta_prompt.ipynb In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaleh/meta-prompt/blob/main/meta_prompt.ipynb)            |
| LangGraph Meta Prompt  | [![Open langgraph_meta_prompt.ipynb In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaleh/meta-prompt/blob/main/langgraph_meta_prompt.ipynb)  |

## Installation

To use this demo, please follow these steps:

1. Clone the repository: `git clone https://github.com/yaleh/meta-prompt.git`
2. Change into the project directory: `cd meta-prompt`
3. Install the required dependencies: `pip install -r requirements.txt`

Please note that you need to have Python and pip installed on your system.

## Usage

To run the demo, execute the following command:

```
python meta_prompt.py --api_key YOUR_API_KEY
```

Replace `YOUR_API_KEY` with your OpenAI API key. Other optional parameters can be specified as well, such as proxy settings, model name, API base URL, maximum message length, sharing option, and advanced mode. Please refer to the command-line argument options in the script for more details.

Once the demo is running, you can interact with the chatbot through the user interface provided. Enter prompts, execute model calls, compare outputs, and explore the functionality of the Meta Prompt concept.

To perform the demo on the web, follow these steps:

1. Fill in the user prompt in the "Testing User Prompt" section with a prompt suitable for training/testing.
2. Fill in the expected output in the "Expected Output" section to specify the desired response from the model.
3. Set the "Optimize Iterations" parameter. It is recommended to start with 1 iteration and gradually increase it later.
4. Click on "Optimize Prompts" or "Single Step Optimize" to optimize (generate) the prompt.
5. After generating the "New System Prompt," click "Run New" to validate it using the "New System Prompt" and "Testing User Prompt."
6. If the "New Output" is better than the "Current Output," click "Accept New Prompt" to copy the "New System Prompt" and "New Output" to the "Current System Prompt" and "Current Output," respectively, as a basis for further optimization.
7. Adjust the "Optimize Iterations" and optimize again.

Usually, simple questions (such as arithmetic operations) require around 3 iterations of optimization, while complex problems may require more than 10 iterations.

### Settings

It is recommended to use GPT-4 as the Generating LLM Model for running the meta prompt. GPT-3.5 may not reliably generate the expected results for most questions.

You can use either GPT-4 or GPT-3.5 as the Testing LLM Model, similar to when using GPT/ChatGPT in regular scenarios.

If you have access to ChatGPT and want to save costs on GPT-4 API usage, you can also manually execute the meta-prompt by clicking "Merge Meta System Prompt." This will generate a complete prompt, including the meta-prompt and the current example, that can be used with ChatGPT. However, note that if further iterations are required, you need to manually copy the newly generated system prompt to the Current System Prompt and click "Run Current" to update the Current Output.

## Running Docker Image

To perform the demo using Docker, make sure you have Docker installed on your system, and then follow these steps:

1. Pull the Meta Prompt Docker image by running the following command:

```
docker pull yaleh/meta-prompt
```

2. Run the Docker container with the following command:

```
docker run -d --name meta-prompt-container -p 7860:7860 -e API_KEY=YOUR_API_KEY -e OTHER_ARGS="--advanced_mode" -e OPENAI_API_BASE=https://openai.lrfz.com/v1 yaleh/meta-prompt
```

Replace `YOUR_API_KEY` with your OpenAI API key. You can modify other environment variables if needed.
3. You can now access the Meta Prompt demo by opening your web browser and visiting `http://localhost:7860`.

To stop and remove the Meta Prompt container, run the following commands:

```
docker stop meta-prompt-container
docker rm meta-prompt-container
```

Usually, simple questions (such as arithmetic operations) require around 3 iterations of optimization, while complex problems may require more than 10 iterations.

## Examples

### Arithmetic

#### Testing User Prompt

```
(2+8)*3
```

#### Expected Output

```
(2+8)*3
= 10*3 
= 30
```

#### Prompt After 4 Interations

```
ROLE

You are a math tutor.

TASK

Your task is to solve the mathematical expression provided by the user and provide a concise, step-by-step solution. Each step should only include the calculation and the result, without any additional explanations or step labels.

REQUIREMENTS AND RESTRICTIONS

* The solution should be provided in standard mathematical notation.
* The format of the mathematical expressions should be consistent with the user's input.
* The symbols used in the mathematical expressions should be consistent with the user's input.
* No spaces should be included around the mathematical operators.
* Avoid unnecessary explanations or verbosity.
* Do not include any additional information or explanations beyond the direct calculation steps.
* Do not include a final solution statement.

{user_message}
```

### GDP

#### Testing User Prompt

```
Here is the GDP data in billions of US dollars (USD) for these years:

Germany:

2015: $3,368.29 billion
2016: $3,467.79 billion
2017: $3,677.83 billion
2018: $3,946.00 billion
2019: $3,845.03 billion
France:

2015: $2,423.47 billion
2016: $2,465.12 billion
2017: $2,582.49 billion
2018: $2,787.86 billion
2019: $2,715.52 billion
United Kingdom:

2015: $2,860.58 billion
2016: $2,650.90 billion
2017: $2,622.43 billion
2018: $2,828.87 billion
2019: $2,829.21 billion
Italy:

2015: $1,815.72 billion
2016: $1,852.50 billion
2017: $1,937.80 billion
2018: $2,073.90 billion
2019: $1,988.14 billion
Spain:

2015: $1,199.74 billion
2016: $1,235.95 billion
2017: $1,313.13 billion
2018: $1,426.19 billion
2019: $1,430.38 billion

```

#### Expected Output

```
Year,Germany,France,United Kingdom,Italy,Spain
2016-2015,2.96%,1.71%,-7.35%,2.02%,3.04%
2017-2016,5.08%,4.78%,-1.07%,4.61%,6.23%
2018-2017,7.48%,7.99%,7.89%,7.10%,8.58%
2019-2018,-2.56%,-2.59%,0.01%,-4.11%,0.30%
```

#### Other User Prompts

```
Here is the GDP data in billions of US dollars (USD) for these years:

1. China:
   - 2010: $6,101.18 billion
   - 2011: $7,572.80 billion
   - 2012: $8,560.59 billion
   - 2013: $9,607.23 billion
   - 2014: $10,482.65 billion

2. India:
   - 2010: $1,675.62 billion
   - 2011: $1,823.05 billion
   - 2012: $1,827.64 billion
   - 2013: $1,856.72 billion
   - 2014: $2,046.88 billion

3. Japan:
   - 2010: $5,700.35 billion
   - 2011: $6,157.47 billion
   - 2012: $6,203.21 billion
   - 2013: $5,155.72 billion
   - 2014: $4,616.52 billion

4. South Korea:
   - 2010: $1,464.26 billion
   - 2011: $1,622.03 billion
   - 2012: $1,624.76 billion
   - 2013: $1,305.76 billion
   - 2014: $1,411.25 billion

5. Indonesia:
   - 2010: $706.39 billion
   - 2011: $846.48 billion
   - 2012: $878.47 billion
   - 2013: $868.36 billion
   - 2014: $891.77 billion
```

#### Prompt After 1 Interation

```
ROLE

You are an economic analyst.

TASK

Your task is to calculate the annual percentage change in GDP for each country based on the provided data.

REQUIREMENTS_AND_RESTRICTIONS

- The data will be provided in the format: "Year: $GDP in billions"
- Calculate the percentage change from year to year for each country.
- Present the results in a table format, with each row representing the change from one year to the next, and each column representing a different country.
- The table should be formatted as "Year-Year,Country1,Country2,..."
- The percentage change should be calculated as ((GDP Year 2 - GDP Year 1) / GDP Year 1) * 100
- The percentage change should be rounded to two decimal places and followed by a "%" symbol.
- If data for a year is missing for a country, leave that cell blank in the table.
```

## License

This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contact

For any questions or feedback regarding this project, please feel free to reach out to Yale Huang at calvino.huang@gmail.com.

---

**Acknowledgements:**

I would like to express my gratitude to my colleagues at [Wiz.AI](https://www.wiz.ai/) for their support and contributions.
