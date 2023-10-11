"""
MIT License

Copyright (c) 2023 Yale Huang
Email: calvino.huang@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import argparse
import os
import openai

import gradio as gr
from prompt_ui import PromptUI

class ChatbotApp:
    def __init__(self, args):
        os.environ["OPENAI_API_KEY"] = args.api_key
        if args.proxy:
            openai.proxy = eval(args.proxy)

        self.prompt_ui = PromptUI(advanced_mode=args.advanced_mode)

        self.ui = gr.TabbedInterface(
            [self.prompt_ui.ui], 
            ['Prompt']
        )
    def launch(self, *args, **kwargs):
        self.ui.launch(*args, **kwargs)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy settings")
    parser.add_argument("--share", action='store_true',
                        help="Launch app with sharing option")
    parser.add_argument("--advanced_mode", action='store_true', default=False,
                        help="Enable advanced mode")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name or IP address")
 
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = ChatbotApp(args)
    app.launch(share=args.share, server_name=args.server_name)
