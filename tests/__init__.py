# # To run this code you need to install the following dependencies:
# # pip install google-genai
#
# import os
# from google import genai
# from google.genai import types
#
#
# def generate():
#     client = genai.Client(
#         api_key="AIzaSyBmZq7_HkbP2ByzII6xV6tBY_1ME9TmA3Q"
#     )
#
#     model = "gemini-3-flash-preview"
#     contents = [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_text(text="你是谁"),
#             ],
#         ),
#     ]
#     generate_content_config = types.GenerateContentConfig(
#         thinking_config=types.ThinkingConfig(
#             thinking_level="LOW",
#         ),
#     )
#
#     for chunk in client.models.generate_content_stream(
#         model=model,
#         contents=contents,
#         config=generate_content_config,
#     ):
#         if text := chunk.text:
#             print(text, end="")
#
# if __name__ == "__main__":
#     generate()
#
#
