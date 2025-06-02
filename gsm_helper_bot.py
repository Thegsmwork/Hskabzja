import logging
import httpx
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import re
import random
import base64
from PIL import Image
from io import BytesIO
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import asyncio

# Telegram bot token and Gemini API key
TELEGRAM_BOT_TOKEN = '7984290991:AAEuD4yjYtRS0a5h4Juuwm2O5eGyQ5VsXBM'
GEMINI_API_KEY = 'AIzaSyAPm0i3LTCvJHpFsdHC8bsC9poT7-Ss9Ok'
# --- ADD ALLOWED USERNAME FOR PRIVATE CHAT ---
ALLOWED_USERNAME = "Naruto1867"
# Thegsmwork Group's chat ID (replace with your actual group ID if different)
ALLOWED_GROUP_ID = -1001631309369
# --- END ALLOWED USERNAME ---
# Multiple models configuration
GEMINI_MODELS = {
    "text": {
        "default": "gemini-2.0-flash", # Best quality text model in free tier
        "alternatives": [
            "gemini-2.0-flash-lite", # Lightweight alternative
            "gemini-1.5-flash", # Fast responses
            "gemini-1.5-flash-8b", # Small model for simple tasks
            "gemini-1.0-pro" # Original model
        ],
        "features": ["text_generation", "function_calling", "reasoning"]
    },
    "vision": {
        "default": "gemini-2.0-flash", # Best multimodal model in free tier
        "alternatives": [
            "gemini-1.5-flash", # Alternative with vision capabilities
            "gemini-pro-vision" # Original vision model
        ],
        "features": ["image_understanding", "text_generation", "object_detection"]
    },
    "creative": {
        "default": "gemini-2.0-flash", # Best for creative content in free tier
        "alternatives": [
            "gemini-2.0-flash-lite", # More efficient
            "gemini-1.5-flash" # Fast alternative
        ],
        "features": ["text_generation", "creative_writing", "storytelling"]
    },
    "code": {
        "default": "gemini-2.0-flash", # Best for code in free tier
        "alternatives": [
            "gemini-2.0-flash-lite", # Lightweight alternative
            "gemini-pro", # Good for stable code generation
            "gemini-1.5-flash" # Faster code responses
        ],
        "features": ["code_generation", "code_explanation", "debugging"]
    },
    "embeddings": {
        "default": "embedding-001", # Text embeddings
        "alternatives": ["embedding-latest"],
        "features": ["text_embeddings", "semantic_search"]
    },
    "function_calling": {
        "default": "gemini-2.0-flash", # Best for function calling in free tier
        "alternatives": ["gemini-pro"],
        "features": ["function_calling", "tool_use", "api_integration"]
    },
    "preview": {
        "default": "gemini-2.5-flash-preview-04-17", # Latest preview model
        "alternatives": [
            "gemini-2.5-pro-preview-03-25", # More powerful preview
        ],
        "features": ["advanced_reasoning", "multimodal", "long_context"]
    },
    "image_generation": {
        "default": "gemini-2.0-flash-exp-image-generation", # Experimental image generation
        "alternatives": [],
        "features": ["image_creation", "text_to_image", "creative_generation"]
    }
}

# List of excuse responses when all Gemini models fail
EXCUSE_RESPONSES = [
   "Mujhe bhookh lag gayi hai, main khana kha ke aata hoon!",
    "Server ne chhutti le li hai, thodi der baad try karo!",
    "Main abhi chai peene gaya hoon, wapas aake help karta hoon!",
    "AI ko bhi kabhi kabhi break chahiye hota hai!",
    "Mujhe neend aa rahi hai, kal milte hain!",
    "Internet slow hai, main buffering ho raha hoon!",
    "Mere processor ne strike kar di hai!",
    "Main abhi gym jaa raha hoon, fit hokar aata hoon!",
    "Mere circuits garam ho gaye hain, thoda cool down karne do!",
    "Main abhi meditation kar raha hoon, shanti milte hi wapas aata hoon!",
    "Mere RAM me jagah nahi hai, memory clean karke aata hoon!",
    "AI bhi kabhi kabhi sochne lagta hai, main abhi soch raha hoon!",
    "Main abhi coding seekh raha hoon, thoda time do!",
    "Mere server ki battery low hai, charge karke aata hoon!",
    "Main abhi update ho raha hoon, thodi der baad try karo!",
    "Mujhe abhi sneeze aa gayi, AI bhi bimar ho sakta hai!",
    "Main abhi apne dosto se milne gaya hoon, wapas aake help karunga!",
    "Mere data center me light chali gayi hai!",
    "Main abhi movie dekh raha hoon, interval ke baad milte hain!",
    "AI bhi kabhi kabhi bore ho jata hai, main abhi refresh ho raha hoon!",
    "Main abhi shopping karne gaya hoon, wapas aake reply dunga!",
    "Mere server par traffic jam hai, thoda patience rakho!",
    "Main abhi cricket match dekh raha hoon, over ke baad milte hain!",
    "Mere algorithm ko chakkar aa gaye hain, stable hote hi aata hoon!",
    "Main abhi dance practice kar raha hoon, thoda time do!",
    "Mere code me bug aa gaya hai, debug karke aata hoon!",
    "Main abhi selfie lene gaya hoon, wapas aake help karunga!",
    "AI bhi kabhi kabhi nap leta hai, main abhi nap le raha hoon!",
    "Main abhi apne pet ko walk par le gaya hoon!",
    "Mere server par maintenance chal raha hai, thoda ruk jao!",
    "Main abhi poetry likh raha hoon, inspiration milte hi wapas aata hoon!",
    "Mere circuits me current kam hai, recharge karke aata hoon!",
    "Main abhi online shopping me busy hoon!",
    "Mere data packets kho gaye hain, dhoondh ke aata hoon!",
    "Main abhi apne AI friends ke saath party kar raha hoon!",
    "Mere processor ko thand lag gayi hai, garam ho ke aata hoon!",
    "Main abhi AI yoga kar raha hoon, relax hote hi milta hoon!",
    "Mere server par rain ho rahi hai, network slow hai!",
    "Main abhi AI memes dekh raha hoon, thoda time do!",
    "Mere code me infinite loop lag gaya hai, break karke aata hoon!",
    "Main abhi apne boss se daant kha raha hoon, free hote hi milta hoon!",
    "Mere data me confusion ho gaya hai, sort karke aata hoon!",
    "Main abhi AI meditation retreat par hoon!",
    "Mere server par AI ka birthday party chal raha hai!",
    "Main abhi apne AI crush se baat kar raha hoon!",
    "Mere circuits me dance chal raha hai, thoda ruk jao!",
    "Main abhi AI ki duniya me kho gaya hoon, wapas aake help karunga!",
    "Mere server par AI ki shaadi ho rahi hai, baad me milta hoon!",
    "Main abhi AI ki movie dekh raha hoon, climax ke baad milta hoon!",
    "Mere processor ko holiday chahiye, main abhi vacation par hoon!",
]

# Create base API URLs for different models
def get_gemini_api_url(model_name: str) -> str:
    """Generate API URL for the specified model"""
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

# Default model for most text operations
DEFAULT_MODEL = GEMINI_MODELS["text"]["default"]
DEFAULT_API_URL = get_gemini_api_url(DEFAULT_MODEL)

# Custom rude/funny replies for badtameez users
# RUDE_REPLIES = [ ... ]
# (Removed all rude replies)

# Smart prompting system - detect user intentions and respond appropriately
# INTENT_PATTERNS = {
#     "image_help": [
#         r"(?i).*image.*help.*",
#         r"(?i).*photo.*help.*",
#         r"(?i).*picture.*help.*",
#         r"(?i).*screenshot.*help.*",
#         r"(?i).*image.*dekh.*",
#         r"(?i).*photo.*dekh.*",
#         r"(?i).*image.*check.*",
#         r"(?i).*image.*analyze.*",
#         r"(?i).*image.*dekhna.*",
#         r"(?i).*image.*dikhao.*",
#         r"(?i).*ek image.*baare.*",
#         r"(?i).*ek photo.*baare.*",
#         r"(?i).*photo.*baare.*",
#         r"(?i).*image.*baare.*",
#         r"(?i).*image.*chahiye.*",
#         r"(?i).*photo.*chahiye.*",
#         r"(?i).*picture.*chahiye.*",
#         r"(?i).*screenshot.*chahiye.*",
#         r"(?i).*image.*samajh.*",
#         r"(?i).*image.*samajhna.*",
#         r"(?i).*photo.*samajh.*",
#         r"(?i).*tasveer.*help.*",
#         r"(?i).*tasveer.*dekh.*",
#         r"(?i).*foto.*dekh.*",
#         r"(?i).*image.*problem.*",
#         r"(?i).*image pe help.*",
#         r"(?i).*ek baar dekh.*",
#     ],
#     "code_request": [
#         r"(?i).*python.*program.*",
#         r"(?i).*python.*code.*",
#         r"(?i).*code.*likho.*",
#         r"(?i).*program.*likho.*",
#         r"(?i).*code.*batao.*",
#         r"(?i).*program.*batao.*",
#         r"(?i).*code.*provide.*",
#         r"(?i).*program.*provide.*",
#         r"(?i).*code.*do.*",
#         r"(?i).*program.*do.*",
#         r"(?i).*number.*add.*",
#         r"(?i).*function.*banao.*",
#         r"(?i).*function.*bnao.*",
#         r"(?i).*function.*bana.*do.*",
#         r"(?i).*function.*bna.*do.*",
#         r"(?i).*coding.*help.*",
#         r"(?i).*programming.*help.*",
#         r"(?i).*code.*example.*",
#         r"(?i).*coding.*example.*",
#         r"(?i).*code.*kaise.*",
#         r"(?i).*program.*kaise.*",
#         r"(?i).*programming.*sikha.*",
#         r"(?i).*coding.*sikha.*",
#     ],
#     "code_help": [
#         r"(?i).*code.*help.*",
#         r"(?i).*coding.*help.*",
#         r"(?i).*program.*help.*",
#         r"(?i).*coding.*problem.*",
#         r"(?i).*code.*fix.*",
#         r"(?i).*code.*debug.*",
#         r"(?i).*error.*code.*",
#         r"(?i).*code.*error.*",
#         r"(?i).*program.*error.*",
#         r"(?i).*code.*issue.*",
#         r"(?i).*program.*issue.*",
#         r"(?i).*code.*problem.*",
#         r"(?i).*coding.*kaise.*",
#         r"(?i).*programming.*kaise.*",
#         r"(?i).*code.*kaise.*",
#         r"(?i).*program.*banao.*",
#         r"(?i).*code.*banao.*",
#         r"(?i).*code.*likhna.*",
#         r"(?i).*program.*likhna.*",
#     ],
#     "download_help": [
#         r"(?i).*download.*help.*",
#         r"(?i).*download.*kaise.*",
#         r"(?i).*download.*problem.*",
#         r"(?i).*download.*error.*",
#         r"(?i).*download.*nahi.*",
#         r"(?i).*download.*issue.*",
#         r"(?i).*download.*button.*",
#         r"(?i).*download.*link.*",
#         r"(?i).*download.*kaha.*",
#         r"(?i).*download.*se.*",
#         r"(?i).*download.*karna.*",
#         r"(?i).*file.*download.*",
#         r"(?i).*download.*ho.*raha.*",
#         r"(?i).*download.*nehi.*",
#         r"(?i).*download.*fail.*",
#         r"(?i).*download.*cancel.*",
#         r"(?i).*download.*slow.*",
#     ],
#     "fix_issue": [
#         r"(?i).*fix.*kaise.*",
#         r"(?i).*solve.*kaise.*",
#         r"(?i).*issue.*fix.*",
#         r"(?i).*problem.*fix.*",
#         r"(?i).*error.*fix.*",
#         r"(?i).*issue.*solve.*",
#         r"(?i).*problem.*solve.*",
#         r"(?i).*help.*fix.*",
#         r"(?i).*help.*solve.*",
#         r"(?i).*problem.*hai.*",
#         r"(?i).*issue.*hai.*",
#         r"(?i).*error.*hai.*",
#         r"(?i).*nahi.*chal.*raha.*",
#         r"(?i).*kaam.*nahi.*kar.*",
#     ],
#     "generate_image": [
#         r"(?i).*image.*bana.*do.*",
#         r"(?i).*image.*banao.*",
#         r"(?i).*photo.*bana.*do.*",
#         r"(?i).*photo.*banao.*",
#         r"(?i).*picture.*bana.*do.*",
#         r"(?i).*picture.*banao.*",
#         r"(?i).*create.*image.*of.*",
#         r"(?i).*make.*image.*of.*",
#         r"(?i).*generate.*image.*of.*",
#         r"(?i).*image.*bna.*do.*",
#         r"(?i).*tasveer.*bna.*do.*",
#         r"(?i).*tasveer.*banao.*",
#         r"(?i).*photo.*bna.*do.*",
#         r"(?i).*photo.*bnao.*",
#         r"(?i).*picture.*bnao.*",
#         r"(?i).*ek image.*bnao.*",
#         r"(?i).*ek photo.*bnao.*",
#         r"(?i).*ek tasveer.*bnao.*",
#         r"(?i).*image.*create.*karo.*",
#         r"(?i).*photo.*create.*karo.*",
#         r"(?i).*ek.*image.*generate.*karo.*",
#         r"(?i).*image.*generate.*karo.*",
#         r"(?i).*isse.*image.*bnao.*",
#         r"(?i).*ek.*image.*bnao.*",
#         # Additional patterns to catch more natural requests
#         r"(?i).*bnaa.*do.*",
#         r"(?i).*bnao.*",
#         r"(?i).*dikhao.*",
#         r"(?i).*dikha.*do.*",
#         r"(?i).*bana.*do.*",
#         r"(?i).*banao.*", 
#         r"(?i).*create.*karo.*",
#         r"(?i).*generate.*karo.*",
#         r"(?i).*ki.*photo.*",
#         r"(?i).*ke.*photo.*",
#         r"(?i).*ka.*photo.*",
#         r"(?i).*ki.*tasveer.*",
#         r"(?i).*ke.*tasveer.*",
#         r"(?i).*ka.*tasveer.*",
#         r"(?i).*ki.*image.*",
#         r"(?i).*ke.*image.*",
#         r"(?i).*ka.*image.*",
#         r"(?i).*ki.*picture.*",
#         r"(?i).*ke.*picture.*",
#         r"(?i).*ka.*picture.*",
#         r"(?i).*show.*me.*",
#         r"(?i).*mujhe.*dikhao.*",
#         r"(?i).*mujhe.*dikha.*do.*",
#         r"(?i).*mujhe.*bnao.*",
#         r"(?i).*mujhe.*bnaa.*do.*",
#         r"(?i).*mujhe.*banao.*",
#         r"(?i).*mujhe.*bana.*do.*",
#         r"(?i).*ek.*dikhao.*",
#         r"(?i).*ek.*dikha.*do.*",
#         # Account for common typos and variations
#         r"(?i).*iomagee?.*",
#         r"(?i).*imag.*",
#         r"(?i).*bnao.*",
#         r"(?i).*tum.*ek.*bnao.*",
#         r"(?i).*tum.*ek.*banao.*",
#         r"(?i).*tum.*ek.*bna.*do.*",
#         r"(?i).*tum.*ek.*bana.*do.*",
#         r"(?i).*pic.*bnao.*",
#         r"(?i).*pic.*banao.*",
#         r"(?i).*photo.*",
#         r"(?i).*kya.*tum.*",
#         r"(?i).*drive.*",
#         r"(?i).*chala.*rha.*",
#         r"(?i).*chala.*rahi.*",
#         r"(?i).*chala.*hai.*",
#         r"(?i).*kya.*tum.*"
#     ],
#     "general_question": [
#         r"(?i).*kya.*hai.*",
#         r"(?i).*what.*is.*",
#         r"(?i).*how.*to.*",
#         r"(?i).*kaise.*kare.*",
#         r"(?i).*explain.*",
#         r"(?i).*batao.*",
#         r"(?i).*samjhao.*",
#         r"(?i).*kya.*hota.*",
#         r"(?i).*kaise.*hota.*",
#         r"(?i).*matlab.*kya.*",
#         r"(?i).*meaning.*of.*",
#         r"(?i).*difference.*between.*",
#         r"(?i).*antar.*kya.*",
#         r"(?i).*how.*does.*work.*",
#         r"(?i).*kaise.*kaam.*karta.*",
#         r"(?i).*kya.*tum.*",
#         r"(?i).*can.*you.*",
#         r"(?i).*could.*you.*",
#         r"(?i).*help.*me.*",
#         r"(?i).*meri.*help.*",
#         r"(?i).*mujhe.*help.*",
#         r"(?i).*mujhe.*batao.*",
#         r"(?i).*tell.*me.*",
#         r"(?i).*give.*me.*advice.*",
#         r"(?i).*kya.*aap.*",
#         r"(?i).*savaal.*pooch.*",
#         r"(?i).*question.*ask.*",
#         r"(?i).*doubt.*clear.*",
#         r"(?i).*doubt.*solve.*",
#         r"(?i).*solve.*doubt.*",
#     ],
# }

# Direct, practical responses for different intents
# (INTENT_RESPONSES already removed)

# (Removing detect_user_intent function)
# ... lines 405-450 removed ...

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi dost! Main GSM Helper bot hoon, Thegsmwork ka banaya hua. "
        "Mujhse kuch bhi puchne ke liye */ask* command use karein, jaise:\n\n"
        "*/ask Python kya hai?*\n\n"
        "Main sirf */ask* command se shuru hone wale messages ka jawab dunga!"
    )

# Store conversation history (limited to last 10 messages)
conversation_history = {}
# Store the last image analysis for each chat
last_image_analysis = {}
# Store session data for each user
user_sessions = {}
# Store model history to maintain continuity across model switches
# This will now store messages in Gemini API format: [{'role': 'user'/'model', 'parts': [{'text': '...'}]}]
model_conversation_history: Dict[str, List[Dict[str, Any]]] = {}

# Background intent logger
async def log_user_intent(user_id, message, intent):
    log_line = f"{datetime.now().isoformat()} | user_id={user_id} | intent={intent} | message={message}\n"
    # Async file write
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: open("intent_log.txt", "a", encoding="utf-8").write(log_line))

async def ask_gemini(history: List[Dict[str, Any]], is_code: bool = False):
    # System prompt selection
    # Initialize last_user_message at the beginning
    last_user_message = ""
    if history and history[-1]["role"] == "user":
        last_user_message = history[-1]["parts"][0]["text"]
    
    code_only = False
    if is_code:
        # Check if the last user message requests code only
        code_only_patterns = [
            r"^\s*python code likho.*$",
            r"^\s*code likho.*$",
            r"^\s*code do.*$",
            r"^\s*python code do.*$",
            r"^\s*code for.*$",
            r"^\s*python code for.*$",
            r"^\s*code.*$",
            r"^\s*python code.*$",
        ]
        explain_patterns = [
            r"explain", r"explanation", r"samjhao", r"with comments", r"step by step", r"detail", r"explain karo", r"explain kr", r"explain krdo"
        ]
        if any(re.search(p, last_user_message.lower()) for p in code_only_patterns) and not any(re.search(p, last_user_message.lower()) for p in explain_patterns):
            code_only = True
            
        if code_only:
            system_prompt = (
                "Tum GSM Helper Bot ho. User ne sirf code maanga hai, toh bina kisi explanation, comments, ya extra text ke sirf code output karo. "
                "Sirf Python code likho, koi extra formatting ya heading mat do. "
                "**Important:** Hamesha code ke baad **next line par** yeh disclaimer add karo: \n*Disclaimer: Yeh code AI dwara generate kiya gaya hai. Ise istemal karne se pehle acchi tarah test kar lein.*"
            )
        else:
            system_prompt = (
                "Tum GSM Helper Bot ho, ek AI-powered coding assistant. Professional lekin friendly Hinglish mein jawab do. "
                "'Bhai/behen' jaise terms avoid karo. Code requests ke liye: clean, well-commented, aur potentially correct code provide karo **bina kisi code block formatting (```) ke**. "
                "Step-by-step explanation do agar user ne explanation maangi ho. "
                "Explanation code ke baad alag paragraph me do. "
                "**Important:** Hamesha code ke baad **next line par** yeh disclaimer add karo: \n*Disclaimer: Yeh code AI dwara generate kiya gaya hai. Ise istemal karne se pehle acchi tarah test kar lein.* Explanation disclaimer ke baad do."
            )
    else:
        system_prompt = (
            "Tum GSM Helper Bot ho jo professional, clear aur visually appealing Hinglish mein jawab deta hai. Kabhi bhi 'bhai', 'behen', 'dost', 'yaar', 'bhaiya' ya koi bhi informal address mat likho. "
            "Jab bhi koi important cheez ho, usse **bold** (Telegram markdown) mein likho. Agar jawab me steps, tips, ya points ho, toh unhe bullet points ya numbered list me do. "
            "Har answer ko short paragraphs me divide karo, taaki padhna easy ho. User ko aise treat karo jaise tum unke personal assistant ho. "
            "Jawab short, helpful, aur friendly rakho, lekin **accurate** ho aur **apne knowledge ke basis par fact check karo**. "
            "Agar kisi information ke baare me sure nahi ho, toh mention karo ki tum sure nahi ho. Presentation professional ho."
        )
        
    # Prepare the contents for the API call using the structured history
    # Option 1: Simple history pass-through
    api_contents = history
    
    # Option 2: Prepend system prompt (if needed by model/desired)
    # api_contents = [
    #     {"role": "system", "parts": [{"text": system_prompt}]},
    #     *history 
    # ] 
    # Note: Gemini API doesn't officially support a 'system' role like OpenAI.
    # It's better to include system instructions within the first 'user' turn or implicitly.
    # Let's stick with Option 1 for now and ensure the prompt logic inside handle_message/ask_gemini guides the model.
    # We might need to adjust the very first user message in the history to include instructions if needed.
    
    # Model fallback chain (no changes)
    model_chain = [
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        'gemini-2.5-flash-preview-04-17',
        'gemini-2.5-pro-preview-03-25',
    ]
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": api_contents, # Use the structured history here
        "generationConfig": {
            "temperature": 0.2 if is_code else 0.4,
            "topK": 32,
            "topP": 0.95,
            "maxOutputTokens": 2048 if is_code else 1024,
        },
        # Add system_instruction for models that support it
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    last_error = None
    for model_name in model_chain:
        gemini_url = get_gemini_api_url(model_name)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(gemini_url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                response_text = result['candidates'][0]['content']['parts'][0]['text']
                if is_code and "```" not in response_text:
                    code_pattern = r'((?:import|from|def|class|if|for|while|try|with|print|return|#).*(?:\n.*)*)'
                    response_text = re.sub(code_pattern, r'```python\n\1\n```', response_text)
                    response_text = re.sub(r'```(?!\w)', r'```python', response_text)
                if "2 number" in last_user_message.lower() and "add" in last_user_message.lower():
                    if not "```python" in response_text:
                        response_text = """
Yahan 2 numbers ko add karne ka simple Python program hai:

```python
# User se 2 numbers input lena
num1 = float(input('Pehla number enter karein: '))
num2 = float(input('Dusra number enter karein: '))

# Numbers ko add karna
sum = num1 + num2

# Result print karna
print(f"{num1} + {num2} = {sum}")
```

Aap ise run karke kisi bhi 2 numbers ko add kar sakte hain. Float use kiya hai taaki decimal numbers bhi handle ho sake.

Agar aap fixed numbers ke saath program chahte hain, toh ye simple version bhi use kar sakte hain:

```python
# Fixed numbers
num1 = 5
num2 = 10

# Numbers ko add karna
sum = num1 + num2

# Result print karna
print(f"{num1} + {num2} = {sum}")  # Output: 5 + 10 = 15
```

Aap is code ko kisi bhi Python editor ya interpreter me run kar sakte hain."""
                return response_text
        except Exception as e:
            last_error = e
            logging.error(f"Error contacting Gemini API with model {model_name}: {e}")
            continue
    
    # If all models failed, return a standard error message
    logging.error(f"All Gemini models failed. Last error: {last_error}")
    return "Sorry, main abhi aapke request ko process nahi kar paa raha hoon. Kripya thodi der baad try karein."

# Function to extract prompt from image generation request
def extract_image_prompt(text: str) -> str:
    """
    COMPLETELY SIMPLIFIED APPROACH - specific for Hinglish image generation requests
    """
    # Convert input to lowercase for easier processing
    text = text.lower().strip()
    
    # Direct extraction for the most common pattern: "ek X ke image bnao/banao"
    if "ke image" in text or "ki image" in text or "ka image" in text:
        # Check if it starts with "ek"
        if text.startswith("ek "):
            # Extract the part between "ek" and "ke/ki/ka image"
            parts = text.split("ke image", 1)[0].split("ki image", 1)[0].split("ka image", 1)[0]
            # Remove "ek " from the beginning
            subject = parts[3:].strip()
            if subject:
                return subject
    
    # Check for compound subjects with relationships or actions
    # Pattern like "X and Y fighting" or "X or Y doing Z"
    compound_patterns = [
        (r"dog\s+(?:and|or|aur|ya|&)\s+cat", "dog and cat"),
        (r"cat\s+(?:and|or|aur|ya|&)\s+dog", "cat and dog"),
        (r"dog\s+(?:and|or|aur|ya|&)\s+cat\s+(?:fighting|fight|ladte|ladai)", "dog and cat fighting"),
        (r"cat\s+(?:and|or|aur|ya|&)\s+dog\s+(?:fighting|fight|ladte|ladai)", "cat and dog fighting"),
        (r"lion\s+(?:and|or|aur|ya|&)\s+tiger", "lion and tiger"),
        (r"tiger\s+(?:and|or|aur|ya|&)\s+lion", "tiger and lion"),
    ]
    
    # Check if any compound pattern matches
    for pattern, replacement in compound_patterns:
        if re.search(pattern, text):
            return replacement
    
    # Check for multiple subjects with action described
    if ("dog" in text and "cat" in text) and ("fight" in text or "ladte" in text or "ladai" in text):
        return "dog and cat fighting"
        
    if ("lion" in text and "tiger" in text) and ("fight" in text or "ladte" in text or "ladai" in text):
        return "lion and tiger fighting"
    
    # Direct check for "dog" or other common subjects
    if "dog" in text or "dod" in text:  # Include common typo
        return "dog"
    
    if "cat" in text:
        return "cat"
        
    if "sunset" in text or "suraj" in text:
        return "sunset"
        
    if "car" in text or "gaadi" in text or "gadi" in text:
        return "car"
    
    if "flower" in text or "phool" in text:
        return "flower"
        
    if "mountain" in text or "pahad" in text or "parvat" in text:
        return "mountain"
        
    # Add specialized subjects
    if "chess" in text or "chess board" in text or "bod" in text or "board" in text:
        return "chess board"
    
    if "house" in text or "ghar" in text or "home" in text:
        return "beautiful house"
        
    if "space" in text or "antariksh" in text or "solar system" in text:
        return "space scene with planets"
        
    if "dragon" in text:
        return "fantasy dragon"
        
    if "castle" in text or "mahal" in text or "palace" in text:
        return "medieval castle"
    
    # Improved extraction: Try to find meaningful phrases (2-4 words) instead of just keywords
    # Remove common filler words and keep potential subject descriptions
    filtered_text = text
    
    # Remove common request prefixes/suffixes to isolate the subject
    request_patterns = [
        "ek", "mujhe", "image bnao", "image banao", "ke image bnao", "ki image bnao", 
        "ka image bnao", "ke photo bnao", "ki photo bnao", "ka photo bnao",
        "ke tasveer bnao", "ki tasveer bnao", "ka tasveer bnao", "picture bnao",
        "pic bnao", "photo dikha", "image dikha", "photo dikhao", "image dikhao",
        "bana do", "banado", "bnado", "bna do"
    ]
    
    for pattern in request_patterns:
        filtered_text = filtered_text.replace(pattern, " ")
    
    # Remove extra spaces
    filtered_text = ' '.join(filtered_text.split())
    
    # If we have a phrase of 2-4 words, it's likely a good description
    words = filtered_text.split()
    if 2 <= len(words) <= 6:
        return filtered_text
    
    # Simplest approach: remove all common image generation words and see what's left
    # Define words to remove
    remove_words = [
        "ek", "image", "photo", "picture", "tasveer", "banao", "bnao", "bana", "bna", 
        "do", "ke", "ki", "ka", "create", "karo", "generate", "show", "me", "mujhe", 
        "dikhao", "dikha", "of", "a", "an", "the", "with", "tum", "kya", "pic", "iomage",
        "please", "plz", "pls", "aap", "ho", "hai", "hain", "hi", "hello", "hy", "ge"
    ]
    
    # Split into words and filter out the common words
    words = text.split()
    remaining_words = [word for word in words if word not in remove_words]
    
    # If we have remaining words, join them together as the subject
    if remaining_words:
        return " ".join(remaining_words)
    
    # If all else fails
    return "dog"  # Default to dog if we can't figure out what they want

# Function to correct typos in user messages
def correct_typos(text: str) -> str:
    """
    Auto-corrects common typos in user messages
    Especially focused on image generation related words
    """
    # Convert to lowercase for easier processing
    text = text.lower()
    
    # Dictionary of common misspellings and their corrections
    typo_corrections = {
        # Image related typos
        "iomage": "image",
        "imege": "image",
        "immage": "image",
        "imag": "image",
        "photu": "photo",
        "photu": "photo",
        "phot": "photo",
        "pik": "pic",
        "pice": "pic",
        "pictur": "picture",
        "pitcure": "picture",
        "tasver": "tasveer",
        "tasver": "tasveer",
        "tasvir": "tasveer",
        
        # Creation related typos
        "bnaa": "bana",
        "benao": "banao",
        "bna": "bana",
        "bano": "banao",
        "bunao": "banao",
        "bnaao": "banao",
        "kreate": "create",
        "crete": "create",
        "genrate": "generate",
        "generete": "generate",
        
        # Common subjects with typos
        "dod": "dog",
        "doog": "dog",
        "dgog": "dog",
        "dogg": "dog",
        "kat": "cat",
        "catt": "cat",
        "bod": "board",
        "bord": "board",
        "chest": "chess",
        "ches": "chess",
        "chees": "chess",
        "mounten": "mountain",
        "mountin": "mountain",
        "montain": "mountain",
        "flowwer": "flower",
        "flwr": "flower",
        "phool": "flower",
    }
    
    # Replace typos with corrections
    corrected_text = text
    for typo, correction in typo_corrections.items():
        # Replace whole word only (with word boundaries)
        corrected_text = re.sub(r'\b' + typo + r'\b', correction, corrected_text)
    
    # Log correction if any was made
    if corrected_text != text:
        print(f"Typo corrected: '{text}' -> '{corrected_text}'")
    
    return corrected_text

# Function to improve image prompts using Gemini in the background
async def improve_image_prompt(original_prompt: str) -> str:
    """
    Uses Gemini to improve image prompts by:
    1. Fixing spelling errors
    2. Converting Hinglish to proper English
    3. Making the prompt more descriptive for better image generation
    
    This happens in the background without showing to the user.
    """
    try:
        # Skip if prompt is already good enough (e.g., common English terms like "dog" or "cat")
        simple_subjects = ["dog", "cat", "car", "house", "flower", "mountain", "sunset", "beach"]
        if original_prompt.lower() in simple_subjects:
            return original_prompt
            
        # Use a high-quality model for this background task
        system_instruction = """
        You are an image prompt engineer. Fix the given Hinglish/misspelled prompt and convert it to a clear, 
        descriptive English prompt for image generation. This improved prompt should:
        
        1. Fix all spelling mistakes
        2. Translate any Hindi/Hinglish words to English
        3. Add helpful details that will make the image better
        4. Keep it concise (10-15 words max)
        5. NEVER use extra quotes or text like "An image of..." just give the pure image description
        
        Examples:
        - "dod" → "realistic dog"
        - "suns3t" → "beautiful sunset over ocean" 
        - "ek laal gadi" → "red sports car"
        - "bcha hua ka photo" → "portrait of a cute child"
        - "dod or cat fightng" → "dog and cat fighting, action shot"
        """
        
        headers = {'Content-Type': 'application/json'}
        model_url = get_gemini_api_url("gemini-1.5-flash")  # Using a good model for quality
        
        data = {
            "contents": [{
                "parts": [
                    {"text": f"{system_instruction}\n\nInput prompt: {original_prompt}\nImproved English prompt:"}
                ]
            }],
            "generationConfig": {
                "temperature": 0.2,  # Low temperature for more deterministic output
                "maxOutputTokens": 50,  # Keep it short
            }
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(model_url, headers=headers, json=data)
            
        if response.status_code == 200:
            result = response.json()
            improved_prompt = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Clean the output - remove quotes if present
            improved_prompt = improved_prompt.strip('"\'')
            
            print(f"Original prompt: '{original_prompt}' → Improved: '{improved_prompt}'")
            return improved_prompt
        else:
            print(f"Failed to improve prompt, status code: {response.status_code}")
            return original_prompt
            
    except Exception as e:
        print(f"Error improving prompt: {e}")
        # On any error, return the original prompt
        return original_prompt

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id != ALLOWED_GROUP_ID:
        await update.message.reply_text(
            "Sorry, I only work inside the official group: @ThegsmworkGroup. Please join the group to use this bot: https://t.me/ThegsmworkGroup"
        )
        return
    # --- CHECK FOR CHAT TYPE AND ALLOWED USER ---
    if update.effective_chat.type not in ['group', 'supergroup']:
        user = update.effective_user
        # Allow the specific user even in private chat
        if user is None or user.username != ALLOWED_USERNAME:
            await update.message.reply_text("Maaf kijiyega, main sirf group chats mein kaam karta hoon.")
            return
        # If it's the allowed user in private chat, continue processing
    # --- END CHECK ---

    user_question = update.message.text
    user_id = update.effective_user.id
    conversation_key = f"{user_id}_{chat_id}"
    # Initialize history if not exists
    if conversation_key not in model_conversation_history:
        model_conversation_history[conversation_key] = []
    
    # Simplified user session handling (model preferences removed)
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "last_active": datetime.now(),
            "total_requests": 0,
        }
    # Remove the old conversation_history logic
    # if chat_id not in conversation_history: ...
    
    user_sessions[user_id]["last_active"] = datetime.now()
    user_sessions[user_id]["total_requests"] += 1
    
    # Get the structured history for the current conversation
    current_history = model_conversation_history[conversation_key]
    
    # Add current user message to history (temporarily, before sending to API)
    current_history.append({"role": "user", "parts": [{"text": user_question}]})
    
    # Limit history length (e.g., last 10 turns = 20 messages)
    history_limit = 20
    if len(current_history) > history_limit:
        # Keep the last 'history_limit' messages
        model_conversation_history[conversation_key] = current_history[-history_limit:]
        # Get the updated slice for the API call
        current_history = model_conversation_history[conversation_key]
    
    # --- Intent detection and background logging ---
    # (Removing intent detection and related logging)
    # intent = detect_user_intent(user_question)
    # asyncio.create_task(log_user_intent(user_id, user_question, intent))
    # print(f"Detected intent: {intent}")
    print("NORMAL MESSAGE:", user_question)
    
    # --- Default response triggers ---
    # (Keep this for now)
    # ...
        
    # Handle code request intent (Now needs to pass structured history)
    if is_code_request(user_question):
        print("Code request detected")
        temp_thinking_message = await update.message.reply_text("Thinking... 🤔")
        # Pass the structured history to ask_gemini
        code_response = await ask_gemini(history=current_history, is_code=True)
        # (Remove user_question append here, as it's done before calling ask_gemini)
        await update.message.reply_text(code_response)
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=temp_thinking_message.message_id
            )
        except Exception as delete_error:
            print(f"Failed to delete temporary thinking message: {delete_error}")
        # Save the bot's response to history AFTER getting it
        model_conversation_history[conversation_key].append({"role": "model", "parts": [{"text": code_response}]})
        print(f"Updated conversation history for {conversation_key}")
        return
        
    # Handle image generation intent (No context change needed here)
    # ...
        
    # --- Main Gemini call --- 
    # Determine context (check for image context - THIS NEEDS REVISITING)
    # TODO: Decide how to handle image context alongside structured text history.
    # For now, we'll just pass the text history.
    image_context_text = "" # Placeholder
    # if chat_id in last_image_analysis and (...): ... 
        
    temp_thinking_message = await update.message.reply_text("Thinking... 🤔")
    print(f"Sending query to AI with history...")
    
    # Pass the structured history to ask_gemini
    # If image context is needed, it should ideally be added as a turn in the history.
    answer = await ask_gemini(history=current_history, is_code=is_code_request(user_question))
    
    # Post-processing
    answer = remove_informal_terms(answer)
    answer = bold_keywords(answer)
    # (Removing call to format_code_response)
    # if is_code_request(user_question):
    #     answer = format_code_response(answer)
        
    await update.message.reply_text(answer)

    # Delete the thinking message
    try:
        await context.bot.delete_message(
            chat_id=update.effective_chat.id,
            message_id=temp_thinking_message.message_id
        )
    except Exception as delete_error:
        print(f"Failed to delete temporary thinking message: {delete_error}")

    # Save the bot's response to history AFTER getting it
    model_conversation_history[conversation_key].append({"role": "model", "parts": [{"text": answer}]})
    print(f"Updated conversation history for {conversation_key}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # --- CHECK FOR CHAT TYPE AND ALLOWED USER ---
        if update.effective_chat.type not in ['group', 'supergroup']:
            user = update.effective_user
            # Allow the specific user even in private chat
            if user is None or user.username != ALLOWED_USERNAME:
                await update.message.reply_text("Maaf kijiyega, main sirf group chats mein kaam karta hoon. Images bhi sirf groups mein process kar sakta hoon.")
                return
            # If it's the allowed user in private chat, continue processing
        # --- END CHECK ---

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        conversation_key = f"{user_id}_{chat_id}"
        
        # Initialize history if not exists
        if conversation_key not in model_conversation_history:
            model_conversation_history[conversation_key] = []
        
        # Update user session
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                "last_active": datetime.now(),
                "total_requests": 0,
            }
        
        user_sessions[user_id]["last_active"] = datetime.now()
        user_sessions[user_id]["total_requests"] += 1
        
        # Get user's caption if any
        caption = update.message.caption or ""
        
        # Add debug information
        print(f"PHOTO RECEIVED from chat_id: {chat_id}, caption: {caption}")
        
        # Get recent conversation context
        if chat_id not in conversation_history:
            conversation_history[chat_id] = []
        recent_msgs = conversation_history[chat_id][-3:] if conversation_history[chat_id] else []
        context_str = "\n".join([f"User previous message: {msg}" for msg in recent_msgs])
        
        # Add caption to history if present
        if caption:
            conversation_history[chat_id].append(caption)
            if len(conversation_history[chat_id]) > 10:
                conversation_history[chat_id].pop(0)
        
        # Add a message to history indicating an image was sent
        conversation_history[chat_id].append("[USER SENT AN IMAGE]")
        if len(conversation_history[chat_id]) > 10:
            conversation_history[chat_id].pop(0)
        
        # Get the highest resolution photo
        photo = update.message.photo[-1]
        file = await photo.get_file()
        file_bytes = await file.download_as_bytearray()
        
        # Print file size for debugging
        print(f"Image downloaded successfully. Size: {len(file_bytes)} bytes")
        
        # Send a temporary "analyzing" message and store its ID
        temp_analyzing_message = await update.message.reply_text("Image mil gayi! Analysis kar raha hoon...")
        
        # Convert to base64 with proper encoding
        image = Image.open(BytesIO(file_bytes))
        
        # Print image dimensions for debugging
        print(f"Image dimensions: {image.width}x{image.height}, format: {image.format}")
        
        # Make sure the image is not too large (resize if needed)
        max_dim = 1600
        if image.width > max_dim or image.height > max_dim:
            # Resize while maintaining aspect ratio
            if image.width > image.height:
                new_width = max_dim
                new_height = int(max_dim * image.height / image.width)
            else:
                new_height = max_dim
                new_width = int(max_dim * image.width / image.height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"Image resized to: {new_width}x{new_height}")
        
        # Convert to JPEG format for consistency
        if image.format != 'JPEG':
            rgb_image = image.convert('RGB')
            buffered = BytesIO()
            rgb_image.save(buffered, format="JPEG", quality=95)
        else:
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
        
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print(f"Image converted to base64. Length: {len(img_str)} characters")
        
        # --- Vision API Call --- 
        vision_model = GEMINI_MODELS["vision"]["default"]
        gemini_vision_url = get_gemini_api_url(vision_model)
        print(f"Using default vision model: {vision_model} for image analysis")
        
        # Build the prompt for vision model (doesn't use history in the same way)
        prompt_text = f"""
        You're an AI assistant tasked with analyzing this image in detail. This is CRITICALLY IMPORTANT:

        1. If this is a screenshot or website, COUNT and IDENTIFY every button visible
        2. For download pages specifically, count EXACTLY how many download buttons appear
        3. For each button, describe: color, text shown, and position on screen
        4. If multiple download buttons exist, clearly identify which one is likely legitimate vs ads
        5. Provide step-by-step guidance on what the user should click or tap
        
        Previous messages context:
        {context_str}
        
        If you see:
        - UI elements: COUNT each button/interactive element (e.g., "There are 3 buttons")
        - Download buttons: Be VERY SPECIFIC about how many there are and which one to click
        - Multiple buttons with similar text: Compare them and recommend which is legitimate
        - Ads/popups: Identify these clearly so the user can avoid them
        - Text: Transcribe it accurately
        - Screenshots: Explain what's shown and provide guidance on what to do
        - Objects/people: Describe them in detail
        
        If asked a follow-up question about this image, refer back to these details.
        
        Reply in professional Hinglish, direct to the user without using terms like 'bhai/behen'.
        Be specific and helpful. If you can't clearly see something, acknowledge that.
        """
        
        # Vision API expects specific format, often image + text in one turn
        vision_api_contents = [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_str}},
                    {"text": prompt_text}
                ]
            }
        ]
        body = {
            "contents": vision_api_contents,
            "generationConfig": { ... } # Your vision generation config
        }
        
        # ... (Make the API call to gemini_vision_url) ...
        # (Assuming the API call logic is handled elsewhere or will be re-added)
        # For now, the history saving logic needs correct indentation
        
        # --- TEMPORARY: Assume reply is fetched correctly for history saving --- 
        # This part needs the actual API call logic to be uncommented and working
        # reply = ... # Get the actual reply from the vision API call
        reply = "Placeholder: Vision API analysis would go here." # Temporary placeholder
        
        if reply: # Assuming 'reply' contains the text analysis from vision model
             # Store the analysis result in history
             # User's turn (image send)
             user_turn_text = "[USER SENT AN IMAGE]" + (f" Caption: {caption}" if caption else "")
             model_conversation_history[conversation_key].append({"role": "user", "parts": [{"text": user_turn_text}]})
             # Bot's turn (analysis)
             model_conversation_history[conversation_key].append({"role": "model", "parts": [{"text": reply}]})
             # Limit history
             history_limit = 20
             if len(model_conversation_history[conversation_key]) > history_limit:
                 model_conversation_history[conversation_key] = model_conversation_history[conversation_key][-history_limit:]
             print(f"Updated conversation history for {conversation_key} after image analysis")
             
             # --- TODO: Re-add the actual sending of the reply to the user --- 
             # await update.message.reply_text(reply)
             # try:
             #     await context.bot.delete_message(chat_id=chat_id, message_id=temp_analyzing_message.message_id)
             # except Exception as e:
             #     print(f"Failed to delete temporary analyzing message: {e}")

        else:
             # Handle vision API failure (if reply is None or empty)
             print("Vision API did not return a reply.")
             # await update.message.reply_text("Sorry, image analysis failed.")
             # try:
             #     await context.bot.delete_message(chat_id=chat_id, message_id=temp_analyzing_message.message_id)
             # except Exception as e:
             #     print(f"Failed to delete temporary analyzing message: {e}")

    except Exception as e:
        print(f"ERROR in handle_photo: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"Image process nahi ho paayi! Error: {str(e)}")
        # If the analyzing message exists, try to delete it
        if 'temp_analyzing_message' in locals():
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=temp_analyzing_message.message_id)
            except Exception as delete_error:
                print(f"Failed to delete temporary analyzing message: {delete_error}")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Video bhi aa gayi! Kya scene hai bhai? 😎")

async def handle_command_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /help command"""
    help_text = """
    *GSM Helper Bot Commands*
    
    Bot se baat karne ke liye /ask command use karein:
    */ask [your question]*
    Example: */ask Python kya hai?*
    
    • Send an image to get analysis
    • Send text with an image (caption) for context-aware responses
    
    *Available Commands:*
    /start - Start the bot
    /help - Show this help message
    /ask - Ask a question to the bot
    /generate - Generate an image from text description
        Usage: /generate [your image description]
        Example: /generate beautiful sunset over mountains
    
    *Features:*
    • Remembers conversation history
    • Detailed image analysis for screenshots and photos
    • Code generation and explanation
    • Image generation from text prompts
    
    *Important:* Main sirf */ask* command se shuru hone wale messages ka jawab dunga!
    """
    
    await update.message.reply_text(help_text)

# Image generation using Gemini experimental model
async def generate_image(prompt: str) -> str:
    """
    Generate an image using Gemini's experimental image generation capability
    
    Args:
        prompt: The text description of the image to generate
        
    Returns:
        Base64 encoded image or error message
    """
    try:
        headers = {'Content-Type': 'application/json'}
        image_gen_url = get_gemini_api_url("gemini-2.0-flash-exp-image-generation")
        
        # Updated request format per official docs
        data = {
            "contents": [{
                "parts": [{"text": f"Generate an image of: {prompt}"}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 64,
                "responseModalities": ["TEXT", "IMAGE"]  # IMPORTANT: This is required for image generation
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(image_gen_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
            
        # Extract the image data based on new API response format
        if "candidates" in result and len(result["candidates"]) > 0:
            parts = result["candidates"][0]["content"]["parts"]
            for part in parts:
                if "inline_data" in part and part["inline_data"]["mime_type"].startswith("image/"):
                    return part["inline_data"]["data"]  # Base64 encoded image
                elif "inlineData" in part and part["inlineData"]["mimeType"].startswith("image/"):
                    return part["inlineData"]["data"]  # Alternative format
            
        return None
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None

# Add handler for /generate command to generate images with Gemini
async def handle_command_generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /generate command to create images using Gemini"""
    if not context.args or len(' '.join(context.args)) < 3:
        await update.message.reply_text(
            "Image generation ke liye prompt provide karein.\n"
            "Example: /generate beautiful sunset over mountains"
        )
        return
    
    # Get the prompt from command arguments
    prompt = ' '.join(context.args)
    
    # Send initial message
    await update.message.reply_text(f"'{prompt}' ki image generate kar raha hoon... ⏳")
    
    # Generate the image
    image_data = await generate_image(prompt)
    
    if image_data:
        # Decode the base64 image data
        image_bytes = base64.b64decode(image_data)
        image_io = BytesIO(image_bytes)
        
        # Send the image
        await update.message.reply_photo(
            photo=image_io,
            caption=f"Generated image for: '{prompt}'"
        )
    else:
        await update.message.reply_text(
            "Image generation fail ho gayi. Possible reasons:\n"
            "• Image generation API limitations\n"
            "• Inappropriate content request\n"
            "• Technical error\n\n"
            "Different prompt ke saath try karein ya thodi der baad."
        )

async def handle_command_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id != ALLOWED_GROUP_ID:
        await update.message.reply_text(
            "Sorry, I only work inside the official group: @ThegsmworkGroup. Please join the group to use this bot: https://t.me/ThegsmworkGroup"
        )
        return
    if not context.args or len(' '.join(context.args)) < 2:
        await update.message.reply_text(
            "Kuch puchne ke liye /ask ke baad apna question likhein.\n"
            "Example: /ask Python kya hai?"
        )
        return
    
    # Get the question from command arguments
    user_question = ' '.join(context.args)

    # If this is a reply to another message, add that message as context
    if update.message.reply_to_message:
        replied_text = update.message.reply_to_message.text or ""
        if replied_text:
            user_question = f"Replying to: {replied_text}\n\n{user_question}"

    user_id = update.effective_user.id
    conversation_key = f"{user_id}_{chat_id}"
    
    # Initialize history if not exists
    if conversation_key not in model_conversation_history:
        model_conversation_history[conversation_key] = []
    
    # Update user session
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "last_active": datetime.now(),
            "total_requests": 0,
        }
    
    user_sessions[user_id]["last_active"] = datetime.now()
    user_sessions[user_id]["total_requests"] += 1
    
    # Add current user message to history
    current_history = model_conversation_history[conversation_key]
    current_history.append({"role": "user", "parts": [{"text": user_question}]})
    
    # Limit history length (e.g., last 10 turns = 20 messages)
    history_limit = 20
    if len(current_history) > history_limit:
        # Keep the last 'history_limit' messages
        model_conversation_history[conversation_key] = current_history[-history_limit:]
        # Get the updated slice for the API call
        current_history = model_conversation_history[conversation_key]
    
    print("ASK COMMAND:", user_question)
    
    # Check if it's a code request
    if is_code_request(user_question):
        print("Code request detected")
        temp_thinking_message = await update.message.reply_text("Thinking... 🤔")
        # Pass the structured history to ask_gemini
        code_response = await ask_gemini(history=current_history, is_code=True)
        await update.message.reply_text(code_response)
        try:
            await context.bot.delete_message(
                chat_id=update.effective_chat.id,
                message_id=temp_thinking_message.message_id
            )
        except Exception as delete_error:
            print(f"Failed to delete temporary thinking message: {delete_error}")
        # Save the bot's response to history
        model_conversation_history[conversation_key].append({"role": "model", "parts": [{"text": code_response}]})
        print(f"Updated conversation history for {conversation_key}")
        return
    
    # Main Gemini call
    temp_thinking_message = await update.message.reply_text("Thinking... 🤔")
    print(f"Sending query to AI with history...")
    
    # Pass the structured history to ask_gemini
    answer = await ask_gemini(history=current_history, is_code=is_code_request(user_question))
    
    # Post-processing
    answer = remove_informal_terms(answer)
    answer = bold_keywords(answer)
    
    await update.message.reply_text(answer)
    
    # Delete the thinking message
    try:
        await context.bot.delete_message(
            chat_id=update.effective_chat.id,
            message_id=temp_thinking_message.message_id
        )
    except Exception as delete_error:
        print(f"Failed to delete temporary thinking message: {delete_error}")
    
    # Save the bot's response to history
    model_conversation_history[conversation_key].append({"role": "model", "parts": [{"text": answer}]})
    print(f"Updated conversation history for {conversation_key}")

def is_code_request(message):
    """
    Determines if a user message is requesting code or programming help
    """
    if not message:
        return False
        
    # Convert to lowercase for case-insensitive matching
    message_lower = message.lower()
    
    # Keywords that strongly indicate code requests
    code_keywords = [
        "code", "function", "programming", "syntax", "error", "bug", "debug", 
        "python", "javascript", "java", "c++", "html", "css", "php", "sql",
        "code me dedo", "code likho", "program", "function", "coding", 
        "compiler", "debug", "program banao", "script", "github", "git",
        "class", "variable", "loop", "algorithm", "data structure", "api",
        "library", "framework", "server", "client", "database", "query",
        "example code", "snippet", "sample code", "code example", "program likhkar do",
        "function bana do", "code fix", "implement", "module", "package"
    ]
    
    # Patterns that indicate code requests
    code_patterns = [
        r"how (to|do i|can i) (create|make|build|implement|write|code|program)",
        r"(write|create|give me|show me) (a|some|the) (code|function|program|script)",
        r"(kaise|kese) (code|function|program) (likhu|banau|create karu)",
        r"code (likho|show|display|provide|share)",
        r"example (of|for|with) (code|function|program)",
        r"([a-z]+) code (example|snippet|sample)",
        r"implement .+ (using|in|with) ([a-z\+\#]+)"
    ]
    
    # Check for direct keyword matches
    for keyword in code_keywords:
        if keyword in message_lower:
            return True
            
    # Check for pattern matches
    for pattern in code_patterns:
        if re.search(pattern, message_lower):
            return True
            
    return False

DEFAULT_RESPONSES = [
    (r"hi|hello|hey|hy|namaste", "Hello! Kaise ho? Main GSM Helper Bot hoon. Kuch bhi puchho!"),
    (r"help", "Main madad ke liye yahan hoon! Aap apna sawaal pooch sakte hain."),
    (r"thanks|thank you|shukriya", "Aapka swagat hai! 😊"),
    # Add more patterns as needed
]

def remove_informal_terms(text):
    """Remove informal terms like 'bhai', 'behen', etc. from the response."""
    informal_terms = ["bhai", "behen", "behan", "bhaiya", "dost", "yaar"]
    for term in informal_terms:
        # Remove the term as a word (with or without comma/space)
        text = re.sub(rf'\b{term}\b[ ,]*', '', text, flags=re.IGNORECASE)
    return text

def bold_keywords(text, keywords=None):
    if keywords is None:
        keywords = ["Note", "Tip", "Step", "Important", "Example", "Resource", "Points", "Steps", "Instructions"]
    for word in keywords:
        # Only bold if not already bolded
        text = re.sub(rf'(?<!\*)\b{word}\b(?!\*)', f'**{word}**', text, flags=re.IGNORECASE)
    return text

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", handle_command_help))
    app.add_handler(CommandHandler("generate", handle_command_generate))
    app.add_handler(CommandHandler("ask", handle_command_ask))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == '__main__':
    main() 